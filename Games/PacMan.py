from ASimulator import Simulator
import copy
from numpy.random import random, randint, seed
import numpy as np

MAX_VALUE = 9999999999
MIN_VALUE = -999999999


class PacMan(Simulator):
    INIT_GAME_MAP = [
        ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
        ['x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x'],
        ['x', ' ', 'x', 'x', ' ', 'x', 'x', 'x', ' ', 'x', ' ', 'x', 'x', 'x', ' ', 'x', 'x', ' ', 'x'],
        ['x', 'o', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'o', 'x'],
        ['x', ' ', 'x', 'x', ' ', 'x', ' ', 'x', 'x', 'x', 'x', 'x', ' ', 'x', ' ', 'x', 'x', ' ', 'x'],
        ['x', ' ', ' ', ' ', ' ', 'x', ' ', ' ', ' ', 'x', ' ', ' ', ' ', 'x', ' ', ' ', ' ', ' ', 'x'],
        ['x', 'x', 'x', 'x', ' ', 'x', 'x', 'x', ' ', 'x', ' ', 'x', 'x', 'x', ' ', 'x', 'x', 'x', 'x'],
        ['x', 'x', 'x', 'x', ' ', 'x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x', ' ', 'x', 'x', 'x', 'x'],
        ['x', 'x', 'x', 'x', ' ', 'x', ' ', 'x', ' ', ' ', ' ', 'x', ' ', 'x', ' ', 'x', 'x', 'x', 'x'],
        ['<', ' ', ' ', ' ', ' ', 'x', ' ', 'x', ' ', ' ', ' ', 'x', ' ', 'x', ' ', ' ', ' ', ' ', '>'],
        ['x', 'x', 'x', 'x', ' ', 'x', ' ', 'x', 'x', 'x', 'x', 'x', ' ', 'x', ' ', 'x', 'x', 'x', 'x'],
        ['x', 'x', 'x', 'x', ' ', 'x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x', ' ', 'x', 'x', 'x', 'x'],
        ['x', 'x', 'x', 'x', ' ', 'x', ' ', 'x', 'x', 'x', 'x', 'x', ' ', 'x', ' ', 'x', 'x', 'x', 'x'],
        ['x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x'],
        ['x', ' ', 'x', 'x', ' ', 'x', 'x', 'x', ' ', 'x', ' ', 'x', 'x', 'x', ' ', 'x', 'x', ' ', 'x'],
        ['x', 'o', ' ', 'x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x', ' ', 'o', 'x'],
        ['x', 'x', ' ', 'x', ' ', 'x', ' ', 'x', 'x', 'x', 'x', 'x', ' ', 'x', ' ', 'x', ' ', 'x', 'x'],
        ['x', ' ', ' ', ' ', ' ', 'x', ' ', ' ', ' ', 'x', ' ', ' ', ' ', 'x', ' ', ' ', ' ', ' ', 'x'],
        ['x', ' ', 'x', 'x', 'x', 'x', 'x', 'x', ' ', 'x', ' ', 'x', 'x', 'x', 'x', 'x', 'x', ' ', 'x'],
        ['x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x'],
        ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x']]

    INIT_GHOST_POSES_array = [[8, 9], [8, 10], [9, 9], [9, 10]]
    INIT_PACMAN_POS = [13, 9]
    NORTH = 0
    EAST = 1
    WEST = 2
    SOUTH = 3
    CHASE_PROB = 0.75
    DEFENSIVE_SLIP = 0.25
    NUM_ACTS = 4
    NUM_OBS = 1 << 16
    NUM_RWD = 20

    def Clone(self):
        return PacMan()

    def getNumRewards(self):
        return PacMan.NUM_RWD

    def getNumObservations(self):
        return PacMan.NUM_OBS

    def getNumActions(self):
        return PacMan.NUM_ACTS

    def __init__(self):
        super()
        self.inTerminalState = False
        self.gameMap = None
        self.ghostPoses = None
        self.ghostDirs = None
        self.pacManPos = None
        self.foodLeft = 0
        self.powerPillCounter = 0
        self.observationId = None
        self.currImmediateReward = None

    def isTerminate(self):
        return self.inTerminalState

    def InitRun(self):
        # seed(seedint)
        self.gameMap = copy.deepcopy(PacMan.INIT_GAME_MAP)
        self.ghostPoses = copy.deepcopy(PacMan.INIT_GHOST_POSES_array)
        self.ghostDirs = []
        for i in range(len(PacMan.INIT_GHOST_POSES_array)):
            self.ghostDirs.append(-1)
        self.pacManPos = copy.deepcopy(PacMan.INIT_PACMAN_POS)
        self.inTerminalState = False
        self.foodLeft = 0
        self.powerPillCounter = 0
        self.placeFood()

    def getGameName(self):
        return "PacMan"

    def placeFood(self):
        self.foodLeft = 0
        for lYPos in range(len(self.gameMap) - 1):
            if lYPos == 0:
                continue
            for lXPos in range(len(self.gameMap[lYPos]) - 1):
                if lXPos == 0:
                    continue
                if self.gameMap[lYPos][lXPos] == ' ' and not (6 < lYPos < 12 and 5 < lXPos < 13) and \
                        not (lYPos == self.pacManPos[0] and lXPos == self.pacManPos[1]) and random() < 0.5:
                    self.gameMap[lYPos][lXPos] = '.'
                    self.foodLeft = self.foodLeft + 1

    def getObservation(self):
        return self.getCurrentObservation()

    def getReward(self):
        return self.currImmediateReward

    def getCurrentObservation(self):
        lObsInfo = np.zeros(shape=(16,))
        lYPos = self.pacManPos[0]
        lXPos = self.pacManPos[1]

        # check for wall
        # north
        if self.gameMap[lYPos - 1][lXPos] == 'x':
            lObsInfo[0] = 1
        # east
        if self.gameMap[lYPos][lXPos + 1] == 'x':
            lObsInfo[1] = 1
        # south
        if self.gameMap[lYPos + 1][lXPos] == 'x':
            lObsInfo[2] = 1
        # west
        if self.gameMap[lYPos][lXPos - 1] == 'x':
            lObsInfo[3] = 1

        # check for food smell
        lFoodManhattanDis = self.computeFoodManhattanDist()

        if lFoodManhattanDis <= 2:
            lObsInfo[4] = 1
        elif lFoodManhattanDis <= 3:
            lObsInfo[5] = 1
        elif lFoodManhattanDis <= 4:
            lObsInfo[6] = 1

        # Check see ghosts or food
        # check north
        tempLoc = copy.deepcopy(self.pacManPos)
        tempLoc[0] = tempLoc[0] - 1
        while tempLoc[0] > 0 and self.gameMap[tempLoc[0]][tempLoc[1]] != 'x':
            for lGhost in range(len(self.ghostPoses)):
                if tempLoc[0] == self.ghostPoses[lGhost][0] and tempLoc[1] == self.ghostPoses[lGhost][1]:
                    lObsInfo[11] = 1
                    break

            if lObsInfo[11] == 1:
                break
            if self.gameMap[tempLoc[0]][tempLoc[1]] == '.':
                lObsInfo[7] = 1
            tempLoc[0] = tempLoc[0] - 1

        # check east
        tempLoc = copy.deepcopy(self.pacManPos)
        tempLoc[1] = tempLoc[1] + 1
        while tempLoc[1] < len(self.gameMap[0]) - 1 and self.gameMap[tempLoc[0]][tempLoc[1]] != 'x':
            for lGhost in range(len(self.ghostPoses)):
                if tempLoc[0] == self.ghostPoses[lGhost][0] and tempLoc[1] == self.ghostPoses[lGhost][1]:
                    lObsInfo[12] = 1
                    break

            if lObsInfo[12] == 1:
                break
            if self.gameMap[tempLoc[0]][tempLoc[1]] == '.':
                lObsInfo[8] = 1
            tempLoc[1] = tempLoc[1] + 1

        # check south
        tempLoc = copy.deepcopy(self.pacManPos)
        tempLoc[0] = tempLoc[0] + 1
        while tempLoc[0] < len(self.gameMap) - 1 and self.gameMap[tempLoc[0]][tempLoc[1]] != 'x':
            for lGhost in range(len(self.ghostPoses)):
                if tempLoc[0] == self.ghostPoses[lGhost][0] and tempLoc[1] == self.ghostPoses[lGhost][1]:
                    lObsInfo[13] = 1
                    break

            if lObsInfo[13] == 1:
                break
            if self.gameMap[tempLoc[0]][tempLoc[1]] == '.':
                lObsInfo[9] = 1
            tempLoc[0] = tempLoc[0] + 1

        # check west
        tempLoc = copy.deepcopy(self.pacManPos)
        tempLoc[1] = tempLoc[1] - 1
        while tempLoc[1] > 0 and self.gameMap[tempLoc[0]][tempLoc[1]] != 'x':
            for lGhost in range(len(self.ghostPoses)):
                if tempLoc[0] == self.ghostPoses[lGhost][0] and tempLoc[1] == self.ghostPoses[lGhost][1]:
                    lObsInfo[14] = 1
                    break

            if lObsInfo[14] == 1:
                break
            if self.gameMap[tempLoc[0]][tempLoc[1]] == '.':
                lObsInfo[10] = 1
            tempLoc[1] = tempLoc[1] - 1

        if self.powerPillCounter >= 0:
            lObsInfo[15] = 1
        from Util import computeIntFromBinary
        self.observationId = computeIntFromBinary(lObsInfo)
        return self.observationId

    def computeFoodManhattanDist(self):
        lMinDist = 5
        lYDiff = -4
        while lYDiff < 4:
            lXDiff = -4
            while lXDiff < 4:
                lYPos = self.pacManPos[0] + lYDiff
                lXPos = self.pacManPos[1] + lXDiff
                if 0 < lYPos < len(self.gameMap) - 1 and 0 < lXPos < len(self.gameMap[0]) - 1:
                    if self.gameMap[lYPos][lXPos] == '.' and abs(lYDiff) + abs(lXDiff) < lMinDist:
                        lMinDist = abs(lYDiff) + abs(lXDiff)
                lXDiff = lXDiff + 1
            lYDiff = lYDiff + 1
        return lMinDist

    def moveGhosts(self):
        for lGhost in range(len(self.ghostPoses)):
            if abs(self.ghostPoses[lGhost][0] - self.pacManPos[0]) + \
                    abs(self.ghostPoses[lGhost][1] - self.pacManPos[1]) <= 5:
                if self.powerPillCounter < 0:
                    self.ghostDirs[lGhost] = self.moveGhostAggressive(lGhost)
                else:
                    self.ghostDirs[lGhost] = self.moveGhostDefensive(lGhost)
            else:
                self.ghostDirs[lGhost] = self.moveGhostRandom(lGhost)

    def moveGhostAggressive(self, pGhost):
        bestDist = MAX_VALUE
        bestDir = -1
        if random() < PacMan.CHASE_PROB:
            validMoves = self.getValidMovements(self.ghostPoses[pGhost][1], self.ghostPoses[pGhost][0])

            for dir in range(4):
                if dir not in validMoves or dir == self.oppositeDir(self.ghostDirs[pGhost]):
                    continue

                dist = self.directionalDistance(self.pacManPos, self.ghostPoses[pGhost], dir)
                if dist <= bestDist:
                    bestDist = dist
                    bestDir = dir

        if bestDir != -1:
            self.makeMove(bestDir, self.ghostPoses[pGhost])
        else:
            self.moveGhostRandom(pGhost)
        return bestDir

    def moveGhostDefensive(self, pGhost):
        bestDist = MIN_VALUE
        bestDir = -1
        if random() > PacMan.DEFENSIVE_SLIP:
            validMoves = self.getValidMovements(self.ghostPoses[pGhost][1], self.ghostPoses[pGhost][0])

            for dir in range(4):
                if dir not in validMoves or dir == self.oppositeDir(self.ghostDirs[pGhost]):
                    continue

                dist = self.directionalDistance(self.pacManPos, self.ghostPoses[pGhost], dir)
                if dist >= bestDist:
                    bestDir = dir
                    bestDist = dist

        if bestDir != -1:
            self.makeMove(bestDir, self.ghostPoses[pGhost])
        else:
            self.moveGhostRandom(pGhost)
        return bestDir

    def getValidMovements(self, pXPos, pYPos):
        lValidMoves = []
        if pYPos + 1 < len(self.gameMap) and self.gameMap[pYPos + 1][pXPos] != 'x':
            lValidMoves.append(PacMan.SOUTH)
        if pYPos - 1 > -1 and self.gameMap[pYPos - 1][pXPos] != 'x':
            lValidMoves.append(PacMan.NORTH)
        if pXPos - 1 > -1 and self.gameMap[pYPos][pXPos - 1] != 'x':
            lValidMoves.append(PacMan.WEST)
        if pXPos + 1 < len(self.gameMap[0]) and self.gameMap[pYPos][pXPos + 1] != 'x':
            lValidMoves.append(PacMan.EAST)
        return lValidMoves

    def oppositeDir(self, direction):
        if direction == PacMan.NORTH:
            return PacMan.SOUTH
        elif direction == PacMan.SOUTH:
            return PacMan.NORTH
        elif direction == PacMan.EAST:
            return PacMan.WEST
        elif direction == PacMan.WEST:
            return PacMan.EAST
        else:
            return -1

    def directionalDistance(self, lhs, rhs, dir):
        if dir == PacMan.NORTH:
            return lhs[0] - rhs[0]
        elif dir == PacMan.EAST:
            return rhs[1] - lhs[1]
        elif dir == PacMan.SOUTH:
            return rhs[0] - lhs[0]
        elif dir == PacMan.WEST:
            return lhs[1] - rhs[1]
        else:
            Exception("invalid direction: " + str(dir))

    def makeMove(self, pMove, pPos):
        if pMove == PacMan.NORTH:
            pPos[0] = pPos[0] - 1
        elif pMove == PacMan.EAST:
            pPos[1] = pPos[1] + 1
        elif pMove == PacMan.SOUTH:
            pPos[0] = pPos[0] + 1
        elif pMove == PacMan.WEST:
            pPos[1] = pPos[1] - 1

        if self.gameMap[pPos[0]][pPos[1]] == '<':
            pPos[1] = len(self.gameMap[0]) - 2
        elif self.gameMap[pPos[0]][pPos[1]] == '>':
            pPos[1] = 1

    def moveGhostRandom(self, pGhost):
        lValidMoves = self.getValidMovements(self.ghostPoses[pGhost][1], self.ghostPoses[pGhost][0])
        idx = randint(0, len(lValidMoves))
        lMove = lValidMoves[idx]
        while lMove == self.oppositeDir(self.ghostDirs[pGhost]):
            lMove = lValidMoves[randint(0, len(lValidMoves))]
        self.makeMove(lMove, self.ghostPoses[pGhost])
        return lMove

    def executeAction(self, aid):
        self.moveGhosts()
        self.currImmediateReward = self.movePacman(aid)

    def movePacman(self, pAct):
        lValidMoves = self.getValidMovements(self.pacManPos[1], self.pacManPos[0])
        if self.powerPillCounter >= 0:
            self.powerPillCounter = self.powerPillCounter - 1

        if pAct not in lValidMoves:
            return self.computeNewStateInformation() - 10.0
        else:
            self.makeMove(pAct, self.pacManPos)
        return self.computeNewStateInformation()

    def computeNewStateInformation(self):
        lReward = -1.0
        for lGhost in range(len(self.ghostPoses)):
            if self.ghostPoses[lGhost][0] == self.pacManPos[0] and self.ghostPoses[lGhost][1] == self.pacManPos[1]:
                if self.powerPillCounter >= 0:
                    lReward = lReward + 25.0
                    self.resetGhost(lGhost)
                else:
                    lReward = lReward - 50.0
                    self.inTerminalState = True

        if self.gameMap[self.pacManPos[0]][self.pacManPos[1]] == '.':
            self.gameMap[self.pacManPos[0]][self.pacManPos[1]] = ' '
            lReward = lReward + 10.0
            self.foodLeft = self.foodLeft - 1
            if self.foodLeft == 0:
                self.inTerminalState = True
                lReward += 100.0
        elif self.gameMap[self.pacManPos[0]][self.pacManPos[1]] == 'o':
            self.gameMap[self.pacManPos[0]][self.pacManPos[1]] = ' '
            self.powerPillCounter = 15
        return lReward

    def resetGhost(self, pGhost):
        self.ghostPoses[pGhost] = copy.deepcopy(PacMan.INIT_GHOST_POSES_array[pGhost])

