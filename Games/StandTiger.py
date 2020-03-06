from ASimulator import Simulator
import numpy as np
from numpy.random import choice, seed
import Paramter


class StandTiger(Simulator):
    NUMActions = 5
    NUMObservations = 6
    NUMRewards = 4
    NUMStates = 6
    Actions = ["Open-Left", "Open-Middle", "Open-Right", "Listen", "Stand-Up"]
    Observations = ["tiger-left-sit", "tiger-middle-sit", "tiger-right-sit", "tiger-left-stand", "tiger-middle-stand",
                    "tiger-right-stand"]
    States = ["tiger-left-sit", "tiger-middle-sit", "tiger-right-sit", "tiger-left-stand", "tiger-middle-stand",
              "tiger-right-stand"]
    StatesID = [0, 1, 2, 3, 4, 5]
    ObservationsID = [0, 1, 2, 3, 4, 5]
    Belief = np.array([0.333333333, 0.333333333, 0.333333333, 0.0, 0.0, 0.0])

    TMatsListen = np.array([[1.0, 0, 0, 0, 0, 0],
                            [0, 1.0, 0, 0, 0, 0],
                            [0, 0, 1.0, 0, 0, 0],
                            [0, 0, 0, 1.0, 0, 0],
                            [0, 0, 0, 0, 1.0, 0],
                            [0, 0, 0, 0, 0.0, 1.0]])
    TMatsOpen = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                          [0.33333333, 0.33333333, 0.33333333, 0.0, 0.0, 0.0],
                          [0.33333333, 0.33333333, 0.33333333, 0.0, 0.0, 0.0],
                          [0.33333333, 0.33333333, 0.33333333, 0.0, 0.0, 0.0]])
    TMatsStand = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                           [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    TMats = dict()
    TMats[0] = TMatsOpen
    TMats[1] = TMatsListen
    TMats[2] = TMatsStand

    OMatsOpen = np.array([[0.33333333, 0.33333333, 0.33333333, 0.0, 0.0, 0.0],
                          [0.33333333, 0.33333333, 0.33333333, 0.0, 0.0, 0.0],
                          [0.33333333, 0.33333333, 0.33333333, 0.0, 0.0, 0.0],
                          [0.33333333, 0.33333333, 0.33333333, 0.0, 0.0, 0.0],
                          [0.33333333, 0.33333333, 0.33333333, 0.0, 0.0, 0.0],
                          [0.33333333, 0.33333333, 0.33333333, 0.0, 0.0, 0.0]])

    OMatsListen = np.array([[0.75, 0.15, 0.1, 0.0, 0.0, 0.0],
                            [0.1, 0.8, 0.1, 0.0, 0.0, 0.0],
                            [0.1, 0.15, 0.75, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.33333333, 0.33333333, 0.33333333],
                            [0.0, 0.0, 0.0, 0.33333333, 0.33333333, 0.33333333],
                            [0.0, 0.0, 0.0, 0.33333333, 0.33333333, 0.33333333]])

    OMatsStand = np.array([[0.0, 0.0, 0.0, 0.33333333, 0.33333333, 0.33333333],
                           [0.0, 0.0, 0.0, 0.33333333, 0.33333333, 0.33333333],
                           [0.0, 0.0, 0.0, 0.33333333, 0.33333333, 0.33333333],
                           [0.0, 0.0, 0.0, 0.33333333, 0.33333333, 0.33333333],
                           [0.0, 0.0, 0.0, 0.33333333, 0.33333333, 0.33333333],
                           [0.0, 0.0, 0.0, 0.33333333, 0.33333333, 0.33333333]])
    OMats = dict()
    OMats[0] = OMatsOpen
    OMats[1] = OMatsListen
    OMats[2] = OMatsStand

    RMatsOpenLeft = np.array([-1000.0, -1000.0, -1000.0, -100.0, 30.0, 30.0])
    RMatsOpenMiddle = np.array([-1000.0, -1000.0, -1000.0, 30.0, -100.0, 30.0])
    RMatsOpenRight = np.array([-1000.0, -1000.0, -1000.0, 30.0, 30.0, -100.0])
    RMatsListenAndStandUp = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    RMats = dict()
    RMats[0] = RMatsOpenLeft
    RMats[1] = RMatsOpenMiddle
    RMats[2] = RMatsOpenRight
    RMats[3] = RMatsListenAndStandUp

    def Clone(self):
        return StandTiger()

    # def getRewardDict(self):
    #     return StandTiger.Rewards

    def getNumActions(self):
        return StandTiger.NUMActions

    def getNumObservations(self):
        return StandTiger.NUMObservations

    def getNumRewards(self):
        return StandTiger.NUMRewards

    def __init__(self):
        super()
        self.actionCount = 0
        self.terminate = False
        self.rewards = None
        self.observations = None
        self.Tiger = None

    def InitRun(self):
        # from numpy.random import randint
        # seed(randint(0, 1000000))
        self.Tiger = choice(a=list(StandTiger.StatesID), p=list(StandTiger.Belief), size=1)[0]
        self.actionCount = 0
        self.terminate = False

    def getGameName(self):
        return "StandTiger"

    def executeAction(self, aid):
        TMats = None
        OMats = None
        RMats = None
        if aid < 3:
            TMats = StandTiger.TMats[0]
            OMats = StandTiger.OMats[0]
            RMats = StandTiger.RMats[aid]
        elif aid == 3:
            TMats = StandTiger.TMats[1]
            OMats = StandTiger.OMats[1]
            RMats = StandTiger.RMats[aid]
        elif aid == 4:
            TMats = StandTiger.TMats[2]
            OMats = StandTiger.OMats[2]
            RMats = StandTiger.RMats[3]
        # Taking Reward
        self.rewards = RMats[self.Tiger]
        # Taking Transition
        p = TMats[self.Tiger]
        NewTiger = choice(a=list(StandTiger.StatesID), p=list(p), size=1)[0]
        self.Tiger = NewTiger
        # Taking Observation
        p1s = OMats[self.Tiger]
        self.observations = choice(a=list(StandTiger.ObservationsID), p=list(p1s), size=1)[0]
        self.actionCount = self.actionCount + 1
        if self.actionCount > Paramter.LengthOfAction:
            self.terminate = True
        # if aid < 3:
        #     self.terminate = True

    def isTerminate(self):
        return self.terminate

    def getObservation(self):
        o = self.observations
        self.observations = None
        if o is None:
            Exception("Tiger95 doesn't generate new observation!")
        return o

    def getReward(self):
        r = self.rewards
        self.rewards = None
        if r is None:
            Exception("Tiger95 doesn't generate new reward!")
        return r
