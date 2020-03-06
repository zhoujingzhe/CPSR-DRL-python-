from ASimulator import Simulator
import numpy as np
from numpy.random import choice
import Paramter


class Shuttle(Simulator):
    NUMActions = 3
    NUMObservations = 5
    NUMRewards = 3
    NUMStates = 8
    Actions = ["TurnAround", "GoForward", "Backup"]
    Observations = ["See-LRV-forward", "See-MRV-forward", "See-that-we-are-docked-in-MRV", "See-nothing",
                    "See-that-we-are-docked-in-LRV"]
    States = ["Docked-in-LRV", "Just-outside-space-station-MRV,-front-of-ship-facing-station", "Space-facing-LRV",
              "Just-outside-space-station-LRV,-back-of-ship-facing-station",
              "Just-outside-space-station-MRV,-back-of-ship-facing-station", "Space,-facing-LRV",
              "Just-outside-space-station-LRV,-front-of-ship-facing-station", "Docked-in-MRV"]
    StatesID = [0, 1, 2, 3, 4, 5, 6, 7]
    ObservationsID = [0, 1, 2, 3, 4]

    TMatsTurnAround = np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    TMatsGoForward = np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    TMatsBackup = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                            [0.0, 0.4, 0.3, 0.0, 0.3, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.1, 0.8, 0.0, 0.0, 0.1, 0.0],
                            [0.7, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.7],
                            [0.0, 0.1, 0.0, 0.0, 0.8, 0.1, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.3, 0.0, 0.3, 0.4, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    TMats = dict()
    TMats[0] = TMatsTurnAround
    TMats[1] = TMatsGoForward
    TMats[2] = TMatsBackup

    OMat = np.array([[0.0, 0.0, 0.0, 0.0, 1.0],
                     [0.0, 1.0, 0.0, 0.0, 0.0],
                     [0.0, 0.7, 0.0, 0.3, 0.0],
                     [0.0, 0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0, 0.0],
                     [0.7, 0.0, 0.0, 0.3, 0.0],
                     [1.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0, 0.0]])
    OMats = dict()
    OMats[0] = OMat
    OMats[1] = OMat
    OMats[2] = OMat

    Belief = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    def Clone(self):
        return Shuttle()

    def getNumActions(self):
        return Shuttle.NUMActions

    def getNumObservations(self):
        return Shuttle.NUMObservations

    def getNumRewards(self):
        return Shuttle.NUMRewards

    def __init__(self):
        super().__init__()
        self.reward = None
        self.observation = None
        self.ship = None
        self.actionCount = 0
        self.terminate = False

    def InitRun(self):
        # seed(seedint)
        self.ship = choice(a=Shuttle.StatesID, p=Shuttle.Belief, size=1)[0]
        self.actionCount = 0
        self.terminate = False

    def getGameName(self):
        return "Shuttle"

    def executeAction(self, aid):
        TMat = Shuttle.TMats[aid]
        OMat = Shuttle.OMats[aid]

        # Taking Transition
        ps = TMat[self.ship]
        Newship = choice(a=Shuttle.StatesID, p=list(ps), size=1)[0]
        self.reward = 0.0
        if aid == 1 and self.ship == 1 and Newship == 1:
            self.reward = -3.0
        elif aid == 1 and self.ship == 6 and Newship == 6:
            self.reward = -3.0
        elif aid == 2 and self.ship == 3 and Newship == 0:
            self.reward = 10.0
        self.ship = Newship
        # Taking Observation
        p1s = OMat[self.ship]
        self.observation = choice(a=Shuttle.ObservationsID, p=list(p1s), size=1)[0]
        self.actionCount = self.actionCount + 1
        if self.actionCount >= Paramter.LengthOfAction:
            self.terminate = True

    def getObservation(self):
        o = self.observation
        self.observation = None
        if o is None:
            Exception("Tiger95 doesn't generate new observation!")
        return o

    def getReward(self):
        r = self.reward
        self.reward = None
        if r is None:
            Exception("Tiger95 doesn't generate new reward!")
        return r

    def isTerminate(self):
        return self.terminate
