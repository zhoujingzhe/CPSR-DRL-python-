from ASimulator import Simulator
import numpy as np
from numpy.random import choice
import Paramter


class Tiger95(Simulator):
    NUMActions = 3
    NUMObservations = 2
    NUMRewards = 3
    NUMStates = 2
    Actions = ["Open-Left", "Open-Right", "Listen"]
    Observations = ["Tiger-Left", "Tiger-Right"]
    States = ["Tiger-Left", "Tiger-Right"]
    StatesID = [0, 1]
    ObservationsID = [0, 1]

    TMatsOpen = np.array([[0.5, 0.5],
                          [0.5, 0.5]])
    TMatsListen = np.array([[1.0, 0.0],
                            [0.0, 1.0]])
    TMats = dict()
    TMats[0] = TMatsOpen
    TMats[1] = TMatsOpen
    TMats[2] = TMatsListen

    OMatsOpen = np.array([[0.5, 0.5],
                          [0.5, 0.5]])
    OMatsListen = np.array([[0.85, 0.15],
                            [0.15, 0.85]])
    OMats = dict()
    OMats[0] = OMatsOpen
    OMats[1] = OMatsOpen
    OMats[2] = OMatsListen

    RMatsOpenLeft = np.array([-100.0, 10.0])
    RMatsOpenRight = np.array([10.0, -100.0])
    RMatsListen = np.array([-1.0, -1.0])
    RMats = dict()
    RMats[0] = RMatsOpenLeft
    RMats[1] = RMatsOpenRight
    RMats[2] = RMatsListen
    Belief = [0.5, 0.5]

    def Clone(self):
        return Tiger95()

    def getNumActions(self):
        return Tiger95.NUMActions

    def getNumObservations(self):
        return Tiger95.NUMObservations

    def getNumRewards(self):
        return Tiger95.NUMRewards

    def __init__(self):
        super().__init__()
        self.reward = None
        self.observation = None
        self.Tiger = None
        self.actionCount = 0
        self.terminate = False

    def InitRun(self):
        # seed(seedint)
        self.Tiger = choice(a=Tiger95.StatesID, p=Tiger95.Belief, size=1)[0]
        self.actionCount = 0
        self.terminate = False

    def getGameName(self):
        return "Tiger95"

    def executeAction(self, aid):
        TMat = Tiger95.TMats[aid]
        OMat = Tiger95.OMats[aid]
        RMat = Tiger95.RMats[aid]
        self.reward = RMat[self.Tiger]
        # Taking Transition
        ps = TMat[self.Tiger]
        NewTiger = choice(a=Tiger95.StatesID, p=list(ps), size=1)[0]
        self.Tiger = NewTiger
        # Taking Observation
        p1s = OMat[self.Tiger]
        self.observation = choice(a=Tiger95.ObservationsID, p=list(p1s), size=1)[0]
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
