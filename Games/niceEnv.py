from ASimulator import Simulator
import numpy as np
from numpy.random import choice, seed
import Paramter

class niceEnv(Simulator):
    NUMActions = 3
    NUMObservations = 5
    NUMRewards = 5
    NUMStates = 5
    Actions = ["Turn-Left", "Stay", "Turn-Right"]
    Observations = ["left-2-ob", "left-1-ob", "middle-ob", "right-1-ob", "right-2-ob"]
    States = ["left-2-st", "left-1-st", "middle-st", "right-1-st", "right-2-st"]
    StatesID = [0, 1, 2, 3, 4]
    ObservationsID = [0, 1, 2, 3, 4]

    TMatsTurnleft = np.array([[0.95, 0.05, 0, 0, 0],
                              [0.7, 0.25, 0.05, 0, 0],
                              [0, 0.7, 0.25, 0.05, 0],
                              [0, 0, 0.7, 0.25, 0.05],
                              [0, 0, 0, 0.7, 0.30]])
    TMatsStay = np.array([[0.8, 0.2, 0.0, 0.0, 0.0],
                          [0.2, 0.6, 0.2, 0.0, 0.0],
                          [0.0, 0.2, 0.6, 0.2, 0.0],
                          [0.0, 0.0, 0.2, 0.6, 0.2],
                          [0.0, 0.0, 0.0, 0.2, 0.8]])
    TMatsTurnRight = np.array([[0.3, 0.7, 0.0, 0.0, 0.0],
                               [0.05, 0.25, 0.7, 0.0, 0.0],
                               [0.0, 0.05, 0.25, 0.7, 0.0],
                               [0.0, 0.0, 0.05, 0.25, 0.7],
                               [0.0, 0.0, 0.0, 0.05, 0.95]])
    TMats = dict()
    TMats[0] = TMatsTurnleft
    TMats[1] = TMatsStay
    TMats[2] = TMatsTurnRight

    OMatsMove = np.array([[0.8, 0.2, 0.0, 0.0, 0.0],
                          [0.1, 0.8, 0.1, 0.0, 0.0],
                          [0.0, 0.1, 0.8, 0.1, 0.0],
                          [0.0, 0.0, 0.1, 0.8, 0.1],
                          [0.0, 0.0, 0.0, 0.2, 0.8]])
    OMats = dict()
    OMats[0] = OMatsMove
    OMats[1] = OMatsMove
    OMats[2] = OMatsMove
    RMatsMove = np.array([-50.0, -10.0, 100.0, -10.0, -50.0])
    RMats = dict()
    RMats[0] = RMatsMove
    RMats[1] = RMatsMove
    RMats[2] = RMatsMove
    Belief = [0.0, 0.0, 1.0, 0.0, 0.0]

    def Clone(self):
        return niceEnv()

    def getNumActions(self):
        return niceEnv.NUMActions

    def getNumObservations(self):
        return niceEnv.NUMObservations

    def getNumRewards(self):
        return niceEnv.NUMRewards

    def __init__(self):
        super().__init__()
        self.reward = None
        self.observation = None
        self.agent = None
        self.actionCount = 0
        self.terminate = False

    def InitRun(self):
        # seed(seedint)
        # self.agent = choice(a=niceEnv.StatesID, p=niceEnv.Belief, size=1)[0]
        self.agent = 2
        self.actionCount = 0
        self.terminate = False

    def getGameName(self):
        return "niceEnv"

    def executeAction(self, aid):
        TMat = niceEnv.TMats[aid]
        OMat = niceEnv.OMats[aid]
        RMat = niceEnv.RMats[aid]

        # Taking Transition
        ps = TMat[self.agent]
        NewTiger = choice(a=niceEnv.StatesID, p=list(ps), size=1)[0]
        self.agent = NewTiger
        # Taking Observation
        p1s = OMat[self.agent]
        self.reward = RMat[self.agent]
        self.observation = choice(a=niceEnv.ObservationsID, p=list(p1s), size=1)[0]
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