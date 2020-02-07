import Paramter
from MultiProcessSimulation import EvaluateMultiProcess, SimulateTrainDataMultiProcess, SimulateRunsOnCPSR
from Util import merge
from numpy.random import randint

class Simulator:

    def isTerminate(self):
        pass

    def Clone(self):
        pass

    def InitRun(self):
        pass

    def getGameName(self):
        pass

    def executeAction(self, aid):
        pass

    def getObservation(self):
        pass

    def getReward(self):
        pass

    def getNumActions(self):
        pass

    def getNumObservations(self):
        pass

    def getNumRewards(self):
        pass

    def getRewardDict(self):
        pass

    maxTestID = 100000

    def SimulateTrainData(self, runs, isRandom, psrModel, trainData, epoch, pool, RunOnVirtualEnvironment, name, rewardDict, ns):
        if not RunOnVirtualEnvironment:
            print("Simulating an agent on Real environment!")
            args = []
            for i in range(Paramter.ThreadPoolSize):
                args.append(
                    [self.Clone(), psrModel.ReturnEmptyObject(name=name), int(runs / Paramter.ThreadPoolSize), isRandom,
                     self.getNumActions(), epoch, randint(low=0, high=9999999), rewardDict, ns])
            TrainDataList = pool.map(func=SimulateTrainDataMultiProcess, iterable=args)
        else:
            print("Simulating an agent on CPSR environment!")
            args = []
            for i in range(Paramter.ThreadPoolSize):
                args.append(
                    [psrModel.ReturnEmptyObject(name=name), int(runs / Paramter.ThreadPoolSize), self.getNumActions(),
                     epoch, randint(low=0, high=9999999), rewardDict, ns])
            TrainDataList = pool.map(func=SimulateRunsOnCPSR, iterable=args)
        for TrainData in TrainDataList:
            trainData = merge(TrainData1=TrainData, OuputData=trainData)
        return trainData

    def calulateMaxTestID(self):
        if Paramter.introduceReward:
            Simulator.maxTestID = (self.getNumActions() * self.getNumObservations() * self.getNumRewards()) \
                                  ** Paramter.maxTestlen + 1
        else:
            Simulator.maxTestID = (self.getNumActions() * self.getNumObservations()) ** Paramter.maxTestlen + 1
        Paramter.maxTestID = Simulator.maxTestID

    def SimulateTestingRun(self, runs, epoch, pool, psrModel, name, rewardDict, ns):
        args = []
        for i in range(Paramter.ThreadPoolSize):
            args.append([int(runs / Paramter.ThreadPoolSize), psrModel.ReturnEmptyObject(name=name), self.getNumActions(),
                         self.Clone(), epoch, randint(low=0, high=1000000000), rewardDict, ns])
        EvalDatas = pool.map(func=EvaluateMultiProcess, iterable=args)
        output = []
        for data in EvalDatas:
            output = output + data
        return output
