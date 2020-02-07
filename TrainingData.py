import Paramter


class TrainingData:
    def __init__(self):
        self.batch = -1
        self.data = []
        self.episodeId = -1
        self.stepId = 0
        self.validActOb = set()
        self.test = None
        self.history = None
        self.retsetAction = None
        self.testDict = dict()
        self.histDict = dict()
        self.testID = 0
        self.histID = 0

    def MergeAllBatchData(self):
        trainData = TrainingData()
        trainData.validActOb = self.validActOb
        trainData.testDict = self.testDict
        trainData.histDict = self.histDict
        trainData.newDataBatch()
        for i in range(self.batch + 1):
            trainData.data[trainData.getBatch()] = trainData.data[trainData.getBatch()] + self.data[i]
        return trainData

    def ReturnEmptyObject(self):
        return TrainingData()

    def WriteData(self, file):
        from Util import writerDataintoDisk
        idx = self.getBatch()
        writerDataintoDisk(file=file, data=self.data[idx])

    @staticmethod
    def LoadData(TrainData, file, rewardDict):
        from Util import readDataintoDisk
        TrainData = readDataintoDisk(file=file, rewardDict=rewardDict, TrainData=TrainData)
        return TrainData

    def isAvailable(self):
        if self.batch < 0:
            return False
        return True

    def getBatch(self):
        return self.batch

    def resetData(self):
        self.episodeId = -1
        self.stepId = 0

    def newDataBatch(self):
        self.batch = self.batch + 1
        self.data.append([])
        self.resetData()

    def newEpisode(self):
        self.data[self.batch].append([])
        self.episodeId = self.episodeId + 1
        self.stepId = 0
        self.test = []
        self.history = []

    def EndEpisode(self):
        if self.retsetAction is None:
            self.retsetAction = (-1, -1, -1)
        self.data[self.batch][self.episodeId].insert(self.stepId, self.retsetAction)
        if Paramter.introduceReward:
            test = 'a' + str(-1) + 'o' + str(-1) + 'r' + \
                   str(-1)
        else:
            test = 'a' + str(-1) + 'o' + str(-1)
        if test not in self.testDict.keys():
            self.testDict[test] = self.testID
            self.testID = self.testID + 1

    def AddData(self, aid, oid, r, ActOb):
        data = (aid, oid, r)
        self.data[self.batch][self.episodeId].insert(self.stepId, data)
        self.stepId = self.stepId + 1
        self.test.append(ActOb)
        self.history.append(ActOb)
        self.validActOb.add(ActOb)
        if len(self.test) > Paramter.maxTestlen:
            self.test.remove(self.test[0])
        if len(self.history) > Paramter.maxHistLen:
            self.history.remove(self.history[0])
        tests = None
        for i in range(len(self.test)):
            oneStep = self.test[len(self.test) - i - 1]
            if tests is None:
                tests = oneStep
            else:
                tests = oneStep + tests
            if tests not in self.testDict.keys():
                self.testDict[tests] = self.testID
                self.testID = self.testID + 1
        tmp = None
        for i in range(len(self.history)):
            if tmp is None:
                tmp = self.history[i]
            else:
                tmp = tmp + self.history[i]
        if tmp not in self.histDict.keys():
            self.histDict[tmp] = self.histID
            self.histID = self.histID + 1
