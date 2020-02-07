from PSRmodel import CompressedPSR
import numpy as np
from MultiProcessSimulation import ConstructingTHMats
import Paramter
from Util import writerDataintoDisk, readaoMatsFromdisk


class TransformedPSR(CompressedPSR):
    def __init__(self, game):
        super().__init__(game=game)


    def build(self, data, aos, pool, rewardDict):
        self.validActObset = aos
        testSize = len(data.testDict)
        histSize = len(data.histDict) + 1
        e = np.zeros((histSize, 1))
        e[0, 0] = 1
        if self.THMat is None and self.HMat is None:
            print("Initialize THMat and HMat")
            self.THMat = np.zeros((testSize, histSize))
            self.HMat = np.zeros((histSize, 1))
        elif self.THMat.shape[0] != testSize or self.HMat.shape[0] != histSize:
            print("expand the size of THMat and HMat")
            oldTestSize = self.THMat.shape[0]
            oldHistSize = self.THMat.shape[1]
            newTHMat = np.zeros((testSize, histSize))
            newHMat = np.zeros((histSize, 1))
            newTHMat[0:oldTestSize:, 0:oldHistSize:] = self.THMat
            newHMat[:oldHistSize:, :] = self.HMat
            self.THMat = newTHMat
            self.HMat = newHMat
        for ao in self.validActObset:
            if ao not in self.aoMats.keys():
                self.aoMats[ao] = np.zeros((testSize, histSize))
        actObsPerThread = int(len(data.data[data.getBatch()]) / Paramter.ThreadPoolSize)
        args = []
        for i in range(Paramter.ThreadPoolSize):
            d = data.data[data.getBatch()][i * actObsPerThread:(i + 1) * actObsPerThread:]
            fileName = "tmp//dataForThread" + str(i) + ".txt"
            writerDataintoDisk(file=fileName, data=d)
            tmpTrainData = data.ReturnEmptyObject()
            args.append([fileName, data.testDict, data.histDict, data.validActOb, "TPSR", i, tmpTrainData, rewardDict])
        outputs = pool.map(func=ConstructingTHMats, iterable=args)
        print("Constructing the TH aoMats is finished!")
        THMat = [output[0] for output in outputs]
        HistMat = [output[1] for output in outputs]
        files = [output[2] for output in outputs]
        aoMats = []
        import os
        for file in files:
            aoMats.append(readaoMatsFromdisk(file=file))
            os.remove(file)
        THMat = np.array(THMat)
        HistMat = np.array(HistMat)
        THMat = np.sum(a=THMat, axis=0)
        HistMat = np.sum(a=HistMat, axis=0)
        if 0 in HistMat:
            Exception("There is a history that not seen before!")
        self.THMat = self.THMat + THMat
        self.HMat = self.HMat + HistMat
        for ao in self.validActObset:
            for i in range(len(aoMats)):
                self.aoMats[ao] = self.aoMats[ao] + aoMats[i][ao]
        ret = self.TruncatedSVD(mats=self.THMat, maxDim=Paramter.svdDim)
        u = ret[0]
        s = ret[1]
        vT = ret[2]
        Z = u.transpose()
        # pseduInverse = np.matmul(np.linalg.inv(s), vT).transpose()
        pseduInverse = np.linalg.pinv(np.matmul(Z, self.THMat))
        self.CaoMats = dict()
        for ao in self.validActObset:
            self.CaoMats[ao] = np.matmul(np.matmul(Z, self.aoMats[ao]), pseduInverse)
        histMat = np.diag(np.squeeze(a=self.HMat, axis=-1))
        histInverse = np.linalg.inv(histMat)
        tmp = np.matmul(Z, self.THMat)
        PQ = np.matmul(tmp, histInverse)
        self.mInf = np.linalg.lstsq(a=PQ.transpose(), b=np.ones((histSize, 1)))[0]
        self.pv = np.matmul(PQ, e)
        self.isbuilt = True

    def writeToExcel(self, testDict, HistDict, epoch):
        import os
        if not os.path.exists("Epoch" + str(epoch)):
            os.makedirs("Epoch" + str(epoch))
        with open(file="Epoch" + str(epoch) + "\\Test-History.csv", mode='w') as f:
            f.write(" ,")
            f.write("null, ")
            s = np.shape(self.THMat)
            rows = s[0]
            cols = s[1]
            for i in range(cols - 1):
                hist = None
                for key in HistDict.keys():
                    if HistDict[key] == i:
                        hist = key
                        break
                f.write(hist)
                f.write(", ")
            f.write("\n")
            for i in range(rows):
                row = None
                for key in testDict.keys():
                    if testDict[key] == i:
                        row = key
                        break
                f.write(row)
                f.write(", ")
                for j in range(cols):
                    f.write(str(self.THMat[i][j]))
                    f.write(", ")
                f.write("\n")