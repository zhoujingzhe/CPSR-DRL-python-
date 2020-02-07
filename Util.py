import Paramter
import numpy as np
import json
import os

def writerMemoryintodisk(file, data):
    with open(file=file, mode='w') as f:
        s = json.dumps(data)
        f.write(s)
    print("Finish writing " + file)


def writerDataintoDisk(file, data):
    with open(file=file, mode='w') as f:
        for episode in data:
            f.write("Terminate!")
            f.write("\n")
            for d in episode:
                import numpy as np
                d = np.array(d)
                a = int(d[0])
                o = int(d[1])
                r = float(d[2])
                if a == -1:
                    continue
                f.write("aid: " + str(a) + ", oid: " + str(o) + ", r: " + str(r))
                f.write("\n")


def readDataintoDisk(file, rewardDict, TrainData):
    with open(file=file, mode='r') as f:
        count = 0
        for line in f:
            if line == "Terminate!\n":
                if count != 0:
                    TrainData.EndEpisode()
                TrainData.newEpisode()
                continue
            if line == "\n":
                continue
            aid, oid, r = extract(line)
            if str(r) not in rewardDict.keys():
                rewardDict[str(r)] = len(rewardDict)
            rid = rewardDict[str(r)]
            if Paramter.introduceReward:
                actOb = "a" + str(aid) + "o" + str(oid) + "r" + str(rid)
            else:
                actOb = "a" + str(aid) + "o" + str(oid)
            TrainData.AddData(aid=aid, oid=oid, r=r, ActOb=actOb)
            count = count + 1
        TrainData.EndEpisode()
    return TrainData

def copyRewardDict(rewardDict, rewardDict1):
    for key in rewardDict1.keys():
        rewardDict[key] = rewardDict1[key]

def readMemoryfromdisk(file):
    with open(file=file, mode='r') as f:
        s = f.readline()
        Mem = json.loads(s)
    print("Finish reading " + file)
    return Mem

def normalization(dist):
    if not isinstance(dist, np.ndarray):
        dist = np.array(dist)
    distsum = np.expand_dims(a=np.sum(a=dist, axis=-1), axis=-1)
    dist = dist / distsum
    return dist

def readaoMatsFromdisk(file):
    aoMats = readMemoryfromdisk(file=file)
    for key in aoMats.keys():
        aoMats[key] = np.array(aoMats[key])
    return aoMats


def computeIntFromBinary(pBinary):
    lResult = 0
    for i in range(len(pBinary)):
        if pBinary[i] == 1:
            lResult += 1 << i
    return lResult


def extract(line):
    aid = line.split("aid:")[-1]
    aid = aid.split(", ")[0]
    oid = line.split("oid:")[-1]
    oid = oid.split(", ")[0]
    r = line.split("r:")[-1]
    r = r.split(", ")[0]
    r = r.split("\n")[0]
    return int(aid), int(oid), float(r)


def ConvertLastBatchToTrainSet(data, RewardDict, epoch, pool, name, psrModel):
    ret = []
    args = []
    Batch = data.data[data.getBatch()]
    numEpisodePerThread = int(len(Batch) / Paramter.ThreadPoolSize)
    for i in range(Paramter.ThreadPoolSize):
        d = Batch[i * numEpisodePerThread:(i + 1) * numEpisodePerThread]
        fileName = "tmp\\RawData" + str(i) + ".txt"
        writerDataintoDisk(file=fileName, data=d)
        psrModel1 = psrModel.ReturnEmptyObject(name)
        tmpTrainData = data.ReturnEmptyObject()
        args.append([psrModel1, fileName, RewardDict, i, epoch, tmpTrainData])
    from MultiProcessSimulation import ConvertToTrainSet
    outputs = pool.map(func=ConvertToTrainSet, iterable=args)
    for file in outputs:
        d = readMemoryfromdisk(file=file)
        os.remove(file)
        ret = ret + d
    return ret

def ConvertToTrainSet(data, RewardDict, epoch, pool, name, psrModel):
    numBatch = data.getBatch() + 1
    ret = []
    args = []
    Batch = []
    for idx in range(numBatch):
        Batch = Batch + data.data[numBatch - idx - 1]
    numEpisodePerThread = int(len(Batch) / Paramter.ThreadPoolSize)
    for i in range(Paramter.ThreadPoolSize):
        d = Batch[i * numEpisodePerThread:(i + 1) * numEpisodePerThread]
        fileName = "tmp\\RawData" + str(i) + ".txt"
        writerDataintoDisk(file=fileName, data=d)
        psrModel1 = psrModel.ReturnEmptyObject(name)
        tmpTrainData = data.ReturnEmptyObject()
        args.append([psrModel1, fileName, RewardDict, i, epoch, tmpTrainData])
    from MultiProcessSimulation import ConvertToTrainSet
    outputs = pool.map(func=ConvertToTrainSet, iterable=args)
    for file in outputs:
        d = readMemoryfromdisk(file=file)
        os.remove(file)
        ret = ret + d
    return ret


def group2test(aid, oid, rid):
    ActOb = "a" + str(aid) + "o" + str(oid)
    if Paramter.introduceReward:
        ActOb = ActOb + "r" + str(rid)
    return ActOb


def merge(TrainData1, OuputData):
    OuputData.data[OuputData.batch] = OuputData.data[OuputData.batch] + TrainData1.data[TrainData1.batch]
    for actOb in TrainData1.validActOb:
        OuputData.validActOb.add(actOb)
    for key in TrainData1.testDict.keys():
        if key not in OuputData.testDict.keys():
            OuputData.testDict[key] = OuputData.testID
            OuputData.testID = OuputData.testID + 1
    for key in TrainData1.histDict.keys():
        if key not in OuputData.histDict.keys():
            OuputData.histDict[key] = OuputData.histID
            OuputData.histID = OuputData.histID + 1
    return OuputData
