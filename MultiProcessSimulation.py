from TrainingData import TrainingData
from Agent import Agent
from Util import writerMemoryintodisk, readDataintoDisk, group2test
from numpy.random import choice, seed, normal
import os
import numpy as np
import Paramter
np.seterr('raise')


def init(mTestID, file, l):
    global maxTestID, lock
    maxTestID = mTestID
    Paramter.readfile(file=file)
    lock = l


def SimulateRunsOnCPSR(args):
    psrModel = args[0]
    runs = args[1]
    numactions = args[2]
    epoch = args[3]
    seed(int(args[4]))
    if epoch > -1:
        psrModel.loadModel(epoch)
    print("the seed:" + str(args[4]))
    rewardDict = args[5]
    dataCollector = TrainingData()
    dataCollector.newDataBatch()
    agent = Agent(PnumActions=numactions, epilson=Paramter.epilson, inputDim=(Paramter.svdDim,),
                  algorithm=Paramter.algorithm, Parrallel=False)
    if epoch >= 0:
        agent.LoadWeight(epoch=epoch)
    for i in range(int(runs)):
        pv = psrModel.getCurrentPV()
        count = 0
        dataCollector.newEpisode()
        while count < Paramter.LengthOfAction:
            CoreTestPredictions = psrModel.generateCoreTestPrediction(pvs=pv)
            aid = agent.getAction(state=CoreTestPredictions)
            preds = psrModel.PredictsForPV(pv1s=pv)
            p = []
            Obs = []
            for ActOb in preds.keys():
                if aid == int(ActOb[1]):
                    p.append(preds[ActOb])
                    Obs.append(ActOb)
            ActOb = choice(a=Obs, p=p, size=1)[0]
            aid = int(ActOb[1])
            oid = int(ActOb[3])
            rid = int(ActOb[5])
            r = None
            for key in rewardDict.keys():
                if rewardDict[key] == rid:
                    r = key
                    break
            pv = psrModel.PassAO(pvs=pv, aos=ActOb)
            if r is None:
                Exception("Cannot find reward!")
            dataCollector.AddData(aid=aid, oid=oid, r=r, ActOb=ActOb)
            count = count + 1
        dataCollector.EndEpisode()
    return dataCollector


def checkRewardDict(r, rewardDict, ns):
    if str(r) not in rewardDict.keys():
        lock.acquire()
        if str(r) not in rewardDict.keys():
            rewardDict[str(r)] = ns.rewardCount
            ns.rewardCount = ns.rewardCount + 1
        lock.release()


def SimulateTrainDataMultiProcess(args):
    Env = args[0]
    Env.calulateMaxTestID()
    psrModel = args[1]
    runs = args[2]
    isRandom = args[3]
    numactions = args[4]
    epoch = args[5]
    seed(int(args[6]))
    rewardDict = args[7]
    ns = args[8]
    if epoch > -1 and not isRandom:
        psrModel.loadModel(epoch)
    print("the seed:" + str(args[6]))
    agent = Agent(PnumActions=numactions, epilson=Paramter.epilson, inputDim=(Paramter.svdDim,),
                  algorithm=Paramter.algorithm, Parrallel=False)
    # agent = Agent(PnumActions=Env.getNumActions(), epilson=Paramter.epilson, buildNet=False, game=Env.getGameName())

    isloaded = False
    if epoch >= 0 and not isRandom:
        isloaded = True
        agent.LoadWeight(epoch=epoch)
    TrainData = TrainingData()
    TrainData.newDataBatch()
    for i in range(runs):
        pv = None
        if psrModel.isbuilt:
            pv = psrModel.getCurrentPV()
        Env.InitRun()
        TrainData.newEpisode()
        count = 0
        while not Env.isTerminate() and count < Paramter.LengthOfAction:
            if isRandom:
                aid = agent.getRandomAction()
            elif not isloaded:
                aid = agent.getRandomAction()
            else:
                CoreTestPredictions = psrModel.generateCoreTestPrediction(pvs=pv)
                aid = agent.getAction(state=CoreTestPredictions)
            Env.executeAction(aid=aid)
            oid = Env.getObservation()
            r = Env.getReward()
            checkRewardDict(r=r, rewardDict=rewardDict, ns=ns)
            rid = rewardDict[str(r)]
            ActOb = group2test(aid=aid, oid=oid, rid=rid)
            TrainData.AddData(aid=aid, oid=oid, r=r, ActOb=ActOb)
            if psrModel.isbuilt:
                pv = psrModel.PassAO(pvs=pv, aos=ActOb)
            count = count + 1
        TrainData.EndEpisode()
    return TrainData


def EvaluateMultiProcess(args):
    runs = args[0]
    psrModel = args[1]
    numactions = args[2]
    Env = args[3]
    epoch = args[4]
    seed(int(args[5]))
    psrModel.loadModel(epoch)
    print("the seed:" + str(args[5]))
    rewardDict = args[6]
    ns = args[7]
    agent = Agent(PnumActions=numactions, epilson=Paramter.epilson, inputDim=(Paramter.svdDim,),
                  algorithm=Paramter.algorithm, Parrallel=False)
    # agent = Agent(PnumActions=Env.getNumActions(), epilson=Paramter.epilson, buildNet=False, game=Env.getGameName())
    if epoch >= 0:
        agent.LoadWeight(epoch=epoch)
    else:
        Exception("The agent hasn't been trained!")
    RArray = []
    for i in range(runs):
        pv = None
        if psrModel.isbuilt:
            pv = psrModel.getCurrentPV()
        else:
            Exception("The CPSR model are not built")
        count = 0
        #####################################################
        # Add a new Episode
        RArray.append([])
        Env.InitRun()
        while not Env.isTerminate() and count < Paramter.LengthOfAction:
            CoreTestPredictions = psrModel.generateCoreTestPrediction(pvs=pv)
            aid = agent.getGreedyAction(state=CoreTestPredictions)
            Env.executeAction(aid=aid)
            oid = Env.getObservation()
            r = Env.getReward()
            checkRewardDict(rewardDict=rewardDict, ns=ns, r=r)
            rid = rewardDict[str(r)]
            ActOb = group2test(aid=aid, oid=oid, rid=rid)
            RArray[i].append((aid, oid, r))
            pv = psrModel.PassAO(pvs=pv, aos=ActOb)
            count = count + 1
    return RArray


def ConvertToTrainSet(args):
    psrModel = args[0]
    fileName = args[1]
    rewardDict = args[2]
    ID = args[3]
    epoch = args[4]
    tmpTrainData = args[5]
    tmpTrainData.newDataBatch()
    tmpTrainData = readDataintoDisk(file=fileName, TrainData=tmpTrainData, rewardDict=rewardDict)
    os.remove(fileName)
    data = tmpTrainData.data[tmpTrainData.getBatch()]
    psrModel.loadModel(epoch=epoch)
    trainData = []
    for episode in data:
        pv = psrModel.getCurrentPV()
        for d in episode:
            aid = d[0]
            oid = d[1]
            r = d[2]
            if aid == -1:
                break
            rid = rewardDict[str(r)]
            actOb = group2test(aid=aid, oid=oid, rid=rid)
            CoreTestPredictions = psrModel.generateCoreTestPrediction(pvs=pv)
            pv1 = psrModel.PassAO(pvs=pv, aos=actOb)
            CoreTestPredictions1 = psrModel.generateCoreTestPrediction(pvs=pv1)
            trainData.append((CoreTestPredictions.tolist(), aid, r, CoreTestPredictions1.tolist()))
            pv = pv1
    file = "tmp\\TrainData" + str(ID) + ".txt"
    writerMemoryintodisk(file=file, data=trainData)
    return file


def MultiProcessTrainingForest(args):
    trainSet = args[0]
    labelSet = args[1]
    forest = args[2]
    id = args[3]
    forest.fit(trainSet, labelSet)
    return forest, id


def MultiPredict(args):
    inputs = args[0]
    tree = args[1]
    id = args[2]
    Param = tree.get_params()
    Param['n_jobs'] = 1
    tree.set_params(**Param)
    output = tree.predict(inputs)
    return output, id



def checkTestID(test):
    if test is None:
        return -1
    if test in tests.keys():
        return tests[test]
    else:
        Exception("There is a test that hasn't been seen before!")
        return -1


def checkHistID(hist):
    if hist in hists.keys():
        return hists[hist] + 1
    else:
        return 0


def ListToString(lists):
    tmp = None
    for t in lists:
        if tmp is None:
            tmp = t
        else:
            tmp = tmp + t
    return tmp


def setTHCountTPSR(tid, hid, THMat, HistMat):
    if tid == -1:
        return THMat, HistMat
    THMat[tid, hid] = THMat[tid, hid] + 1
    HistMat[hid] = HistMat[hid] + 1
    return THMat, HistMat


def setTHCount(tid, hid, THMat, HistMat):
    if tid == -1:
        return THMat, HistMat
    tidVector = generateRandomVector(ids=tid, ishistory=False)
    hidVector = generateRandomVector(ids=hid, ishistory=True)
    ##############################################################
    # Counting HistVectors
    HistMat = HistMat + hidVector
    ##############################################################
    if not isinstance(tidVector, np.ndarray):
        row = np.array(tidVector)
        col = np.array(hidVector)
        col = np.transpose(a=col, axes=[1, 0])
    else:
        row = tidVector
        col = np.transpose(a=hidVector, axes=[1, 0])
    ##############################################################
    # Counting THMatrix
    ret = np.matmul(row, col)
    THMat = THMat + ret
    ##############################################################
    return THMat, HistMat


def setNullHistoryTHMats(aoSequence, THMat, HistMat):
    if len(aoSequence) > Paramter.maxTestlen:
        return THMat, HistMat
    tmp = ListToString(lists=aoSequence)
    tid = tests[tmp]
    hid = 0
    if psrType == "CPSR":
        return setTHCount(tid=tid, hid=hid, THMat=THMat, HistMat=HistMat)
    elif psrType == "TPSR":
        return setTHCountTPSR(tid=tid, hid=hid, THMat=THMat, HistMat=HistMat)
    else:
        Exception("psrType are unknown!")


def ParseAoSequenceTHMats(aoSequence, THMat, HistMat):
    hist = []
    size = len(aoSequence)
    for i in range(size):
        hist.append(aoSequence[i])
        if len(hist) > Paramter.maxHistLen:
            hist.remove(hist[0])
        # back to java program
        test = aoSequence[i + 1::]
        if len(test) > Paramter.maxTestlen or len(test) == 0:
            continue
        tmp = ListToString(lists=hist)
        tmp1 = ListToString(lists=test)
        tid = checkTestID(test=tmp1)
        hid = checkHistID(hist=tmp)
        if psrType == "CPSR":
            THMat, HistMat = setTHCount(tid=tid, hid=hid, THMat=THMat, HistMat=HistMat)
        elif psrType == "TPSR":
            THMat, HistMat = setTHCountTPSR(tid=tid, hid=hid, THMat=THMat, HistMat=HistMat)
        else:
            Exception("psrType are unknown!")
    return THMat, HistMat

def ConstructingTHMats(args):
    global tests, hists, validActObs, psrType
    filename = args[0]
    tests = args[1]
    hists = args[2]
    validActObs = args[3]
    psrType = args[4]
    ID = args[5]
    tmpTrainData = args[6]
    rewardDict = args[7]
    tmpTrainData.newDataBatch()
    tmpTrainData = readDataintoDisk(file=filename, rewardDict=rewardDict, TrainData=tmpTrainData)
    os.remove(filename)
    data = tmpTrainData.data[tmpTrainData.getBatch()]
    aoMats = dict()
    THMat = None
    HistMat = None
    if psrType == "CPSR":
        THMat = np.zeros((Paramter.ProjDim, Paramter.ProjDim + 1))
        HistMat = np.zeros((Paramter.ProjDim + 1, 1))
        for ao in validActObs:
            if ao not in aoMats.keys():
                aoMats[ao] = np.zeros((Paramter.ProjDim, Paramter.ProjDim + 1))
    elif psrType == "TPSR":
        THMat = np.zeros((len(tests), len(hists) + 1))
        HistMat = np.zeros((len(hists) + 1, 1))
        for ao in validActObs:
            if ao not in aoMats.keys():
                aoMats[ao] = np.zeros((len(tests), len(hists) + 1))
    else:
        Exception("psrType are not TPSR or CPSR")
    global randomVectorCache
    randomVectorCache = dict()
    for episode in data:
        aoSequences = []
        for d in episode:
            aid = d[0]
            oid = d[1]
            r = d[2]
            if aid != -1:
                rid = rewardDict[str(r)]
            else:
                rid = -1
            if rid is None:
                Exception("the reward:" + str(r) + "is not in rewardDict")
            Actobs = group2test(aid=aid, oid=oid, rid=rid)
            aoSequences.append(Actobs)
            THMat, HistMat = setNullHistoryTHMats(aoSequence=aoSequences, THMat=THMat, HistMat=HistMat)
            THMat, HistMat = ParseAoSequenceTHMats(aoSequence=aoSequences, THMat=THMat, HistMat=HistMat)
            aoMats = setNullHistoryAOMats(aoSequence=aoSequences, aoMats=aoMats)
            aoMats = ParseAOSequenceAOMats(aoSequence=aoSequences, aoMats=aoMats)
    for key in aoMats.keys():
        aoMats[key] = aoMats[key].tolist()
    file = "tmp//aoMats" + str(ID) + ".txt"
    writerMemoryintodisk(file=file, data=aoMats)
    return THMat, HistMat, file



def setNullHistoryAOMats(aoSequence, aoMats):
    ao = aoSequence[0]
    test = aoSequence[1::]
    strt = ListToString(lists=test)
    if len(test) > Paramter.maxTestlen:
        return aoMats
    tid = checkTestID(test=strt)
    hid = 0
    if psrType == "CPSR":
        return IncrementAOMats(tid=tid, hid=hid, aos=ao, aoMats=aoMats)
    elif psrType == "TPSR":
        return IncrementAOMatsTPSR(tid=tid, hid=hid, aos=ao, aoMats=aoMats)
    else:
        Exception("psrType are unknown")


def ParseAOSequenceAOMats(aoSequence, aoMats):
    hist = []
    size = len(aoSequence)
    for i in range(size):
        hist.append(aoSequence[i])
        if len(hist) > Paramter.maxHistLen:
            hist.remove(hist[0])
        if i + 1 >= size:
            return aoMats
        ao = aoSequence[i + 1]
        if len(ao) == 0:
            continue
        test = aoSequence[i + 2::]
        if len(test) > Paramter.maxTestlen or len(test) == 0:
            continue
        tmp = ListToString(lists=hist)
        tmp1 = ListToString(lists=test)
        hid = checkHistID(hist=tmp)
        tid = checkTestID(test=tmp1)
        if psrType == "CPSR":
            aoMats = IncrementAOMats(tid=tid, hid=hid, aos=ao, aoMats=aoMats)
        elif psrType == "TPSR":
            aoMats = IncrementAOMatsTPSR(tid=tid, hid=hid, aos=ao, aoMats=aoMats)
        else:
            Exception("psrType are unknown")
    return aoMats


def IncrementAOMatsTPSR(tid, hid, aos, aoMats):
    if tid == -1:
        return aoMats
    aoMats[aos][tid, hid] = aoMats[aos][tid, hid] + 1
    return aoMats


def IncrementAOMats(tid, hid, aos, aoMats):
    if tid == -1:
        return aoMats
    tidVector = generateRandomVector(ids=tid, ishistory=False)
    hidVector = generateRandomVector(ids=hid, ishistory=True)
    if not isinstance(tidVector, np.ndarray):
        row = np.array(tidVector)
        col = np.array(hidVector)
        col = np.transpose(a=col, axes=[1, 0])
    else:
        row = tidVector
        col = np.transpose(a=hidVector, axes=[1, 0])
    ret = np.matmul(row, col)
    aoMats[aos] = aoMats[aos] + ret
    return aoMats


def generateRandomVector(ids, ishistory):
    if ishistory:
        ids = ids + maxTestID
    if ids in randomVectorCache.keys():
        return randomVectorCache[ids]
    else:
        if ids == maxTestID:
            if Paramter.RandomInit:
                vector = np.ones((Paramter.ProjDim,))
            else:
                vector = np.zeros((Paramter.ProjDim,))
            vector = np.concatenate([[1], vector])
        else:
            seed(ids)
            vector = normal(loc=0, scale=1.0, size=Paramter.ProjDim)
            # vector = vector / float(Paramter.ProjDim)
            if ishistory:
                if Paramter.RandomInit:
                    vector = np.concatenate([[1], vector])
                else:
                    vector = np.concatenate([[0], vector])
        vector = np.expand_dims(a=vector, axis=1)
        randomVectorCache[ids] = vector
        return vector
