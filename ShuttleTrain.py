from TPSR import TransformedPSR
from PSRmodel import CompressedPSR
from TrainingData import TrainingData
from Agent import Agent
import Paramter
from multiprocessing import Pool, Manager, Lock
from Util import ConvertToTrainSet
import os
import numpy as np
from MultiProcessSimulation import init
from Games.Shuttle import Shuttle


def WriteFile(test, Preds, GameName, epoch, numActions, numObservations, predictive_state, rewardDict):
    if not os.path.exists("Epoch" + str(epoch)):
        os.makedirs("Epoch" + str(epoch))
    f = open(file='Epoch' + str(epoch) + "\\" + test + '.txt', mode="w")
    f.write("PredictiveState: ")
    predictive_state = np.squeeze(a=predictive_state, axis=1)
    for i in predictive_state:
        f.write(str(i) + " ")
    f.write("\n")
    for aid in range(numActions):
        for oid in range(numObservations):
            for ao in Preds.keys():
                if int(ao[1]) != aid or int(ao[3]) != oid:
                    continue
                likelihood = Preds[ao]
                if GameName == "Shuttle":
                    if Paramter.introduceReward:
                        r = None
                        for key in rewardDict.keys():
                            if rewardDict[key] == int(ao[5]):
                                r = key
                                break
                        f.write(
                            Shuttle.Actions[int(ao[1])] + " " + Shuttle.Observations[int(ao[3])]
                            + " " + str(r))
                    else:
                        f.write(
                            Shuttle.Actions[int(ao[1])] + " " + Shuttle.Observations[int(ao[3])])
                else:
                    Exception("Doesn't see the game")
                likelihood = np.round(a=likelihood, decimals=2)
                f.write(" : " + str(likelihood))
                f.write("\n")
    f.close()


def EncodeStringToTest(t, rewardDict):
    lines = t.split(",")
    out = ""
    for line in lines:
        blocks = line.split(" ")
        action = blocks[0]
        observation = blocks[1]
        reward = blocks[2]
        aid = None
        oid = None
        for i in range(len(Shuttle.Actions)):
            if Shuttle.Actions[i] == action:
                aid = i
                break
        for i in range(len(Shuttle.Observations)):
            if Shuttle.Observations[i] == observation:
                oid = i
                break
        rid = rewardDict[reward]
        if aid is None or oid is None:
            Exception("action and observation are None")
        if Paramter.introduceReward:
            out = out + "a" + str(aid) + "o" + str(oid) + "r" + str(rid)
        else:
            out = out + "a" + str(aid) + "o" + str(oid)
    return out


def modelQualityOnShuttle(psrModel, Shuttle, epoch, numActions, numObservations, rewardDict):
    t = ""
    p, pv = psrModel.Predicts(test=t)
    WriteFile(test=t, Preds=p, GameName=Shuttle.getGameName(), epoch=epoch,
              numActions=numActions, numObservations=numObservations, predictive_state=pv, rewardDict=rewardDict)
    t0 = "GoForward See-nothing 0.0"
    t0 = EncodeStringToTest(t=t0, rewardDict=rewardDict)
    p0, pv0 = psrModel.Predicts(test=t0)
    WriteFile(test=t0, Preds=p0, GameName=Shuttle.getGameName(), epoch=epoch,
              numActions=numActions, numObservations=numObservations, predictive_state=pv0, rewardDict=rewardDict)
    t1 = "GoForward See-nothing 0.0,GoForward See-LRV-forward 0.0"
    t1 = EncodeStringToTest(t=t1, rewardDict=rewardDict)
    p1, pv1 = psrModel.Predicts(test=t1)
    WriteFile(test=t1, Preds=p1, GameName=Shuttle.getGameName(), epoch=epoch,
              numActions=numActions, numObservations=numObservations, predictive_state=pv1, rewardDict=rewardDict)
    t2 = "GoForward See-nothing 0.0,GoForward See-LRV-forward 0.0,GoForward See-LRV-forward 0.0"
    t2 = EncodeStringToTest(t=t2, rewardDict=rewardDict)
    p2, pv2 = psrModel.Predicts(test=t2)
    WriteFile(test=t2, Preds=p2, GameName=Shuttle.getGameName(), epoch=epoch,
              numActions=numActions, numObservations=numObservations, predictive_state=pv2, rewardDict=rewardDict)
    t3 = "GoForward See-nothing 0.0,GoForward See-LRV-forward 0.0,GoForward See-LRV-forward 0.0,TurnAround See-nothing 0.0"
    t3 = EncodeStringToTest(t=t3, rewardDict=rewardDict)
    p3, pv3 = psrModel.Predicts(test=t3)
    WriteFile(test=t3, Preds=p3, GameName=Shuttle.getGameName(), epoch=epoch,
              numActions=numActions, numObservations=numObservations, predictive_state=pv3, rewardDict=rewardDict)
    t4 = "GoForward See-nothing 0.0,GoForward See-LRV-forward 0.0,GoForward See-LRV-forward 0.0,TurnAround See-nothing 0.0,Backup See-nothing 0.0"
    t4 = EncodeStringToTest(t=t4, rewardDict=rewardDict)
    p4, pv4 = psrModel.Predicts(test=t4)
    WriteFile(test=t4, Preds=p4, GameName=Shuttle.getGameName(), epoch=epoch,
              numActions=numActions, numObservations=numObservations, predictive_state=pv4, rewardDict=rewardDict)
    t5 = "GoForward See-nothing 0.0,GoForward See-LRV-forward 0.0,GoForward See-LRV-forward 0.0,TurnAround See-nothing 0.0,Backup See-nothing 0.0,Backup See-that-we-are-docked-in-LRV 10.0"
    t5 = EncodeStringToTest(t=t5, rewardDict=rewardDict)
    p5, pv5 = psrModel.Predicts(test=t5)
    WriteFile(test=t5, Preds=p5, GameName=Shuttle.getGameName(), epoch=epoch,
              numActions=numActions, numObservations=numObservations, predictive_state=pv5, rewardDict=rewardDict)

def WriteEvalUateData(EvalData, Env, epoch):
    if not os.path.exists("Epoch" + str(epoch)):
        os.makedirs("Epoch" + str(epoch))
    with open(file="Epoch" + str(epoch) + "\\summary", mode='w') as f:
        with open(file="Epoch" + str(epoch) + "\\trajectory", mode='w') as f1:
            TotalRewards = []
            winTimes = 0
            failTime = 0
            lenActions = []
            for Episode in EvalData:
                EpisodeRewards = 0
                winTimesEpisode = 0
                failTimesEpisode = 0
                for ActOb in Episode:
                    if ActOb[0] == -1:
                        continue
                    a = Env.Actions[ActOb[0]]
                    o = Env.Observations[ActOb[1]]
                    r = ActOb[2]
                    EpisodeRewards = EpisodeRewards + r
                    f1.write(a + " " + o + " " + str(r) + ",")
                    if Env.getGameName() == "Shuttle":
                        if r == 10.0:
                            winTimesEpisode = winTimesEpisode + 1
                        elif r == -3.0:
                            failTimesEpisode = failTimesEpisode + 1
                        else:
                            Exception("reward" + str(r) + "are not seen")
                winTimes = winTimes + winTimesEpisode
                failTime = failTime + failTimesEpisode
                if winTimesEpisode + failTimesEpisode != 0:
                    lenActions.append(winTimesEpisode / (winTimesEpisode + failTimesEpisode))
                TotalRewards.append(EpisodeRewards)
                f1.write('\n')
            averageValue = np.mean(a=TotalRewards, axis=0)
            variance = np.var(TotalRewards)
            std = np.std(TotalRewards)
            if (winTimes + failTime) != 0:
                winProb = winTimes / (winTimes + failTime)
            else:
                winProb = 0
            # how many actions the agent takes before making final decision
            f.write("Average Value For Each Episode: " + str(averageValue) + '\n')
            f.write("The Variance of EpisodeReward: " + str(variance) + '\n')
            f.write("The Standard Variance of EpisodeReward: " + str(std) + '\n')
            f.write("The Winning Probability of the agent: " + str(winProb) + '\n')
            if len(lenActions) == 0:
                w = -1
            else:
                w = np.mean(a=lenActions, axis=-1)
            f.write("The steps of actions the agent takes before making final decision: " + str(w) + '\n')


def loadCheckPoint(trainData, psrModel, epoch, rewardDict):
    trainData.newDataBatch()
    TrainingData.LoadData(TrainData=trainData, file="RandomSampling0.txt", rewardDict=rewardDict)
    for i in range(epoch):
        TrainingData.LoadData(TrainData=trainData, file="epilsonGreedySampling" + str(i) + ".txt",
                              rewardDict=rewardDict)
    # psrModel.loadModel(epoch=epoch)


from Util import writerMemoryintodisk
import sys
import time
from Util import ConvertLastBatchToTrainSet

vars = sys.float_info.min
if __name__ == "__main__":
    psrType = "TPSR"
    manager = Manager()
    rewardDict = manager.dict()
    ns = manager.Namespace()
    ns.rewardCount = 0
    trainIterations = 30
    file = "Setting\\Shuttle.json"
    Paramter.readfile(file=file)
    RandomSamplingForPSR = True
    isbuiltPSR = True
    buildOnlyOnce = False
    game = Shuttle()
    #################################################
    rewardDict["10.0"] = 0
    rewardDict["-3.0"] = 1
    rewardDict["0.0"] = 2
    #################################################

    # copyRewardDict(rewardDict=rewardDict, rewardDict1=StandTiger.Rewards)
    game.calulateMaxTestID()
    Paramter.maxTestID = game.maxTestID
    trainData = TrainingData()
    iters = 0
    agent = Agent(PnumActions=game.getNumActions(), epilson=Paramter.epilson,
                  inputDim=(Paramter.svdDim,), algorithm=Paramter.algorithm, Parrallel=True)
    if Paramter.algorithm == "fitted_Q":
        print("learning algorithm is fitted Q learning")
    elif Paramter.algorithm == "DRL":
        print("learning algorithm is distributional Q-learning")
    if psrType == "CPSR" or psrType == "PSR":
        psrModel = CompressedPSR(game.getGameName())
    elif psrType == "TPSR":
        psrModel = TransformedPSR(game.getGameName())
    # loadCheckPoint(trainData=trainData, rewardDict=rewardDict, psrModel=psrModel, epoch=iters)
    PSRpool = Pool(Paramter.ThreadPoolSize, initializer=init, initargs=(Paramter.maxTestID, file, Lock(),))
    print("Finishing Preparation!")
    trainSet = None
    while iters < trainIterations:
        print("Start " + str(iters + 1) + " Iteration")
        if RandomSamplingForPSR:
            trainData.newDataBatch()
            game.SimulateTrainData(runs=Paramter.runsForCPSR, isRandom=True, psrModel=psrModel,
                                   trainData=trainData, epoch=iters - 1, pool=PSRpool,
                                   RunOnVirtualEnvironment=False, name=game.getGameName(), rewardDict=rewardDict,
                                   ns=ns)
            psrModel.validActObset = trainData.validActOb
            WriteEvalUateData(EvalData=trainData.data[trainData.getBatch()], epoch=-1, Env=game)
            trainData.WriteData(file="RandomSampling" + str(iters) + ".txt")
            RandomSamplingForPSR = False
        if isbuiltPSR:
            if psrType == "PSR":
                psrModel.loadModel(epoch=0)
            else:
                psrModel.build(data=trainData, aos=trainData.validActOb, pool=PSRpool, rewardDict=rewardDict)
                aos = "Backup See-that-we-are-docked-in-MRV 0.0"
                aos = EncodeStringToTest(t=aos, rewardDict=rewardDict)
                psrModel.Starting(aos=aos)
            if buildOnlyOnce:
                isbuiltPSR = False
        psrModel.saveModel(epoch=iters)
        modelQualityOnShuttle(psrModel=psrModel, epoch=iters, Shuttle=game,
                              numActions=game.getNumActions(), numObservations=game.getNumObservations(),
                              rewardDict=rewardDict)

        writerMemoryintodisk(file="rewardDict.txt", data=rewardDict.copy())
        print("Convert sampling data into training forms")
        if trainSet is None:
            trainSet = ConvertToTrainSet(data=trainData, RewardDict=rewardDict,
                                         pool=PSRpool, epoch=iters, name=game.getGameName(), psrModel=psrModel)
        else:
            trainSet = trainSet + ConvertLastBatchToTrainSet(data=trainData, RewardDict=rewardDict,
                                                             pool=PSRpool, epoch=iters, name=game.getGameName(),
                                                             psrModel=psrModel)
        print("start training")
        tick1 = time.time()
        agent.Train_And_Update(data=trainSet, epoch=iters, pool=PSRpool)
        tick2 = time.time()
        print("The time spent on training:" + str(tick2 - tick1))
        agent.SaveWeight(epoch=iters)
        print("Evaluating the agent")
        tick3 = time.time()
        EvalData = game.SimulateTestingRun(runs=Paramter.TestingRuns, epoch=iters, pool=PSRpool,
                                           psrModel=psrModel, name=game.getGameName(), rewardDict=rewardDict, ns=ns)
        WriteEvalUateData(EvalData=EvalData, Env=game, epoch=iters)
        tick4 = time.time()
        print("The time spent on Evaluate:" + str(tick4 - tick3))
        trainData.newDataBatch()
        game.SimulateTrainData(runs=Paramter.runsForLearning, psrModel=psrModel, trainData=trainData,
                               isRandom=False, epoch=iters, pool=PSRpool,
                               RunOnVirtualEnvironment=Paramter.TrainingOnVirtualEnvironment,
                               name=game.getGameName(), rewardDict=rewardDict, ns=ns)
        trainData.WriteData(file="epilsonGreedySampling" + str(iters) + ".txt")
        iters = iters + 1
