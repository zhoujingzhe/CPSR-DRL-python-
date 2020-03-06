from TPSR import TransformedPSR
from PSRmodel import CompressedPSR
from TrainingData import TrainingData
import Paramter
from multiprocessing import Pool, Manager, Lock
from Util import ConvertToTrainSet
import os
import numpy as np
from MultiProcessSimulation import init
from Games.Tiger95 import Tiger95
from Games.Maze import Maze
from Games.StandTiger import StandTiger
def loadGroundTruePrediction(file, numberOfActions, numberOfObservations, gameName, rewardDict):
    groundTruePredicts = dict()
    if file == "":
        file = "likelihood_null"
    fgameName = gameName
    if Paramter.introduceReward:
        fgameName = fgameName + "Reward"
    with open(file='groundTruePredictions\\' + fgameName + "\\" + file) as f:
        for line in f:
            strs = line.split(" ")
            action = strs[0]
            observation = strs[2].split(",")[0]
            reward = None
            if Paramter.introduceReward:
                reward = strs[3].split(",")[0]
            likelihood = line.split("The likelihoods of ao is ")[-1].split("\n")[0]
            likelihood = float(likelihood)
            aid = -1
            oid = -1
            rid = -1
            for i in range(numberOfActions):
                if gameName == "Maze":
                    if Maze.Actions[i] == action:
                        aid = i
                        break
                elif gameName == "StandTiger":
                    if StandTiger.Actions[i] == action:
                        aid = i
                        break
                elif gameName == "Tiger95":
                    if Tiger95.Actions[i] == action:
                        aid = i
                        break
            for i in range(numberOfObservations):
                if gameName == "Maze":
                    if Maze.Observations[i] == observation:
                        oid = i
                        break
                elif gameName == "StandTiger":
                    if StandTiger.Observations[i] == observation:
                        oid = i
                        break
                elif gameName == "Tiger95":
                    if Tiger95.Observations[i] == observation:
                        oid = i
                        break
            if Paramter.introduceReward:
                if gameName == "StandTiger":
                    for key in rewardDict.keys():
                        if key == reward:
                            rid = rewardDict[key]
                            break
            if Paramter.introduceReward:
                if rid == -1:
                    Exception("the reward Unseen!")
                actOb = 'a' + str(aid) + 'o' + str(oid) + 'r' + str(rid)
            else:
                actOb = 'a' + str(aid) + 'o' + str(oid)
            groundTruePredicts[actOb] = likelihood
    return groundTruePredicts
def WriteFile(test, Preds, GameName, epoch, numActions, numObservations, predictive_state, rewardDict):
    if not os.path.exists("Epoch" + str(epoch)):
        os.makedirs("Epoch" + str(epoch))
    # groundTruePredictions = loadGroundTruePrediction(file=test, numberOfActions=numActions,
    #                                                  numberOfObservations=numObservations, gameName=GameName,
    #                                                  rewardDict=rewardDict)
    # loss = []
    f = open(file='Epoch' + str(epoch) + "\\" + test + '.txt', mode="w")
    f.write("PredictiveState: ")
    predictive_state = np.squeeze(a=predictive_state, axis=1)
    # predictive_state = np.squeeze(a=predictive_state, axis=-1)
    for i in predictive_state:
        f.write(str(i) + " ")
    f.write("\n")
    for aid in range(numActions):
        for oid in range(numObservations):
            for ao in Preds.keys():
                if int(ao[1]) != aid or int(ao[3]) != oid:
                    continue
                likelihood = Preds[ao]
                if GameName == "StandTiger":
                    if Paramter.introduceReward:
                        r = None
                        for key in rewardDict.keys():
                            if rewardDict[key] == int(ao[5]):
                                r = key
                                break
                        f.write(
                            StandTiger.Actions[int(ao[1])] + " " + StandTiger.Observations[int(ao[3])]
                            + " " + str(r))
                    else:
                        Exception("StandTiger need reward signals")
                else:
                    Exception("Doesn't see the game")
                likelihood = np.round(a=likelihood, decimals=2)
                f.write(" : " + str(likelihood))
                # label = groundTruePredictions[ao]
                # import math
                # kl_loss = (-label * math.log((likelihood + vars) / (label + vars)))
                # loss.append(kl_loss)
                # f.write(" KLloss: " + str(kl_loss))
                f.write("\n")
    # f.write("The sum of the loss:" + str(sum(loss)))
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
        for i in range(len(StandTiger.Actions)):
            if StandTiger.Actions[i] == action:
                aid = i
                break
        for i in range(len(StandTiger.Observations)):
            if StandTiger.Observations[i] == observation:
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
def modelQualityOnStandTiger(psrModel, StandTiger, epoch, numActions, numObservations, rewardDict):
    t = ""
    p, pv = psrModel.Predicts(test=t)
    WriteFile(test=t, Preds=p, GameName=StandTiger.getGameName(), epoch=epoch,
              numActions=numActions, numObservations=numObservations, predictive_state=pv, rewardDict=rewardDict)
    t0 = "Listen tiger-left-sit -1.0"
    t0 = EncodeStringToTest(t=t0, rewardDict=rewardDict)
    p0, pv0 = psrModel.Predicts(test=t0)
    WriteFile(test=t0, Preds=p0, GameName=StandTiger.getGameName(), epoch=epoch,
              numActions=numActions, numObservations=numObservations, predictive_state=pv0, rewardDict=rewardDict)
    t1 = "Listen tiger-left-sit -1.0,Listen tiger-left-sit -1.0"
    t1 = EncodeStringToTest(t=t1, rewardDict=rewardDict)
    p1, pv1 = psrModel.Predicts(test=t1)
    WriteFile(test=t1, Preds=p1, GameName=StandTiger.getGameName(), epoch=epoch,
              numActions=numActions, numObservations=numObservations, predictive_state=pv1, rewardDict=rewardDict)
    t2 = "Listen tiger-left-sit -1.0,Listen tiger-left-sit -1.0,Stand-Up tiger-right-stand -1.0"
    t2 = EncodeStringToTest(t=t2, rewardDict=rewardDict)
    p2, pv2 = psrModel.Predicts(test=t2)
    WriteFile(test=t2, Preds=p2, GameName=StandTiger.getGameName(), epoch=epoch,
              numActions=numActions, numObservations=numObservations, predictive_state=pv2, rewardDict=rewardDict)
    t3 = "Listen tiger-left-sit -1.0,Listen tiger-left-sit -1.0,Stand-Up tiger-right-stand -1.0,Open-Middle tiger-left-sit 30.0"
    t3 = EncodeStringToTest(t=t3, rewardDict=rewardDict)
    p3, pv3 = psrModel.Predicts(test=t3)
    WriteFile(test=t3, Preds=p3, GameName=StandTiger.getGameName(), epoch=epoch,
              numActions=numActions, numObservations=numObservations, predictive_state=pv3, rewardDict=rewardDict)
    t4 = "Listen tiger-middle-sit -1.0"
    t4 = EncodeStringToTest(t=t4, rewardDict=rewardDict)
    p4, pv4 = psrModel.Predicts(test=t4)
    WriteFile(test=t4, Preds=p4, GameName=StandTiger.getGameName(), epoch=epoch,
              numActions=numActions, numObservations=numObservations, predictive_state=pv4, rewardDict=rewardDict)
    t5 = "Listen tiger-middle-sit -1.0,Listen tiger-middle-sit -1.0"
    t5 = EncodeStringToTest(t=t5, rewardDict=rewardDict)
    p5, pv5 = psrModel.Predicts(test=t5)
    WriteFile(test=t5, Preds=p5, GameName=StandTiger.getGameName(), epoch=epoch,
              numActions=numActions, numObservations=numObservations, predictive_state=pv5, rewardDict=rewardDict)
    t6 = "Listen tiger-middle-sit -1.0,Listen tiger-middle-sit -1.0,Stand-Up tiger-middle-stand -1.0"
    t6 = EncodeStringToTest(t=t6, rewardDict=rewardDict)
    p6, pv6 = psrModel.Predicts(test=t6)
    WriteFile(test=t6, Preds=p6, GameName=StandTiger.getGameName(), epoch=epoch,
              numActions=numActions, numObservations=numObservations, predictive_state=pv6, rewardDict=rewardDict)
    t7 = "Listen tiger-middle-sit -1.0,Listen tiger-middle-sit -1.0,Stand-Up tiger-middle-stand -1.0,Open-Middle tiger-left-sit -100.0"
    t7 = EncodeStringToTest(t=t7, rewardDict=rewardDict)
    p7, pv7 = psrModel.Predicts(test=t7)
    WriteFile(test=t7, Preds=p7, GameName=StandTiger.getGameName(), epoch=epoch,
              numActions=numActions, numObservations=numObservations, predictive_state=pv7, rewardDict=rewardDict)
    t4 = "Listen tiger-right-sit -1.0"
    t4 = EncodeStringToTest(t=t4, rewardDict=rewardDict)
    p4, pv4 = psrModel.Predicts(test=t4)
    WriteFile(test=t4, Preds=p4, GameName=StandTiger.getGameName(), epoch=epoch,
              numActions=numActions, numObservations=numObservations, predictive_state=pv4, rewardDict=rewardDict)
    t5 = "Listen tiger-right-sit -1.0,Listen tiger-right-sit -1.0"
    t5 = EncodeStringToTest(t=t5, rewardDict=rewardDict)
    p5, pv5 = psrModel.Predicts(test=t5)
    WriteFile(test=t5, Preds=p5, GameName=StandTiger.getGameName(), epoch=epoch,
              numActions=numActions, numObservations=numObservations, predictive_state=pv5, rewardDict=rewardDict)
    t6 = "Listen tiger-right-sit -1.0,Listen tiger-right-sit -1.0,Stand-Up tiger-left-stand -1.0"
    t6 = EncodeStringToTest(t=t6, rewardDict=rewardDict)
    p6, pv6 = psrModel.Predicts(test=t6)
    WriteFile(test=t6, Preds=p6, GameName=StandTiger.getGameName(), epoch=epoch,
              numActions=numActions, numObservations=numObservations, predictive_state=pv6, rewardDict=rewardDict)
    t7 = "Listen tiger-right-sit -1.0,Listen tiger-right-sit -1.0,Stand-Up tiger-left-stand -1.0,Open-Left tiger-right-sit 30.0"
    t7 = EncodeStringToTest(t=t7, rewardDict=rewardDict)
    p7, pv7 = psrModel.Predicts(test=t7)
    WriteFile(test=t7, Preds=p7, GameName=StandTiger.getGameName(), epoch=epoch,
              numActions=numActions, numObservations=numObservations, predictive_state=pv7, rewardDict=rewardDict)


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
                    if Env.getGameName() == "Tiger95":
                        if r == 10:
                            winTimesEpisode = winTimesEpisode + 1
                        elif r == -100:
                            failTimesEpisode = failTimesEpisode + 1
                        else:
                            if r != -1:
                                Exception("reward" + str(r) + "are not seen")
                    elif Env.getGameName() == "Maze":
                        if r == 10.0:
                            winTimesEpisode = winTimesEpisode + 1
                        elif r == -100.0:
                            failTimesEpisode = failTimesEpisode + 1
                    elif Env.getGameName() == "StandTiger":
                        if r == 30:
                            winTimesEpisode = winTimesEpisode + 1
                        elif r == -100:
                            failTimesEpisode = failTimesEpisode + 1
                winTimes = winTimes + winTimesEpisode
                failTime = failTime + failTimesEpisode
                if winTimesEpisode + failTimesEpisode != 0:
                    lenActions.append(len(Episode) / (winTimesEpisode + failTimesEpisode))
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

from Agent import Agent
vars = sys.float_info.min
if __name__ == "__main__":
    manager = Manager()
    rewardDict = manager.dict()
    ns = manager.Namespace()
    ns.rewardCount = 0
    trainIterations = 30
    file = "Setting\\StandTiger.json"
    Paramter.readfile(file=file)
    RandomSamplingForPSR = True
    isbuiltPSR = True
    onlyBuildOnce = False
    psrType = "TPSR"
    game = StandTiger()

    #################################################
    rewardDict["-1000.0"] = 2
    rewardDict["30.0"] = 1
    rewardDict["-100.0"] = 0
    rewardDict["-1.0"] = 3
    #################################################

    game.calulateMaxTestID()
    Paramter.maxTestID = game.maxTestID
    trainData = TrainingData()
    iters = 0
    agent = Agent(PnumActions=game.getNumActions(), epilson=Paramter.epilson,
                  inputDim=(Paramter.svdDim,), algorithm=Paramter.algorithm, Parrallel=True)
    # agent = Agent(PnumActions=game.getNumActions(), epilson=Paramter.epilson, buildNet=True, game=game.getGameName())
    if Paramter.algorithm == "fitted_Q":
        print("learning algorithm is fitted Q learning")
    elif Paramter.algorithm == "DRL":
        print("learning algorithm is distributional Q-learning")
    if psrType == "TPSR":
        psrModel = TransformedPSR(game.getGameName())
    elif psrType == "CPSR" or psrType == "PSR":
        psrModel = CompressedPSR(game.getGameName())

    PSRpool = Pool(Paramter.ThreadPoolSize, initializer=init, initargs=(Paramter.maxTestID, file, Lock(), ))
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
                psrModel.validActObset = trainData.validActOb
            else:
                psrModel.build(data=trainData, aos=trainData.validActOb, pool=PSRpool, rewardDict=rewardDict)
                aos = "Open-Left tiger-middle-sit -1000.0"
                aos = EncodeStringToTest(t=aos, rewardDict=rewardDict)
                psrModel.Starting(aos=aos)
                trainSet = None
            if onlyBuildOnce:
                isbuiltPSR = False
        psrModel.saveModel(epoch=iters)
        modelQualityOnStandTiger(psrModel=psrModel, epoch=iters, StandTiger=game,
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
