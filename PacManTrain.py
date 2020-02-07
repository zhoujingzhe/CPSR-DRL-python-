from PSRmodel import CompressedPSR
from TrainingData import TrainingData

import Paramter
from multiprocessing import Pool, Manager, Lock
from Util import ConvertToTrainSet
import os
import numpy as np
from MultiProcessSimulation import init

def WriteEvalUateDataForPacMan(EvalData, epoch):
    if not os.path.exists("Epoch" + str(epoch)):
        os.makedirs("Epoch" + str(epoch))
    with open(file="Epoch" + str(epoch) + "\\summary", mode='w') as f:
        TotalRewards = []
        lenActions = []
        count_wins = 0
        for Episode in EvalData:
            EpisodeRewards = 0
            EpisodeLength = 0
            for ActOb in Episode:
                a = ActOb[0]
                if a == -1:
                    break
                r = ActOb[2]
                EpisodeRewards = EpisodeRewards + r
                EpisodeLength = EpisodeLength + 1
                if r >= 90:
                    count_wins = count_wins + 1
            lenActions.append(EpisodeLength)
            TotalRewards.append(EpisodeRewards)
        averageValue = np.mean(a=TotalRewards, axis=0)
        variance = np.var(TotalRewards)
        std = np.std(TotalRewards)
        # how many actions the agent takes before making final decision
        f.write("Average Value For Each Episode: " + str(averageValue) + '\n')
        f.write("The Variance of EpisodeReward: " + str(variance) + '\n')
        f.write("The Standard Variance of EpisodeReward: " + str(std) + '\n')
        f.write("The average length of a game is: " + str(np.mean(a=lenActions, axis=-1)) + "\n")
        if count_wins != 0:
            f.write("The wining Probability:" + str(len(TotalRewards) / count_wins))
        else:
            f.write("The wining Probability:" + str(-1))


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
                    r = Env.Rewards[ActOb[2]]
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
                    lenActions.append(Paramter.LengthOfAction / (winTimesEpisode + failTimesEpisode))
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
        trainData.newDataBatch()
        TrainingData.LoadData(TrainData=trainData, file="epilsonGreedySampling" + str(i) + ".txt",
                              rewardDict=rewardDict)
    # psrModel.loadModel(epoch=epoch)


import sys
from Games.PacMan import PacMan
import time
from Util import ConvertLastBatchToTrainSet, readMemoryfromdisk
vars = sys.float_info.min
# from Agent import Agent
from AgentNet import Agent
if __name__ == "__main__":
    manager = Manager()
    rewardDict = manager.dict()
    ns = manager.Namespace()
    ns.rewardCount = 0
    trainIterations = 30
    file = "Setting\\PacMan.json"
    Paramter.readfile(file=file)
    RandomSamplingForPSR = True
    isbuiltPSR = True
    onlyBuildOnce = True
    LoadModel = False
    game = PacMan()
    game.calulateMaxTestID()
    Paramter.maxTestID = game.maxTestID
    trainData = TrainingData()
    iters = 0
    # agent = Agent(PnumActions=game.getNumActions(), epilson=Paramter.epilson,
    #               inputDim=(Paramter.svdDim,), algorithm=Paramter.algorithm, Parrallel=True)
    agent = Agent(PnumActions=game.getNumActions(), epilson=Paramter.epilson, buildNet=True, game=game.getGameName())
    if Paramter.algorithm == "fitted_Q":
        print("learning algorithm is fitted Q learning")
    elif Paramter.algorithm == "DRL":
        print("learning algorithm is distributional Q-learning")

    psrModel = CompressedPSR(game.getGameName())
    if LoadModel:
        psrModel.loadModel(epoch=-2)
    PSRpool = Pool(Paramter.ThreadPoolSize, initializer=init, initargs=(Paramter.maxTestID, file, Lock(), ))
    ##########################################################################
    rewardDict = readMemoryfromdisk(file="rewardDict.txt")
    ns.rewardCount = len(rewardDict)
    ##########################################################################
    loadCheckPoint(trainData=trainData, psrModel=psrModel, epoch=iters, rewardDict=rewardDict)
    trainSet = None
    print("Finishing Preparation!")
    while iters < trainIterations:
        print("Start " + str(iters + 1) + " Iteration")
        if RandomSamplingForPSR:
            trainData.newDataBatch()
            game.SimulateTrainData(runs=Paramter.runsForCPSR, isRandom=True, psrModel=psrModel,
                                   trainData=trainData, epoch=iters - 1, pool=PSRpool,
                                   RunOnVirtualEnvironment=False, name=game.getGameName(), rewardDict=rewardDict,
                                   ns=ns)
            psrModel.validActObset = trainData.validActOb
            WriteEvalUateDataForPacMan(EvalData=trainData.data[trainData.getBatch()], epoch=-1)
            trainData.WriteData(file="RandomSampling" + str(iters) + ".txt")
            RandomSamplingForPSR = False
        if isbuiltPSR:
            psrModel.build(data=trainData, aos=trainData.validActOb, pool=PSRpool, rewardDict=rewardDict)
            if onlyBuildOnce:
                isbuiltPSR = False
        psrModel.saveModel(epoch=iters)
        from Util import writerMemoryintodisk
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
        if not onlyBuildOnce:
            trainSet = None
        tick2 = time.time()
        print("The time spent on training:" + str(tick2 - tick1))
        agent.SaveWeight(epoch=iters)
        print("Evaluating the agent")
        tick3 = time.time()
        EvalData = game.SimulateTestingRun(runs=Paramter.TestingRuns, epoch=iters, pool=PSRpool,
                                           psrModel=psrModel, name=game.getGameName(), rewardDict=rewardDict, ns=ns)
        tick4 = time.time()
        print("The time spent on Evaluate:" + str(tick4 - tick3))
        trainData.newDataBatch()
        game.SimulateTrainData(runs=Paramter.runsForLearning, psrModel=psrModel, trainData=trainData,
                               isRandom=False, epoch=iters, pool=PSRpool,
                               RunOnVirtualEnvironment=Paramter.TrainingOnVirtualEnvironment,
                               name=game.getGameName(), rewardDict=rewardDict, ns=ns)
        writerMemoryintodisk(file="rewardDict.txt", data=rewardDict.copy())
        trainData.WriteData(file="epilsonGreedySampling" + str(iters) + ".txt")
        WriteEvalUateDataForPacMan(EvalData=EvalData, epoch=iters)
        iters = iters + 1
