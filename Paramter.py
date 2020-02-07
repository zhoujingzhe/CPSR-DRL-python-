
global numAtoms, learningRate, decay, vmin, vmax, gamma, alpha, TrainingOnVirtualEnvironment, \
    epilson, algorithm, svdDim, ProjDim, maxTestID, maxTestlen, maxHistLen, introduceReward, RandomInit, \
    LengthOfAction, runsForLearning, runsForCPSR, TestingRuns, SimulateBatch, ThreadPoolSize, lr_min, maxEpoches
import json
def readfile(file):
    with open(file, 'r') as f:
        Params = json.load(f)
        global numAtoms, learningRate, decay, vmin, vmax, gamma, alpha, TrainingOnVirtualEnvironment,\
            epilson, algorithm, svdDim, ProjDim, maxTestID, maxTestlen, maxHistLen, introduceReward, RandomInit,\
            LengthOfAction, runsForLearning, runsForCPSR, TestingRuns, SimulateBatch, ThreadPoolSize, lr_min, maxEpoches
        numAtoms = Params["numAtoms"]
        learningRate = Params["learningRate"]
        decay = Params["decay"]
        vmin = float(Params["vmin"])
        vmax = float(Params["vmax"])
        gamma = Params["gamma"]
        alpha = Params["alpha"]
        TrainingOnVirtualEnvironment = Params["TrainingOnVirtualEnvironment"]
        epilson = Params["epilson"]
        algorithm = Params["algorithm"]
        maxEpoches = Params["maxEpoch"]
        ########################################################
        # PSR setting
        svdDim = Params["svdDim"]
        ProjDim = Params["ProjDim"]
        maxTestlen = Params["maxTestlen"]
        maxHistLen = Params["maxHistLen"]
        introduceReward = Params["introduceReward"]
        ########################################################
        RandomInit = Params["RandomInit"]
        LengthOfAction = Params["LengthOfAction"]
        runsForLearning = Params["runsForLearning"]
        runsForCPSR = Params["runsForCPSR"]
        TestingRuns = Params["TestingRuns"]
        maxTestID = Params["maxTestID"]
        ThreadPoolSize = Params["ThreadPoolSize"]
        lr_min = Params["minimum_learningRate"]

def writefile(file):
    Params = dict()
    Params["numAtoms"] = numAtoms
    Params["learningRate"] = learningRate
    Params["decay"] = decay
    Params["vmin"] = vmin
    Params["vmax"] = vmax
    Params["gamma"] = gamma
    Params["alpha"] = alpha
    Params["TrainingOnVirtualEnvironment"] = TrainingOnVirtualEnvironment
    Params["epilson"] = epilson
    Params["algorithm"] = algorithm
    ########################################################
    # PSR setting
    Params["svdDim"] = svdDim
    Params["ProjDim"] = ProjDim
    Params["maxTestlen"] = maxTestlen
    Params["maxHistLen"] = maxHistLen
    Params["introduceReward"] = introduceReward
    ########################################################
    Params["RandomInit"] = RandomInit
    Params["LengthOfAction"] = LengthOfAction
    Params["runsForLearning"] = runsForLearning
    Params["runsForCPSR"] = runsForCPSR
    Params["TestingRuns"] = TestingRuns
    Params["maxTestID"] = maxTestID
    Params["ThreadPoolSize"] = ThreadPoolSize
    Params["minimum_learningRate"] = lr_min
    Params["maxEpoch"] = maxEpoches
    f = open(file=file, mode='w')
    json.dump(Params, f)
    f.close()

if __name__ == "__main__":
    writefile(file="Setting\\Maze.json")