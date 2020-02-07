import Paramter
import numpy as np
from numpy.random import choice, random
import os
from Util import normalization
from sklearn.ensemble import RandomForestRegressor


class Agent:
    def __init__(self, PnumActions, epilson, inputDim, algorithm, Parrallel):
        self.projectionFunction = None
        self.loss = []
        self.Actions = list(np.arange(0, PnumActions, 1, np.int))
        self.p = list(np.ones((PnumActions,)) * (1.0 / PnumActions))
        self.numActions = PnumActions
        self.epilson = epilson
        self.algorithm = algorithm
        self.incrementalGain = (Paramter.vmax - Paramter.vmin) / (Paramter.numAtoms - 1)
        self.distribution = np.arange(Paramter.vmin, Paramter.vmax + 1, self.incrementalGain, np.float)
        self.TrainFunction = None
        self.Forest = None
        # testing Block: TrainFunction
        if algorithm == "fitted_Q":
            self.initialValue = 0
            if Parrallel:
                self.Forest = RandomForestRegressor(n_estimators=10, max_features=Paramter.svdDim, min_samples_split=3,
                                                    min_samples_leaf=5, n_jobs=10)
            else:
                self.Forest = None
        elif algorithm == "DRL":
            self.initialValue = []
            import scipy.stats
            GaussianDist = scipy.stats.norm(0, 200)
            for i in range(Paramter.numAtoms):
                self.initialValue.append(GaussianDist.pdf(self.distribution[i]))
            self.initialValue = normalization(self.initialValue)
            self.initialValue = np.expand_dims(a=self.initialValue, axis=0)
            self.Forests = []
            self.proj_fun = None
            self.genNextState = None
            for i in range(Paramter.numAtoms):
                self.Forests.append(RandomForestRegressor(n_estimators=10, max_features=Paramter.svdDim,
                                                          min_samples_split=3,
                                                          min_samples_leaf=3, n_jobs=30))

    def SaveWeight(self, epoch):
        if self.algorithm == "fitted_Q":
            if self.Forest is None:
                Exception("Forest is not built")
            import pickle
            if not os.path.exists("Epoch" + str(epoch)):
                os.makedirs("Epoch" + str(epoch))
            with open(file="Epoch" + str(epoch) + "\\model.sav", mode="wb") as f:
                pickle.dump(self.Forest, f)
        elif self.algorithm == "DRL":
            if self.Forests is None:
                Exception("Forests is not built")
            import pickle
            if not os.path.exists("Epoch" + str(epoch)):
                os.makedirs("Epoch" + str(epoch))
            for i in range(Paramter.numAtoms):
                with open(file="Epoch" + str(epoch) + "\\model" + str(i) + ".sav", mode="wb") as f:
                    pickle.dump(self.Forests[i], f)

    def LoadWeight(self, epoch):
        if self.algorithm == "fitted_Q":
            import pickle
            if os.path.exists("Epoch" + str(epoch) + "\\model.sav"):
                with open(file="Epoch" + str(epoch) + "\\model.sav", mode="rb") as f:
                    self.Forest = pickle.load(f)
                    Param = self.Forest.get_params()
                    Param['n_jobs'] = 1
                    self.Forest.set_params(**Param)
            else:
                Exception("The weight file are not existed!")
        elif self.algorithm == "DRL":
            import pickle
            for i in range(Paramter.numAtoms):
                if os.path.exists("Epoch" + str(epoch) + "\\model" + str(i) + ".sav"):
                    with open(file="Epoch" + str(epoch) + "\\model" + str(i) + ".sav", mode="rb") as f:
                        self.Forests[i] = pickle.load(f)
                        Param = self.Forests[i].get_params()
                        Param['n_jobs'] = 1
                        self.Forests[i].set_params(**Param)
                else:
                    Exception("The weight file are not existed!")

    def getGreedyAction(self, state):
        input = np.transpose(a=state, axes=[1, 0])
        expectedActions = []
        for a in self.Actions:
            A = np.ones(shape=(1, 1)) * a
            t = np.concatenate([input, A], axis=-1)
            if self.algorithm == "fitted_Q":
                expectedActions.append(self.Forest.predict(X=t)[0])
            elif self.algorithm == "DRL":
                tmpdistribution = []
                for i in range(Paramter.numAtoms):
                    tmpdistribution.append(self.Forests[i].predict(X=t)[0])
                tmpdistribution = np.array(tmpdistribution)
                stdDist = np.std(a=tmpdistribution, axis=-1) * 10
                score = np.sum(a=tmpdistribution * self.distribution, axis=0) + stdDist
                expectedActions.append(score)
        expectedActions = np.array(expectedActions)
        aid = np.argmax(a=expectedActions, axis=-1)
        return aid
    def projection(self):
        # import os
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        # os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import keras.backend as K
        import tensorflow as tf
        reward = K.placeholder(shape=(None,), dtype='float64')
        Pro_Dis = K.placeholder(shape=(None, Paramter.numAtoms), dtype='float64')
        m_prob = K.zeros(shape=(tf.shape(reward)[0], Paramter.numAtoms), dtype='float64')
        for j in range(Paramter.numAtoms):
            Tz = K.cast(x=K.minimum(x=K.cast(x=Paramter.vmax, dtype="float64"),
                                    y=K.maximum(x=K.cast(x=Paramter.vmin, dtype="float64"),
                                                y=K.cast(x=reward + Paramter.gamma * self.distribution[j],
                                                         dtype="float64"))),
                        dtype='float64')
            bj = (Tz - Paramter.vmin) / self.incrementalGain
            m_l, m_u = tf.math.floor(bj), tf.math.ceil(bj)

            m_l_id = K.reshape(x=K.cast(x=m_l, dtype='int64'), shape=(-1, 1))
            m_u_id = K.reshape(x=K.cast(x=m_u, dtype='int64'), shape=(-1, 1))
            temp = K.reshape(x=K.arange(0, K.shape(reward)[0], 1, dtype='int64'), shape=(-1, 1))
            index_m_l = K.concatenate([temp, m_l_id], axis=-1)
            index_m_u = K.concatenate([temp, m_u_id], axis=-1)
            cond = K.equal(x=m_u, y=0)
            m_u = K.cast(x=cond, dtype='float64') + m_u
            tmp1 = Pro_Dis[:, j] * (m_u - bj)
            tmp2 = Pro_Dis[:, j] * (bj - m_l)
            m_prob = m_prob + tf.scatter_nd(indices=index_m_l, updates=tmp1,
                                            shape=K.cast(x=(K.shape(reward)[0], Paramter.numAtoms), dtype='int64'))
            m_prob = m_prob + tf.scatter_nd(indices=index_m_u, updates=tmp2,
                                            shape=K.cast(x=(K.shape(reward)[0], Paramter.numAtoms), dtype='int64'))
        return K.function([reward, Pro_Dis], [m_prob])

    # def projection(self):
    #     # import os
    #     # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    #     # os.environ["CUDA_VISIBLE_DEVICES"] = ""
    #     import keras.backend as K
    #     import tensorflow as tf
    #     reward = K.placeholder(shape=(None,), dtype='float64')
    #     Pro_Dis = K.placeholder(shape=(None, Paramter.numAtoms), dtype='float64')
    #     m_prob = K.zeros(shape=(tf.shape(reward)[0], Paramter.numAtoms), dtype='float64')
    #     for j in range(Paramter.numAtoms):
    #         Tz = K.cast(x=K.minimum(x=K.cast(x=Paramter.vmax, dtype="float64"),
    #                                 y=K.maximum(x=K.cast(x=Paramter.vmin, dtype="float64"),
    #                                             y=K.cast(x=reward + Paramter.gamma * self.distribution[j],
    #                                                      dtype="float64"))),
    #                     dtype='float64')
    #         bj = (Tz - Paramter.vmin) / self.incrementalGain
    #         m_l, m_u = tf.math.floor(bj), tf.math.ceil(bj)
    #
    #         m_l_id = K.reshape(x=K.cast(x=m_l, dtype='int64'), shape=(-1, 1))
    #         m_u_id = K.reshape(x=K.cast(x=m_u, dtype='int64'), shape=(-1, 1))
    #         temp = K.reshape(x=K.arange(0, K.shape(reward)[0], 1, dtype='int64'), shape=(-1, 1))
    #         index_m_l = K.concatenate([temp, m_l_id], axis=-1)
    #         index_m_u = K.concatenate([temp, m_u_id], axis=-1)
    #         cond = K.equal(x=m_u, y=0)
    #         m_u = K.cast(x=cond, dtype='float64') + m_u
    #         tmp1 = Pro_Dis[:, j] * (m_u - bj)
    #         tmp2 = Pro_Dis[:, j] * (bj - m_l)
    #         m_prob = m_prob + tf.scatter_nd(indices=index_m_l, updates=tmp1,
    #                                         shape=(K.shape(reward)[0], Paramter.numAtoms))
    #         m_prob = m_prob + tf.scatter_nd(indices=index_m_u, updates=tmp2,
    #                                         shape=(K.shape(reward)[0], Paramter.numAtoms))
    #     return K.function([reward, Pro_Dis], [m_prob])

    def getAction(self, state):
        if state is None or random() < self.epilson:
            return self.getRandomAction()
        return self.getGreedyAction(state=state)

    def getRandomAction(self):
        return choice(a=list(self.Actions), p=list(self.p), size=1)[0]

    def generateNextState(self):
        import keras.backend as K
        import tensorflow as tf
        ExpDistsForEachAction = K.placeholder(shape=(None, self.numActions, Paramter.numAtoms), dtype='float64')
        ExpDists = ExpDistsForEachAction * self.distribution
        Score = K.sum(x=ExpDists, axis=-1)
        BestActions = K.argmax(x=Score, axis=-1)
        BestAids = K.expand_dims(x=BestActions, axis=1)
        idx = K.arange(0, K.shape(Score)[0], 1, dtype="int64")
        idx1 = K.expand_dims(x=idx, axis=1)
        indices = K.concatenate([idx1, BestAids], axis=-1)
        maxProbDist = tf.gather_nd(params=ExpDistsForEachAction, indices=indices)
        return K.function([ExpDistsForEachAction], [maxProbDist])

    def Train_And_Update(self, data, epoch, pool):
        self.TrainInFit(data=data, epoch=epoch, pool=pool)

    def TrainInFit(self, data, epoch, pool):
        data = np.array(data)
        randidx = np.arange(0, len(data), 1, np.int)
        from numpy.random import shuffle
        shuffle(randidx)
        StartStateSet = data[:, 0][randidx]
        actionSet = data[:, 1][randidx]
        rewardSet = data[:, 2][randidx]
        EndStateSet = data[:, 3][randidx]
        actionSet = np.expand_dims(a=actionSet, axis=1)
        StartStateSet = np.array(list(StartStateSet))
        StartStateSet = np.squeeze(a=StartStateSet, axis=-1)
        EndStateSet = np.array(list(EndStateSet))
        EndStateSet = np.squeeze(a=EndStateSet, axis=-1)
        trainX0 = np.concatenate([StartStateSet, actionSet], axis=-1)
        if self.algorithm == "fitted_Q":
            initialValue = np.zeros((len(trainX0),))
            self.Forest.fit(trainX0, initialValue)
        elif self.algorithm == "DRL":
            from MultiProcessSimulation import MultiProcessTrainingForest
            if self.proj_fun is None:
                self.proj_fun = self.projection()
            if self.genNextState is None:
                self.genNextState = self.generateNextState()
            initialValue = np.repeat(a=self.initialValue, repeats=len(trainX0), axis=0)
            self.ParallelTrain(trainX=trainX0, labelY=initialValue, pool=pool)
        print("training:" + str(min(int(30 * (epoch * 0.6 + 1)), Paramter.maxEpoches)) + "epoches")
        for e in range(min(int(30 * (epoch * 0.6 + 1)), Paramter.maxEpoches)):
            ExpValue1 = []
            for a in self.Actions:
                act = np.ones(shape=(len(EndStateSet), 1)) * a
                trainX1 = np.concatenate([EndStateSet, act], axis=-1)
                expValue1 = self.Predict(X=trainX1, pool=pool)
                ExpValue1.append(expValue1)
            if self.algorithm == "fitted_Q":
                ExpValue1 = np.transpose(a=ExpValue1, axes=[1, 0])
                maxValue1 = np.max(a=ExpValue1, axis=-1)
                labelValue0 = rewardSet + Paramter.gamma * maxValue1
                self.Forest.fit(trainX0, labelValue0)
            elif self.algorithm == "DRL":
                ExpDist = np.transpose(a=ExpValue1, axes=[1, 0, 2])
                # Parallel Code:
                maxProbDist = self.genNextState([ExpDist])[0]
                labelDist = self.proj_fun([rewardSet, maxProbDist])[0]
                labelDist = normalization(labelDist)
                self.ParallelTrain(trainX=trainX0, labelY=labelDist, pool=pool)
        print("Finishing Training")

    def ParallelTrain(self, trainX, labelY, pool):
        args1 = []
        for i in range(Paramter.numAtoms):
            args1.append([trainX, labelY[:, i], self.Forests[i], i])
        from MultiProcessSimulation import MultiProcessTrainingForest
        outputs = pool.map(func=MultiProcessTrainingForest, iterable=args1)
        for forest, idx in outputs:
            self.Forests[idx] = forest

    def Predict(self, X, pool):
        if self.algorithm == "fitted_Q":
            return self.Forest.predict(X)
        elif self.algorithm == "DRL":
            args = []
            for i in range(Paramter.numAtoms):
                args.append([X, self.Forests[i], i])
            from MultiProcessSimulation import MultiPredict
            outputs = pool.map(func=MultiPredict, iterable=args)
            value = []
            for i in range(Paramter.numAtoms):
                for output in outputs:
                    if output[1] == i:
                        value.append(output[0])
            value = np.transpose(a=value, axes=[1, 0])
            value = normalization(value)
            return value
