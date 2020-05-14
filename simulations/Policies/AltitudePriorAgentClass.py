# Foundations
import numpy as nu
import pandas as pd
# AI modules
from tensorflow import keras as kr


class AltitudePriorAgent():
    def __init__(self, env):
        self.env = env
        self.actionAmount = len(env.possibleActions)
        self.stateAmount =
        self.mainNetwork = self.__buildDQN()
        self.targetNetwork = self.__buildDQN()


    def train(self, episodeAmount=50, epsilon=0.9, epsilonDecayRate=0.98, discountRate=0.99, batchSize=10, maxSteps=1000, syncSteps=10, shouldPrint=True):
        historyRewards = []
        memories = []
        for episodeCount in range(episodeAmount):
            # initiate state
            state = self.env.reset()
            didEnd = False
            stepCount = 0
            episodeRewards = 0
            # main loop
            while not didEnd and stepCount < maxSteps:
                stepCount += 1
                # get next action
                if nu.random.random() <= epsilon:  # use random policy
                    actionChosen = nu.random.choice(self.env.possibleActions)
                else:  # use model policy
                    actionIndexChosen = nu.argmax(self.mainNetwork.predict(state))
                    actionChosen = self.env.possibleActions[actionIndexChosen]
                # step forwards
                newState, didEnd, reward = self.env.step(actionChosen)
                # add to memories
                memories.append((state, action, reward, newState, didEnd))
                # randomly choose from memories
                if len(memories) <= batchSize:
                    memoriesToBeUsedToTrain = memories
                else:
                    memoriesToBeUsedToTrain = nu.random.sample(memories, size=batchSize)
                # train
                self.__trainMainNetwork(memoriesToBeUsedToTrain)
                # sync target and main network
                if stepCount % syncSteps == 0:
                    self.targetNetwork.set_weights(self.mainNetwork.get_weights())
                # prepare for next loop
                state = newState
                episodeRewards += reward
                epsilon *= epsilonDecayRate
            historyRewards.append(totalRewards)
            if shouldPrint:
                print(f'Episode: {episodeCount+1} done =======================================================\n')


    def __buildDQN(self):
        model = kr.Sequential()
        model.add(kr.layers.Conv3D(filters=50, kernel_size=(3, 3, 3), activation='relu', input_shape=self.env.sensorArray.readToState().shape))
        model.add(kr.layers.MaxPooling3D((2, 2, 2)))
        model.add(kr.layers.Conv3D(filters=30, kernel_size=(3, 3, 3), activation='relu'))
        model.add(kr.layers.MaxPooling3D((2, 2, 2)))
        model.add(kr.layers.Conv3D(filters=10, kernel_size=(3, 3, 3), activation='relu'))
        model.add(kr.layers.Flatten())
        model.add(kr.layers.Dense(100, activation='relu'))
        model.add(kr.layers.Dense(len(self.env.possibleActions), activation='relu'))
        model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])
        return model


    " https://gist.github.com/pocokhc/0ddb7c28608523d3fa692c5b0c7d3bb0 "
    def __trainMainNetwork(self, memories):
        batchSize = len(memories)
        nextExpectedBestRewardArray = nu.zeros(batchSize)
        statesForTraining = []
        for (index, (state, action, reward, newState, didEnd)) in enumerate(memoriesToBeUsedToTrain):
            statesForTraining.append(state)
            predictedMainQs = self.mainNetwork.predict(newState)
            predictedTargetQs = self.targetNetwork.predict(newState)
            nextExpectedReward = predictedTargetQs[nu.argmax(predictedMainQs)]
            adjustedReward = reward + discountRate * nextExpectedReward
            nextExpectedBestRewardArray[index] = adjustedReward
        self.mainNetwork.train_on_batch(statesForTraining, nextExpectedBestRewardArray)
