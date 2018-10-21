import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

rng = np.random


class Model:
    def __init__(self, episode=1000, epsilon=0.01, gamma=0.9, lamda=0.9, eta=0.1, alpha=0.1):
        # self.data_set = data_set

        self.epsilon = epsilon
        self.gamma = gamma
        self.lamda = lamda
        self.eta = eta
        self.alpha = alpha

        self.theta = np.transpose(np.array([[np.random.uniform(), np.random.uniform()]]))

        self.episode = episode
        self.episode_observations, self.episode_actions, self.episode_rewards, self.episode_states = [], [], [], []
        self.test_actions, self.test_rewards, self.test_states = [], [], []

        self.true_return, self.rl_return = [], []

    @staticmethod
    def __rl_return(self, a, s):
        temp = np.ones_like(s)
        temp[0][0] = (s[0][0] - s[0][1]) / 100
        return np.matmul(temp, a)

    # @staticmethod
    def calculate_true_return(self, a, s):
        temp = a
        temp[1][0] = 1 - temp[0][0]
        return np.matmul(s / 100, temp)

    @staticmethod
    def __calculate_nabla(self, s):
        temp = np.ones_like(s)
        temp[0][0] = (s[0][0] - s[0][1]) / 100
        return np.transpose(temp)

    def choose_action(self):
        action = self.theta
        if rng.uniform() < self.epsilon:
            action[0][0] = rng.uniform()
        return action

    def store_transition(self, s, a, r, s_):
        self.episode_states.append(s)
        self.episode_actions.append(a)
        self.episode_rewards.append(r)
        self.episode_observations.append(s_)

    def store_test_transition(self, s, a, r):
        self.test_states.append(s)
        self.test_actions.append(a)
        self.test_rewards.append(r)

    def reset_test_episode(self):
        self.test_actions, self.test_rewards, self.test_states = [], [], []

    def reset_episode(self):
        self.episode_observations, self.episode_actions, self.episode_rewards, self.episode_states = [], [], [], []

    def ska_learn(self):
        s = self.episode_states[-1]
        a = self.episode_actions[-1]
        r = self.episode_rewards[-1]
        s_= self.episode_observations[-1]

        e = np.transpose(np.array([[0, 0]]))
        rl_ret = self.__rl_return(self, a, s)
        rl_ret_next = self.__rl_return(self, a, s_)

        nabla = self.__calculate_nabla(self, s)

        delta = r + self.gamma * rl_ret_next - rl_ret
        e = self.gamma * self.lamda * e + nabla
        self.theta += self.alpha * delta * e
        if self.theta[0][0] > 1:
            self.theta[0][0] = 1
        if self.theta[0][0] < 0:
            self.theta[0][0] = 0
        # years.append(index)
        self.true_return.append(r)
        self.rl_return.append(rl_ret)

    def aka_learn(self):
        e = np.transpose(np.array([[0, 0]]))
        for i in range(self.episode):
            for s, a, r, s_ in zip(self.episode_states, self.episode_actions, self.episode_rewards,
                                   self.episode_observations):
                # true_ret = self.__true_return(a, s)
                rl_ret = self.__rl_return(self, a, s)
                rl_ret_next = self.__rl_return(self, a, s_)

                nabla = self.__calculate_nabla(self, s)

                delta = r + self.gamma * rl_ret_next - rl_ret
                e = self.gamma * self.lamda * e + nabla
                self.theta += self.alpha * delta * e
                if self.theta[0][0] > 1:
                    self.theta[0][0] = 1
                if self.theta[0][0] < 0:
                    self.theta[0][0] = 0
                # years.append(index)
                self.true_return.append(r)
                self.rl_return.append(rl_ret)

    def plot(self):
        import matplotlib.pyplot as plt

        rl_value = []
        for a, s in zip(self.test_actions, self.test_states):
            rl_value.append(self.__rl_return(self, a, s))
        # setting plot
        axes = plt.gca()
        axes.set_ylim(0, 1)
        axes.set_xlim(0, 41)
        # plot true value and agent value
        plt.scatter(np.arange(len(self.test_rewards)), self.test_rewards, c='b')
        plt.scatter(np.arange(len(self.test_actions)) + len(self.test_rewards), rl_value, c='g')
        # linear regression line
        x = np.linspace(0, 41)
        plt.plot(x, self.theta[0][0] * x + self.theta[1][0])

        plt.ylabel('Value')
        plt.xlabel('Year')
        plt.show()

    def manual_plot(self, a2, a3, a4, line):
        import matplotlib.pyplot as plt

        # setting plot
        axes = plt.gca()
        # axes.set_ylim(0, 1)
        axes.set_xlim(0, len(a4))

        # linear regression line
        # x = np.linspace(0, len(a2))
        plt.plot(np.arange(len(a2)), a2, 'b')
        plt.plot(np.arange(len(a3)), a3, 'g')
        plt.plot(np.arange(len(a4)), a4, 'r')
        plt.plot(np.arange(len(a4)), line, 'k')

        plt.gca().legend(('A2', 'A3', 'A4', 'CA_SKA'))
        plt.ylabel('Value')
        plt.xlabel('Year')
        plt.show()