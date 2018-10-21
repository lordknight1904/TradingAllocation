from model import Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
rng = np.random

episode = 10000


class QModel:
    def __init__(self, state_space, action_space, epsilon=0.01, gamma=0.9, lamda=0.9, eta=0.1, alpha=0.1):
        # Initialize Q(s, a) randomly and e(s,a) = 0 for all s,a
        self.state_space = state_space
        self.action_space = action_space
        self.q = np.random.rand(state_space, action_space)
        self.e = np.zeros((state_space, action_space))

        self.epsilon = epsilon
        self.gamma = gamma
        self.lamda = lamda
        self.eta = eta
        self.alpha = alpha
        self.actions = np.array([
            [[0, 1.0]],
            [[0.25, 0.75]],
            [[0.5, 0.5]],
            [[0.75, 0.25]],
            [[1.0, 0]]
        ])
        self.episode_observations, self.episode_actions, self.episode_rewards, self.episode_states = [], [], [], []

    def initialize(self):
        self.q = np.random.rand(self.state_space, self.action_space)
        self.e = np.zeros((self.state_space, self.action_space))

    def store_transition(self, s, a, r, s_):
        self.episode_states.append(s)
        self.episode_actions.append(a)
        self.episode_rewards.append(r)
        self.episode_observations.append(s_)

    def reset_episode(self):
        self.episode_observations, self.episode_actions, self.episode_rewards, self.episode_states = [], [], [], []

    def choose_action(self, s):
        action = np.argmax(self.q[s])
        if rng.uniform() < self.epsilon:
            action = rng.randint(0, 4)
        return action

    def invest(self, action, value):
        # print(self.actions[action])
        # print(np.transpose(value))
        return np.matmul(self.actions[action], np.transpose(value)).item() / 100

    def train(self, s, a, r, s_):
        # optimal action
        optimal_action = np.argmax(self.q[s_])
        # δ ← rt + γQ(st + 1, a* ) − Q(st,at)
        delta = r + self.gamma * self.q[s_][optimal_action] - self.q[s][a]
        # e(st,at) ← 1
        self.e[s][a] = 1
        for s in range(self.state_space):
            for a in range(self.action_space):
                self.q[s][a] += self.alpha * delta * self.e[s][a]
                if self.choose_action(s_) == optimal_action:
                    self.e[s][a] *= self.gamma * self.lamda
                else:
                    self.e[s][a] = 0
        # For all s,a:
        #   Q(s, a) ← Q(s, a) + αδe(s, a)
        #   If at + 1 = a*,  then e(s, a) ← γλe(s, a)
        #   else e(s, a) ← 0
        # End For

    def plot(self):
        print(self.q)
