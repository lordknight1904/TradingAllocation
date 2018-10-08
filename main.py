import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
rng = np.random


def rl_return(a1, a2, snp, agg):
    return a1 * (snp - agg) /100 + a2


def true_return(a, snp, agg):
    return a * snp / 100 + (1 - a) * agg / 100


def calculate_nabla(snp, agg):
    return np.transpose([[(snp - agg) / 100, 1]])


xls = pd.ExcelFile('./data.xlsx')
data = pd.read_excel(xls, 'Sheet1')
data = data.values
# snp = data[:, 0]
# agg = data[:, 1]

state = np.zeros((len(data[:, 0]),2))
for i, (s, a) in enumerate(zip(data[:, 0], data[:, 1])):
    state[i][0] = s>0
    state[i][1] = a>0

# print(state)
# print(data)

gamma = 0.9
lamda = 0.9
epsilon = 0.01
eta = 0.1
alpha = 0.1

episode_number = 1000


theta = np.transpose(np.array([[np.random.uniform(), np.random.uniform()]]))

funding = 10000

episode_action = []
episode_reward = []
episode_observation = []

training = data[:30]
testing = data[-17:]


def learn(e_a, e_s, e_s_):
    theta = []
    e = np.transpose(np.array([[0, 0]]))
    for (a, s, s_) in zip(e_a, e_s, e_s_):

        true_ret = true_return(a, s[0], s[1])
        rl_ret = rl_return(a, a[1][0], s[0], s[1])
        rl_ret_next = rl_return(a, a[1][0], s_[0], s_[1])

        nabla = calculate_nabla(s[0], s[1])

        delta = true_ret + gamma * rl_ret_next - rl_ret

        e = gamma * lamda * e + nabla * rl_ret
        temp = alpha * delta * e
        if temp[0][0] >= 1:
            temp[0][0] = 1.
        if temp[0][0] <= 0:
            temp[0][0] = 0.
        print(temp)
        theta.append(temp)
    return theta


for i in range(episode_number):
    episode_action = []
    episode_reward = []
    episode_observation = []
    for j, s in enumerate(zip(training)):
        if j == 29:
            break
        a = theta[0][0]
        if rng.uniform() < epsilon:
            a = rng.uniform()

        episode_action.append(theta)
        episode_reward.append(s[0])
        episode_observation.append(training[:][j + 1])

        delta_theta = learn(episode_action, episode_reward, episode_observation)
        for t in delta_theta:
            theta += t

        print(theta[0][0])
        # input("Press Enter to continue...")
        if j == i:
            continue

for j, s in enumerate(zip(testing)):
    if j == 16:
        break
    a = theta[0][0]
    if rng.uniform() < epsilon:
        a = rng.uniform()

    true_ret = true_return(theta[0][0], s[0][0], s[0][1])
    funding += funding * true_ret
    print(theta[0][0])
    print(funding)
    print("------------------")
    episode_action.append(theta)
    episode_reward.append(s[0])
    episode_observation.append(testing[:][j + 1])

    delta_theta = learn(episode_action, episode_reward, episode_observation)
    # for t in delta_theta:
    #     theta += t
    # input("Press Enter to continue...")

# for (snp, agg) in zip(testing[:, 0], testing[:, 1]):
#     true_ret = true_return(theta[0][0], snp, agg)
#     # print(theta[0][0])
#     # print(true_ret)
#     print(funding)
#     print("------------------")
#     funding += funding * true_ret
print(funding)