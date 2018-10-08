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

episode_number = 5000


theta = np.transpose(np.array([[np.random.uniform(), np.random.uniform()]]))

funding = 10000

training = data[:30]
testing = data[-17:]

for i in range(episode_number):
    e = np.transpose(np.array([[0, 0]]))
    for j, (snp, agg) in enumerate(zip(training[:, 0], training[:, 1])):
        if j == 29:
            break
        a = theta[0][0]
        if rng.uniform() < epsilon:
            a = rng.uniform()


        true_ret = true_return(a, snp, agg)
        rl_ret = rl_return(a, theta[1][0], snp, agg)

        nabla = calculate_nabla(snp, agg)
        delta = true_ret + gamma * rl_return(a, theta[1][0], training[:, 0][j + 1], training[:, 1][j + 1]) - rl_ret
        e = gamma * lamda * e + nabla * rl_ret
        theta += alpha * delta * e

        if theta[0][0] >= 1:
            theta[0][0] = 1.
        if theta[0][0] <= 0:
            theta[0][0] = 0.
        # print(theta)
        # print("------------------")
        if j == i:
            continue
for (snp, agg) in zip(testing[:, 0], testing[:, 1]):
    true_ret = true_return(theta[0][0], snp, agg)
    # print(theta[0][0])
    # print(true_ret)
    print(funding)
    print("------------------")
    funding += funding * true_ret
print(funding)