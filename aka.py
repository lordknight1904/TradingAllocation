from model import Model
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
rng = np.random

xls = pd.ExcelFile('./data.xlsx')
read_data = pd.read_excel(xls, 'Sheet1')
read_data = read_data.values

data = []
for d in read_data:
    data.append(np.array([[d[0], d[1]]]))
training = data[:26]
testing = data[-15:]

funding = 10000
m = Model(episode=100)
plot_x = []


for index in range(100):
    funding = 10000
    for i in range(len(training)):
        if i == len(training)-1:
            break
        action = m.choose_action()

        reward = m.calculate_true_return(action, training[i])
        m.store_transition(training[i], action, reward, training[i+1])
    m.aka_learn()

    episode_result = [funding]
    for i in range(len(testing)):
        # if i == len(testing):
        #     break
        action = m.choose_action()

        reward = m.calculate_true_return(action, testing[i])
        if i < len(testing)-1:
            m.store_transition(testing[i], action, reward, testing[i+1])
            m.aka_learn()
        funding += funding * reward.item()

        episode_result.append(funding)

    plot_x.append(funding)
    print(funding)
    print("---------")

print(sum(plot_x)/len(plot_x))
print(max(plot_x))
print(min(plot_x))
print(np.std(plot_x))