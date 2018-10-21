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
result = []
episode_result = []
for index in range(100):
    # funding = 10000
    for i in range(len(training)):
        if i == len(training)-1:
            break
        action = m.choose_action()

        reward = m.calculate_true_return(action, training[i])
        m.store_transition(training[i], action, reward, training[i+1])
    m.aka_learn()
    m.reset_episode()

for index in range(100):
    funding = 10000
    episode_result = [funding]
    for i in range(len(testing)):
        # if i == len(testing):
        #     break
        action = m.choose_action()

        reward = m.calculate_true_return(action, testing[i])
        m.store_test_transition(testing[i], action, reward)
        funding += funding * reward.item()
        episode_result.append(funding)
        # print("{}|{}".format(funding, reward.item()))
    result.append(episode_result)
    plot_x.append(funding)
    # print(funding)
    print("---------")

a2 = np.transpose([[0.25, 0.75]])
a2_funding = 10000
a2_result = [a2_funding]

a3 = np.transpose([[0.5, 0.5]])
a3_funding = 10000
a3_result = [a3_funding]

a4 = np.transpose([[0.75, 0.25]])
a4_funding = 10000
a4_result = [a4_funding]

for i in range(len(testing)):
    reward2 = m.calculate_true_return(a2, testing[i])
    reward3 = m.calculate_true_return(a3, testing[i])
    reward4 = m.calculate_true_return(a4, testing[i])

    a2_funding += a2_funding * reward2.item()
    a2_result.append(a2_funding)

    a3_funding += a3_funding * reward3.item()
    a3_result.append(a3_funding)

    a4_funding += a4_funding * reward4.item()
    a4_result.append(a4_funding)


max_result_index = 0
max_result = 0
for i in range(len(result)):
    if result[i][-1] > max_result:
        max_result = result[i][-1]
        max_result_index = i

print(sum(plot_x)/len(plot_x))
print(max(plot_x))
print(min(plot_x))
print(np.std(plot_x))

m.manual_plot(a2_result, a3_result, a4_result, result[max_result_index])

