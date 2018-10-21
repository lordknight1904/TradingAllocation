from Qmodel import QModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
rng = np.random

xls = pd.ExcelFile('./data.xlsx')
read_data = pd.read_excel(xls, 'Sheet1')
read_data = read_data.values

data = []
value = []
for d in read_data:
    value.append([[d[0], d[1]]])
    data.append(int(d[0] < 0) * 1 + int(d[1] < 0) * 2)
training = data[:26]
testing = data[-15:]

m = QModel(state_space=4, action_space=5)
episode = 1000

plot_x = []
for index in range(1):
    m.initialize()
    for ep in range(episode):
        for s, s_, v in zip(training[:-1], training[1:], value[:25]):
            # Choose at from st using policy derived from Q(ε−greedy)
            action = m.choose_action(s)
            # Take action at, observe rt and st+1
            ret = m.invest(action, v)
            # Choose at+1 from st+1 using policy derived from Q(ε−greedy)
            # next_action = m.choose_action(s_)
            # print("state: {}".format(s))
            # print("action: {}".format(action))
            # print("return: {}".format(ret))
            # input()

            m.train(s, action, ret, s_)
            # input()

    funding = 10000
    for s, v in zip(testing, value[-15:]):
        action = m.choose_action(s)
        ret = m.invest(action, v)
        print(ret)
        funding += funding * ret
    plot_x.append(funding)
    print(funding)
    print("----------")

print(sum(plot_x)/len(plot_x))
print(max(plot_x))
print(min(plot_x))
print(np.std(plot_x))