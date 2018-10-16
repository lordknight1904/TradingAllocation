import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
rng = np.random


def rl_return(a, s):
    temp = np.ones_like(s)
    temp[0][0] = (s[0][0] - s[0][1]) / 100
    return np.matmul(temp, a)


def true_return(a, s):
    temp = a
    temp[1][0] = 1 - temp[0][0]
    return np.matmul(s/100, temp)


def calculate_nabla(s):
    temp = np.ones_like(s)
    temp[0][0] = (s[0][0] - s[0][1]) / 100
    return np.transpose(temp)


xls = pd.ExcelFile('./data.xlsx')
data = pd.read_excel(xls, 'Sheet1')
data = data.values

l_data = []
for d in data:
    l_data.append(np.array([[d[0], d[1]]]))

gamma = 0.9
lamda = 0.9
epsilon = 0.01
eta = 0.1
alpha = 0.1

episode_number = 1
# print(theta)
funding = 10000

episode_action = []
episode_reward = []
episode_observation = []

training = l_data[:25]
testing = l_data[-16:]

ret = []
ret_rl = []
years = []
plt.show()


def learn(e_a, e_s, e_s_):
    temp = []
    for index, (a, s, s_) in enumerate(zip(e_a, e_s, e_s_)):
        e = np.transpose(np.array([[0, 0]]))
        true_ret = true_return(a, s)
        rl_ret = rl_return(a, s)
        rl_ret_next = rl_return(a, s_)

        nabla = calculate_nabla(s)

        delta = true_ret + gamma * rl_ret_next - rl_ret

        # e = gamma * lamda * e + nabla * rl_ret
        e = gamma * lamda * e + nabla
        temp.append(alpha * delta * e)

        years.append(index)
        ret.append(true_return(theta, l_data[index]).item())
        ret_rl.append(rl_return(theta, l_data[index]).item())
    return temp


x = np.linspace(0,40)
hl, = plt.plot([], [])
plot_x = []
axes = plt.gca()
axes.set_ylim(0, 10)
axes.set_xlim(0, 40)
# line, = axes.plot(x, theta[0][0]*x + theta[1][0], 'r-')

for abc in range(1):
    funding = 10000
    theta = np.transpose(np.array([[np.random.uniform(), np.random.uniform()]]))
    # print(theta)
    for index in range(25, len(l_data)):
        data_set = l_data[:index]
        # for i in range(episode_number):
        episode_action = []
        episode_reward = []
        episode_observation = []
        for j, s in enumerate(zip(data_set)):
            print(np.squeeze(np.transpose(theta)))
            ret = []
            ret_rl = []
            years = []
            if j == len(data_set) - 1:
                break
            action = theta
            if rng.uniform() < epsilon:
                action[0][0] = rng.uniform()

            # print("action: {}".format(action))
            episode_action.append(action)
            episode_reward.append(s[0])
            episode_observation.append(data_set[:][j + 1])

            temp = learn(episode_action, episode_reward, episode_observation)
            print(sum(temp)/len(temp))
            theta += sum(temp)/len(temp)
            if theta[0][0] > 1:
                theta[0][0] = 1.
            if theta[0][0] < 0:
                theta[0][0] = 0.
                # input("Press Enter to continue...")
                # if j == i:
                #     continue

            plt.scatter(years, ret, c='b')
            plt.scatter(years, ret_rl, c='g')
            # plt.plot(x, theta[0][0]*x + theta[1][0])
            hl.set_xdata(x)
            hl.set_ydata(theta[0][0]*x + theta[1][0])
            plt.draw()
            plt.pause(1e-17)
            time.sleep(1)
            # input()

        hl.set_xdata(x)
        hl.set_ydata(theta[0][0]*x + theta[1][0])
        if index >= 25:
            funding += funding * true_return(theta, l_data[index]).item()
    plot_x.append(funding)
    print(funding)
    print("------------")
print('here')
print(sum(plot_x)/len(plot_x))
print(max(plot_x))
print(min(plot_x))
print(np.std(plot_x))
# plt.show()
# for j, s in enumerate(zip(testing)):
#     if j == 16:
#         break
#
#     funding += funding * true_return(theta, s[0]).item()
#
#     episode_action.append(theta)
#     episode_reward.append(s[0])
#     episode_observation.append(testing[:][j + 1])
#
#     temp = learn(episode_action, episode_reward, episode_observation, theta)
#     theta += sum(temp) / len(temp)
#     if theta[0][0] > 1:
#         theta[0][0] = 1.
#     if theta[0][0] < 0:
#         theta[0][0] = 0.

# print(funding)
