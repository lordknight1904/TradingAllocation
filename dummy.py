import numpy as np
import matplotlib.pyplot as plt


ret = []
x = np.linspace(0,40)
years = [1,2,3,4,5]

axes = plt.gca()
axes.set_xlim(0, 40)
# plt.scatter(years, ret, c='b')
print(x)
print(0.5*x + 1)
plt.plot(x, 0.5*x + 1)
plt.draw()
plt.pause(1e-17)
plt.show()