import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
import pandas as pd


img_dpi = 300

n = 100
x = random.rand(n) / 2
y = [z + (random.rand() - 0.5) / 2 for z in x]
# y = 2 * x + 1

plt.scatter(x, y)

b0 = random.rand()
b1 = random.rand()

lsrl_x = np.linspace(-100000, 100000, 100)
lsrl_y = b0 + b1 * lsrl_x

plt.plot(lsrl_x, lsrl_y)
plt.xlim([-0.25, 1])
plt.ylim([-0.25, 1])

files = ['temp/og.png']
plt.savefig(files[0], dpi=img_dpi)
plt.close()

def mse():
    global x, y, b0, b1, n
    mean = 0
    for xi, yi in zip(x, y):
        mean += pow(yi - (b0 + b1 * xi), 2)
    mean /= n
    return mean

print(mse())

num_iter = 100
learning_rate = 0.5
for i in range(num_iter):
    db0 = db1 = 0
    for xi, yi in zip(x, y):
        db0 += yi - (b0 + b1 * xi)
        db1 += (yi - (b0 + b1 * xi)) * xi
    db0 *= -2 / n
    db1 *= -2 / n
    b0 = b0 - learning_rate * db0
    b1 = b1 - learning_rate * db1

    plt.scatter(x, y)
    lsrl_x = np.linspace(-1, 2, 100)
    lsrl_y = b0 + b1 * lsrl_x
    plt.plot(lsrl_x, lsrl_y)
    plt.xlim([-0.25, 1])
    plt.ylim([-0.25, 1])
    files.append('temp/' + str(i) + '.png')
    plt.savefig('temp/' + str(i) + '.png', dpi=img_dpi)
    plt.close()
    print(mse())

with imageio.get_writer('linreg4.gif', mode='I') as writer:
    for file in files:
        writer.append_data(imageio.imread(file))
for file in files:
    os.remove(file)