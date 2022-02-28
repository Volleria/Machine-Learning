from random import random

import matplotlib.pyplot as plt
import wrapt
import math

w_banlance = [23.1, 20.4, 15.7, 16.7, 1.8, 12.0,
              6.3, 7.5, 5.2, 15.7, 13.0,
              15.1, 12.3, 13.7, 10.6, 17.1,
              14.3, 15.0, 12.3, 14.1, 10.9]
time_min = [0, 3.1, 4.2, 5.1, 14.7, 19.2,
            22.1, 22.8, 23.1, 33.2, 33.5,
            34.2, 36.4, 43.5, 44.2, 49.1,
            51.2, 51.8, 52.5, 58.3, 60]

W = []
T = []
time_temp = [i * 10 for i in time_min]
for i in range(len(w_banlance)):
    W.append(w_banlance[i])
    if i < len(w_banlance) - 1:
        # n = int(time_temp[i + 1] - time_temp[i])
        n = int(time_min[i+1] - time_min[i])
        for j in range(n):
            m = (w_banlance[i + 1] - w_banlance[i]) / n
            W.append(w_banlance[i]+m*j + 2*math.sin(random()))


time_min = [i / 10 for i in range(len(W))]
#print(time_min)

#plt.scatter(time_min,W, c='k', zorder=0.5)
plt.plot(time_min, W, c='r', label="W_Banlance")
plt.xlabel('time_min')
plt.ylabel('w_banlance')
plt.legend()
plt.show()
