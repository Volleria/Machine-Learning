from random import random

import matplotlib.pyplot as plt
import wrapt
import math

w_banlance = [24,17,17.3,18.1,14
              ,13.8,10.1,11.7,8.8,1.5
              ,9.5,10,12,13.5,17.2,14,10.5,13.2,0]
time_min = [0,1,2,2.5,3,4,5
            ,13,16,17,17.5,21
            ,30,34,40,42,45,46,50]

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
