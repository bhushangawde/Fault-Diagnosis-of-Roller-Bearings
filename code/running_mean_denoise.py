

import numpy as np
import csv
import matplotlib.pyplot as plt

my_list=np.genfromtxt('original data at 1300rpm.csv',delimiter=",")

def moving_average(a, n=20) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

nl=np.empty([1000,2])
for i in range(0,999):
    nl[i,0]=my_list[i,1]

#plt.plot(nl[:,0])

my_list2=moving_average(nl[:,0])
plt.plot(nl[:,0])
plt.plot(my_list2)

