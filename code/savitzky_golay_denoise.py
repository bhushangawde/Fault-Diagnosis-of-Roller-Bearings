import csv
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame as df
import pandas as pd
import scipy
from math import factorial

my_list=np.genfromtxt('original data at 1300rpm.csv',delimiter=",")

nl=np.empty([500,2])
for i in range(0,499):
    nl[i,0]=my_list[i,1]
    
#window_size size must be a positive odd number

def denoise(y, window_size, order, deriv=0, rate=1):
	window_size = np.abs(np.int(window_size))
	order = np.abs(np.int(order))
	order_range = range(order+1)
	half_window = (window_size -1) // 2
	b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
	m = np.linalg.pinv(b).A[deriv]
	firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
	lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
	y = np.concatenate((firstvals, y, lastvals))
	return np.convolve( m[::-1], y, mode='valid')

ysg = denoise(nl[:,0], window_size=31, order=2)
plt.plot(nl[:,0])
plt.plot(ysg)	
plt.show()
