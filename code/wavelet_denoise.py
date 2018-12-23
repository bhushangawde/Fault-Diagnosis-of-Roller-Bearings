import csv
import numpy as np
from math import factorial
import matplotlib.pyplot as plt

import pywt
from statsmodels.robust import mad

def waveletDen( x, wavelet, level, title=None ):
    
    coeff = pywt.wavedec( x, wavelet, mode="per")
    sigma = mad( coeff[-level])
    uthresh = sigma * np.sqrt(2*np.log( len( x ) ) )
    coeff[1:] = ( pywt.threshold( i, value=uthresh, mode="soft" ) for i in coeff[1:] )
    
    y = pywt.waverec( coeff, wavelet, mode="per" )
    f, ax = plt.subplots()
    plt.plot( x, color="b", alpha=0.5 )
    plt.plot( y, color="r" )
    plt.show()
    if title:
        ax.set_title(title)
    ax.set_xlim((0,len(y)))


my_list=np.genfromtxt('original data at 1300rpm.csv',delimiter=",")
#my_list=pd.read_csv('original data at 1300rpm.csv',sep=",",header=None,names=['c1','c2'])
#plt.scatter(my_list[:,0],my_list[:,1])
#for i in range(0,999):
    #plt.scatter(my_list[i,0])    
#my_list.sort_values('c2',inplace=True)
nl=np.empty([1000,2])
for i in range(0,999):
    nl[i,0]=my_list[i,1]

waveletDen(nl[:,0],wavelet="db1",level=2)
#plt.plot(nli)
#np.sort(nl[:,0],kind='mergesort')
#print(nl[:,0])
#plt.plot(nl[:,0])
#plt.axis([-9,8,-14,12])
#showing the graph
#plt.show()
#print(nl[:,0])