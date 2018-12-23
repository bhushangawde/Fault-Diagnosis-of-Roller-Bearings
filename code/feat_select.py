import csv 
import numpy as np
from numpy import *

my_list=np.genfromtxt('denoised_data_1300.csv',delimiter=",")

def sqrt(val):
    return val**0.5

def my_mean(lst):
    if sum(lst)>0:
        return sum(lst)/len(lst)
    else:
        return 0

def rms(lst):
    return sqrt(my_mean(lst))

def var(lst):
    res=0
    mean_val=my_mean(lst)
    for i in range(0,len(lst)-1):
        res=res+((lst[i]-mean_val)**2)
    if res>0:
        return res/len(lst)
    else:
        return 0

def max_peak(array):
    min_val=max_val=array[0]
    for i in range(0,len(array)-1):
        if array[i]>=max_val:
            max_val=array[i]
        if array[i]<min_val:
            min_val=array[i]
    if abs(min_val)>=abs(max_val):
        return min_val
    else:
        return max_val

def stddev(lst):
    variance = 0
    mn = my_mean(lst)
    for e in lst:
        variance += (e-mn)**2
    variance /= len(lst)
    return sqrt(variance)

def peak(array):
    min_val=max_val=array[0]
    for i in range(0,len(array)-1):
        if array[i]>max_val:
            max_val=array[i]
        if array[i]<min_val:
            min_val=array[i]
    return max_val-min_val


def skew(lst):
    mean_val=my_mean(lst)
    std_val=stddev(lst)
    len_val=len(lst)
    sum_val=0
    for i in range(0,len_val-1):
        sum_val=sum_val+((lst[i]-mean_val)**3)
    return sum_val/((len_val-1)*(std_val**3))

def kurt(obs):
    num = np.sum((obs - my_mean(obs))**4)/len(obs)
    denom = var(obs)**2
    return num / denom

def crest(array):
    peak=max_peak(array)
    if peak>0 and rms(array)>0:
        return peak/rms(array)
    else:
        return 0

def impulse(array):
    peak=max_peak(array)
    addition=sum(abs(i) for i in array)
    return peak/(addition/(len(array)))

def clearance(array):
    peak=max_peak(array)
    addition=0
    addition=sum(sqrt(abs(i)) for i in array)
    return peak/((addition/len(array))**2)

def shape_factor(array):
    rms_val=rms(array)
    addition=sum(abs(i) for i in array)
    return rms_val/(addition/len(array))


nl = np.fft.fft(my_list[:,0])
nl_2 = np.fft.fft(my_list[:,1])
print("The fourier transform of this data is\n\n\n")
#print(nl)

k=0
cdata=[]
for i in range(0,900):
    data=[]
    s = 1024*i
    f = ((i+1)*1024)
    hey=my_list[s:f,0]
    data.append(rms(my_list[s:f,0]))
    data.append(kurt(my_list[s:f,0]))
    data.append(impulse(my_list[s:f,0]))
    data.append(crest(my_list[s:f,0]))
    data.append(peak(my_list[s:f,0]))
    data.append(skew(my_list[s:f,0]))
    data.append(clearance(my_list[s:f,0]))
    data.append(shape_factor(my_list[s:f,0]))
    data.append(stddev(my_list[s:f,0]))
    data.append(rms(my_list[s:f,1]))
    data.append(kurt(my_list[s:f,1]))
    data.append(impulse(my_list[s:f,1]))
    data.append(crest(my_list[s:f,1]))
    data.append(peak(my_list[s:f,1]))
    data.append(skew(my_list[s:f,1]))
    data.append(clearance(my_list[s:f,1]))
    data.append(shape_factor(my_list[s:f,1]))
    data.append(stddev(my_list[s:f,1]))
    cdata.append(data)
    

with open("x2.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(cdata)
    
newlist =np.genfromtxt('x2.csv',delimiter=",")

def pcaAlg(X):
    X=array(X)
    avg = mean(X,axis=0)
    avg = tile(avg,(X.shape[0],1))
    X -= avg;
    
    C = dot(X.transpose(),X)/(X.shape[0]-1)
    eig_values,eig_vecs = linalg.eig(C)
    idx = eig_values.argsort()
    idx = idx[ : :-1]
    eig_values = eig_values[idx]
    eig_vecs = eig_vecs[:,idx]
    
    Y = dot(X,eig_vecs)
    return (Y, eig_vecs, eig_values)

Y,eig_vecs,eig_values = pcaAlg(newlist)


print(eig_values)
final_eig=[]
for x in eig_values:
    if(x>1):
       final_eig.append(x) 
       
print("Final eigen values are: \n")
print(final_eig)