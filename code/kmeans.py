import csv 
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial import distance
import random
from random import randint

den_list=np.genfromtxt('denoised_data_1300.csv',delimiter=",")

MAX_ITERATIONS = 10

def kmeans(data, k, c):
    centroids = []

    centroids = randomize_centroids(data, centroids, k)  

    old_centroids = [[] for i in range(k)] 

    iterations = 0
    while not (has_converged(centroids, old_centroids, iterations)):
        iterations += 1

        clusters = [[] for i in range(k)]

        # assign data points to clusters
        clusters = euclidean_dist(data, centroids, clusters)

        # recalculate centroids
        index = 0
        for cluster in clusters:
            old_centroids[index] = centroids[index]
            centroids[index] = np.mean(cluster, axis=0).tolist()
            index += 1


    print("The total number of data instances is: " + str(len(data)))
    print("The total number of iterations necessary is: " + str(iterations))
    print("The means of each cluster are: " + str(centroids))
    print("The clusters are as follows:")
    color=['r','b','g']
    n=randint(0,2)
    for cluster in clusters:
        print("Cluster with a size of " + str(len(cluster)) + " starts here:")
        print("\nCluster shape:",np.array(cluster).shape)
        print("Cluster ends here.")
        n=n+1
        plt.scatter(np.array(cluster)[:,0],np.array(cluster)[:,1],color=color[(n+1)%3])
    print("\nCentroids shape:",np.array(centroids).shape)
    plt.scatter(np.array(centroids)[:,0],np.array(centroids)[:,1],color='k')
    plt.show()
    return

def euclidean_dist(data, centroids, clusters):
    for instance in data:  
        mu_index = min([(i[0], np.linalg.norm(instance-centroids[i[0]])) \
                            for i in enumerate(centroids)], key=lambda t:t[1])[0]
        try:
            clusters[mu_index].append(instance)
        except KeyError:
            clusters[mu_index] = [instance]

    for cluster in clusters:
        if not cluster:
            cluster.append(data[np.random.randint(0, len(data), size=1)].flatten().tolist())

    return clusters


def randomize_centroids(data, centroids, k):
    for cluster in range(0, k):
        centroids.append(data[np.random.randint(0, len(data), size=1)].flatten().tolist())
    return centroids


def has_converged(centroids, old_centroids, iterations):
    
    if iterations > MAX_ITERATIONS:
        return True
    return old_centroids == centroids

num=randint(0,1000999)
kmeans(den_list,3,den_list[num:num+3])
