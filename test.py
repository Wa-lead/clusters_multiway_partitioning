from dis import dis
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import random
#creates multiple random points in "np" format, then I convert them to normal list form.
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from Algorithem import twoWayPartitioning, divideIntoEvenClusters
from sklearn.cluster import KMeans
from k_means_constrained import KMeansConstrained



fig, ax = plt.subplots(2, figsize=(8, 14))

array_database = []

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                            #Preparing the poisson distribution pibts
#Simulation window parameters
xMin=0;xMax=14
yMin=0;yMax=14
xDelta=xMax-xMin
yDelta=yMax-yMin; #rectangle dimensions
areaTotal=xDelta*yDelta
 
#Point process parameters
lambda0=1; #intensity (ie mean density) of the Poisson process
 
#Simulate Poisson point process
numbPoints = (scipy.stats.poisson( lambda0*areaTotal ).rvs())*5#Poisson number of points # multiply by 2 to make it always even
x = np.random.uniform(size= numbPoints, low = xMin, high =xMax)# coordinates of Poisson points
y = np.random.uniform(size= numbPoints, low = yMin, high =yMax)#y coordinates of Poisson 

print(x[0])

# array =[]

# for i in range(len(x)):
#     array.append([x[i][0],y[i][0]])

# X = np.array(array)

# kmeans = KMeansConstrained(n_clusters=5, size_min=numbPoints/5, size_max=numbPoints/5)
# kmeans.fit(X)
# print(kmeans.labels_)

# zero = 0
# one = 0

# for label in kmeans.labels_:
#     if label == 0:
#         zero += 1
#     elif label == 1:
#         one += 1


# print(one)
# print(zero)


# plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow')
# plt.show()
