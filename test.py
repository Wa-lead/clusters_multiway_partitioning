import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import random
#creates multiple random points in "np" format, then I convert them to normal list form.
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from Algorithem import twoWayPartitioning
from sklearn.cluster import KMeans



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
x = xDelta*scipy.stats.uniform.rvs(0,1,((numbPoints,1)))+xMin#x coordinates of Poisson points
y = yDelta*scipy.stats.uniform.rvs(0,1,((numbPoints,1)))+yMin#y coordinates of Poisson 

array =[]

for i in range(len(x)):
    array.append([x[i][0],y[i][0]])

X = np.array(array)

kmeans = KMeans(n_clusters=5, algorithm='full')
kmeans.fit(X)
print(kmeans.labels_)

zero = 0
one = 0

for label in kmeans.labels_:
    if label == 0:
        zero += 1
    elif label == 1:
        one += 1


print(one)
print(zero)


plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow')
plt.show()
