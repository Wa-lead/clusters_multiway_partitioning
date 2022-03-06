from tracemalloc import start
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import random
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from AlgorithemRefactored import *
from collections import Counter
from copy import copy, deepcopy
from point import Point


fig, ax = plt.subplots(3, figsize=(3, 7))
# fig, ax = plt.subplots(2, figsize=(4,7))
connectingDistance = 1
numberOfClusters = 2

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Preparing the poisson distribution pibts
# Simulation window parameters
xMin = 0
xMax = 15
yMin = 0
yMax = 15
xDelta = xMax-xMin
yDelta = yMax-yMin  # rectangle dimensions
areaTotal = xDelta*yDelta

lambda0 =1
numbPoints = (scipy.stats.poisson(lambda0*areaTotal).rvs())
x = np.random.uniform(size=numbPoints, low=xMin, high=xMax)
y = np.random.uniform(size=numbPoints, low=yMin,
                      high=yMax)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Dividing the set into several even-size clusters
groups = divideIntoEvenClusters(x, y, numberOfClusters)
print(numbPoints)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Preparing The data point data base

array_database = []

for i in range(numbPoints):
    array_database.append(Point(groups[i], i, x[i], y[i], 1 if random.randint(0, 9) > 8 else 0, 0)
                          )

arrayOfSubsets = getClustersPoints(numberOfClusters, array_database)



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#         Prepare the edgeMatrix
#         Why? because we want to assign cost to each edge

edgeMatrix = [[0 for i in range(len(array_database))]
              for j in range(len(array_database))]

for index_first_point in range(numbPoints):  # x
    for index_second_point in range(index_first_point, numbPoints):
        point1 = array_database[index_first_point]
        point2 = array_database[index_second_point]
        distance = sqrt((point1.x-point2.x)**2 +
                        (point1.y-point2.y)**2)
        if distance <= connectingDistance and distance != 0:
            point1.connect(point2)
            edgeMatrix[index_first_point][index_second_point] = 1
            edgeMatrix[index_second_point][index_first_point] = edgeMatrix[index_first_point][index_second_point]

# here we're finding the outer connected grouops for each and the also the outer points
for group in arrayOfSubsets:
    group.findConnectedGroup()

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Plot the first graph before partioning


edgeMatrixCopy = deepcopy(edgeMatrix)
firewalls = findFirewalls(arrayOfSubsets,edgeMatrixCopy)
print(len(firewalls))


ax[0].scatter(x, y, c=groups, cmap='rainbow')


for index_first_point in range(numbPoints):
    for index_second_point in range(index_first_point,numbPoints):
        point1 = array_database[index_first_point]
        point2 = array_database[index_second_point]
        if edgeMatrixCopy[index_first_point][index_second_point] ==2 or edgeMatrixCopy[index_second_point][index_first_point] == 2:
            pointX = [point1.x, point2.x]
            pointY = [point1.y, point2.y]
            ax[0].plot(pointX, pointY, 'green')

conver = []
for point in firewalls:
    conver.append([point.x,point.y])

X = np.array(conver)
if len(X) != 0:
    ax[0].scatter(X[:,0],X[:,1], c='green')

# ax[0].scatter(x, y, c=groups, cmap='rainbow')

# ax[0].plot([(xMax+xMin)/4, (yMax+yMin)/4],
#            [(xMax+xMin)/4, (yMax+yMin)/1.25], color='black')
# ax[0].plot([(xMax+xMin)/4, (yMax+yMin)/1.25],
#            [(xMax+xMin)/1.25, (yMax+yMin)/1.25], color='black')
# ax[0].plot([(xMax+xMin)/1.25, (yMax+yMin)/1.25],
#            [(xMax+xMin)/1.25, (yMax+yMin)/4], color='black')
# ax[0].plot([(xMax+xMin)/1.25, (yMax+yMin)/4],
#            [(xMax+xMin)/4, (yMax+yMin)/4], color='black')
# # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Multi-way Partioning
i = 0
while i < numberOfClusters:
    startAgain = False
    for group in arrayOfSubsets[i].connectedGroup:
        indicator = twoWayPartitioningEdgePoint(
            arrayOfSubsets[i], arrayOfSubsets[group], edgeMatrix, groups)
        if indicator > 0:
            arrayOfSubsets[i].findConnectedGroup()
            arrayOfSubsets[group].findConnectedGroup()
            startAgain = True
    i = 0 if startAgain else i + 1    
            


for group in arrayOfSubsets:
    group.findConnectedGroup()


# # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

ax[1].scatter(x, y, c=groups, cmap='rainbow')



ax[1].plot([(xMax+xMin)/4, (yMax+yMin)/4], [(xMax+xMin)/4, (yMax+yMin)/1.25], color='black')
ax[1].plot([(xMax+xMin)/4, (yMax+yMin)/1.25], [(xMax+xMin)/1.25, (yMax+yMin)/1.25], color='black')
ax[1].plot([(xMax+xMin)/1.25, (yMax+yMin)/1.25], [(xMax+xMin)/1.25, (yMax+yMin)/4], color='black')
ax[1].plot([(xMax+xMin)/1.25, (yMax+yMin)/4], [(xMax+xMin)/4, (yMax+yMin)/4], color='black')

edgeMatrixCopy = deepcopy(edgeMatrix)
firewalls = findFirewalls(arrayOfSubsets,edgeMatrixCopy)
print(len(firewalls))


for index_first_point in range(numbPoints):
    for index_second_point in range(index_first_point,numbPoints):
        point1 = array_database[index_first_point]
        point2 = array_database[index_second_point]
        if edgeMatrixCopy[index_first_point][index_second_point] ==2 or edgeMatrixCopy[index_second_point][index_first_point] == 2:
            pointX = [point1.x, point2.x]
            pointY = [point1.y, point2.y]
            ax[1].plot(pointX, pointY, 'green')




conver = []
for point in firewalls:
    conver.append([point.x,point.y])

X = np.array(conver)
if len(X) != 0:
    ax[1].scatter(X[:,0],X[:,1], c='green')







## one point interchange


i=0
while i < numberOfClusters:
    startAgain = False
    for group in arrayOfSubsets[i].connectedGroup:
        indicator = onePointInterchange(
            arrayOfSubsets[i], arrayOfSubsets[group], edgeMatrix, groups, numbPoints/numberOfClusters, 0.1)
        if indicator > 0:
            arrayOfSubsets[i].findConnectedGroup()
            arrayOfSubsets[group].findConnectedGroup()
            startAgain = True
    i = 0 if startAgain else i + 1    
            
ax[2].scatter(x, y, c=groups, cmap='rainbow')

firewalls = findFirewalls(arrayOfSubsets, edgeMatrix)
print(len(firewalls))


ax[2].plot([(xMax+xMin)/4, (yMax+yMin)/4], [(xMax+xMin)/4, (yMax+yMin)/1.25], color='black')
ax[2].plot([(xMax+xMin)/4, (yMax+yMin)/1.25], [(xMax+xMin)/1.25, (yMax+yMin)/1.25], color='black')
ax[2].plot([(xMax+xMin)/1.25, (yMax+yMin)/1.25], [(xMax+xMin)/1.25, (yMax+yMin)/4], color='black')
ax[2].plot([(xMax+xMin)/1.25, (yMax+yMin)/4], [(xMax+xMin)/4, (yMax+yMin)/4], color='black')



for index_first_point in range(numbPoints):
    for index_second_point in range(index_first_point,numbPoints):
        point1 = array_database[index_first_point]
        point2 = array_database[index_second_point]
        if edgeMatrix[index_first_point][index_second_point] ==2 or edgeMatrix[index_second_point][index_first_point] == 2:
            pointX = [point1.x, point2.x]
            pointY = [point1.y, point2.y]
            ax[2].plot(pointX, pointY, 'green')




conver = []
for point in firewalls:
    conver.append([point.x,point.y])

X = np.array(conver)
if len(X) != 0:
    ax[2].scatter(X[:,0],X[:,1], c='green')




# # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Drawing the boundries of the range
ax[2].plot([(xMax+xMin)/4, (yMax+yMin)/4],
           [(xMax+xMin)/4, (yMax+yMin)/1.25], color='black')
ax[2].plot([(xMax+xMin)/4, (yMax+yMin)/1.25],
           [(xMax+xMin)/1.25, (yMax+yMin)/1.25], color='black')
ax[2].plot([(xMax+xMin)/1.25, (yMax+yMin)/1.25],
           [(xMax+xMin)/1.25, (yMax+yMin)/4], color='black')
ax[2].plot([(xMax+xMin)/1.25, (yMax+yMin)/4],
           [(xMax+xMin)/4, (yMax+yMin)/4], color='black')

# # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

plt.show()
