import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import random
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from Algorithem import *
from collections import Counter
from copy import copy, deepcopy



fig, ax = plt.subplots(2, figsize=(8, 14))
# fig, ax = plt.subplots(2, figsize=(4,7))
connectingDistance = 1
numberOfClusters = 4

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Preparing the poisson distribution pibts
# Simulation window parameters
xMin = 0
xMax = 30
yMin = 0
yMax = 30
xDelta = xMax-xMin
yDelta = yMax-yMin  # rectangle dimensions
areaTotal = xDelta*yDelta

lambda0 = 0.2
numbPoints = (scipy.stats.poisson(lambda0*areaTotal).rvs())*numberOfClusters
x = np.random.uniform(size=numbPoints, low=xMin, high=xMax)
y = np.random.uniform(size=numbPoints, low=yMin,
                      high=yMax)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Dividing the set into several even-size clusters
groups = divideIntoEvenClusters(x, y, numberOfClusters)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Preparing The data point data base

array_database = []

for i in range(numbPoints):
    array_database.append({"group": groups[i], "index": i, "x": x[i], "y": y[i],
                          "infected": 1 if random.randint(0, 9) > 8 else 0, "Dvalue": 0})

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
        distance = sqrt((point1['x']-point2['x'])**2 +
                        (point1['y']-point2['y'])**2)
        edgeMatrix[index_first_point][index_second_point] = 1 if distance <= connectingDistance else 0
        edgeMatrix[index_second_point][index_first_point] = edgeMatrix[index_first_point][index_second_point]

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#         # Plot the first graph before partioning
# # for index_first_point in range(numbPoints):
# #     for index_second_point in range(numbPoints):
# #         point1 = array_database[index_first_point]
# #         point2 = array_database[index_second_point]
# #         if edgeMatrix[index_first_point][index_second_point] != 0:
# #             pointX = [point1['x'], point2['x']]
# #             pointY = [point1['y'], point2['y']]
# #             ax[0].plot(pointX, pointY, 'grey')

# ax[0].scatter(x, y, c=groups, cmap='rainbow')

# ax[0].plot([2, 2], [2, 12], color='black')
# ax[0].plot([2, 12], [12, 12], color='black')
# ax[0].plot([12, 12], [12, 2], color='black')
# ax[0].plot([12, 2], [2, 2], color='black')

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Multi-way Partioning
i = 0
while i < numberOfClusters:
    j = i + 1
    while j < numberOfClusters:
        arrayOfSubsets[i]['subset'], arrayOfSubsets[j]['subset'], indicator, groups = twoWayPartitioning(
            arrayOfSubsets[i], arrayOfSubsets[j], edgeMatrix, groups)
        # print(indicator)
        j = j + 1
        if indicator > 0:
            i = 0
            j = i + 1
    i = i + 1

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
arrayOfSubsets = getClustersPoints(numberOfClusters, array_database)
arrayOfEdgePoints = findOuterEdges(arrayOfSubsets, array_database, edgeMatrix) 
edgeMatrixCopy = deepcopy(edgeMatrix)     
firewalls1 = findFirewalls(deepcopy(arrayOfEdgePoints), deepcopy(array_database), edgeMatrixCopy, arrayOfSubsets)

ax[0].scatter(x, y, c=groups, cmap='rainbow')

conver = []
for point in firewalls1:
    conver.append([point['x'],point['y']])

X = np.array(conver)
if len(X) != 0:
    ax[0].scatter(X[:,0],X[:,1], c='green')

ax[0].plot([(xMax+xMin)/4, (yMax+yMin)/4], [(xMax+xMin)/4, (yMax+yMin)/1.25], color='black')
ax[0].plot([(xMax+xMin)/4, (yMax+yMin)/1.25], [(xMax+xMin)/1.25, (yMax+yMin)/1.25], color='black')
ax[0].plot([(xMax+xMin)/1.25, (yMax+yMin)/1.25], [(xMax+xMin)/1.25, (yMax+yMin)/4], color='black')
ax[0].plot([(xMax+xMin)/1.25, (yMax+yMin)/4], [(xMax+xMin)/4, (yMax+yMin)/4], color='black')

        # Plot the first graph before partioning
for index_first_point in range(numbPoints):
    for index_second_point in range(numbPoints):
        point1 = array_database[index_first_point]
        point2 = array_database[index_second_point]
        edge = edgeMatrixCopy[index_first_point][index_second_point]
        if edge ==2:
            pointX = [point1['x'], point2['x']]
            pointY = [point1['y'], point2['y']]
            ax[0].plot(pointX, pointY, 'white' if edge ==1 else 'green')



onePointInterchange(groups, arrayOfEdgePoints, arrayOfSubsets, edgeMatrix)
arrayOfSubsets = getClustersPoints(numberOfClusters, array_database)
arrayOfEdgePoints = findOuterEdges(arrayOfSubsets, array_database, edgeMatrix)
firewalls = findFirewalls(arrayOfEdgePoints, array_database, edgeMatrix, arrayOfSubsets)

print(len(firewalls))
print(len(firewalls1))
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Plotting the points on graph
        # iterates through the the array and connects every node within distance 1 kilometer

for index_first_point in range(numbPoints):
    for index_second_point in range(index_first_point, numbPoints):
        point1 = array_database[index_first_point]
        point2 = array_database[index_second_point]
        edge = edgeMatrix[index_first_point][index_second_point]
        if edge ==2 :
            pointX = [point1['x'], point2['x']]
            pointY = [point1['y'], point2['y']]
            ax[1].plot(pointX, pointY, 'white' if edge == 1 else 'green')

# plots the points on the graph, if the point is healthy make it green, else red
# array_database = sorted(array_database, key=lambda x: x['index'])
ax[1].scatter(x, y, c=groups, cmap='rainbow')

conver = []
for point in firewalls:
    conver.append([point['x'],point['y']])

X = np.array(conver)
if len(X) != 0:
    ax[1].scatter(X[:,0],X[:,1], c='green')


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Drawing the boundries of the range
ax[1].plot([(xMax+xMin)/4, (yMax+yMin)/4], [(xMax+xMin)/4, (yMax+yMin)/1.25], color='black')
ax[1].plot([(xMax+xMin)/4, (yMax+yMin)/1.25], [(xMax+xMin)/1.25, (yMax+yMin)/1.25], color='black')
ax[1].plot([(xMax+xMin)/1.25, (yMax+yMin)/1.25], [(xMax+xMin)/1.25, (yMax+yMin)/4], color='black')
ax[1].plot([(xMax+xMin)/1.25, (yMax+yMin)/4], [(xMax+xMin)/4, (yMax+yMin)/4], color='black')

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

plt.show()
