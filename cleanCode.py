import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import random
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from Algorithem import twoWayPartitioning, divideIntoEvenClusters, getClustersPoints, findOuterEdges
from collections import Counter


fig, ax = plt.subplots(2, figsize=(8, 14))
connectingDistance = 1
numberOfClusters = 2

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Preparing the poisson distribution pibts
# Simulation window parameters
xMin = 0
xMax = 14
yMin = 0
yMax = 14
xDelta = xMax-xMin
yDelta = yMax-yMin  # rectangle dimensions
areaTotal = xDelta*yDelta

lambda0 = 0.5
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
        # Plot the first graph before partioning
# for index_first_point in range(numbPoints):
#     for index_second_point in range(numbPoints):
#         point1 = array_database[index_first_point]
#         point2 = array_database[index_second_point]
#         if edgeMatrix[index_first_point][index_second_point] != 0:
#             pointX = [point1['x'], point2['x']]
#             pointY = [point1['y'], point2['y']]
#             ax[0].plot(pointX, pointY, 'grey')

ax[0].scatter(x, y, c=groups, cmap='rainbow')

ax[0].plot([2, 2], [2, 12], color='black')
ax[0].plot([2, 12], [12, 12], color='black')
ax[0].plot([12, 12], [12, 2], color='black')
ax[0].plot([12, 2], [2, 2], color='black')

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

for element in arrayOfEdgePoints:
    for i in range(numbPoints):
        if edgeMatrix[element['point']['index']][i] == 1:
            edgeMatrix[element['point']['index']][i] = 2
            edgeMatrix[i][element['point']['index']] = 2
    arrayOfEdgePoints.pop(0)['point']['infected'] = 2
    arrayOfEdgePoints = findOuterEdges(
        arrayOfSubsets, array_database, edgeMatrix)

firewalls = []
for point in array_database:
    if point['infected'] == 2:
        firewalls.append(point)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Plotting the points on graph
        # iterates through the the array and connects every node within distance 1 kilometer
for index_first_point in range(numbPoints):
    for index_second_point in range(index_first_point, numbPoints):
        point1 = array_database[index_first_point]
        point2 = array_database[index_second_point]
        # the initial assumption is that only nodes within 1 kilometers of each other can connect
        if edgeMatrix[index_first_point][index_second_point] != 0:
            pointX = [point1['x'], point2['x']]
            pointY = [point1['y'], point2['y']]
            # isInA = point1 in arrayOfSubsets[0]
            # isInB = point1 in arrayOfSubsets[1]
            ax[1].plot(pointX, pointY, 'grey')

# plots the points on the graph, if the point is healthy make it green, else red
array_database = sorted(array_database, key=lambda x: x['index'])
ax[1].scatter(x, y, c=groups, cmap='rainbow')

conver = []
for point in firewalls:
    conver.append([point['x'],point['y']])

X = np.array(conver)
ax[1].scatter(X[:,0],X[:,1], c='green')


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Drawing the boundries of the range
ax[1].plot([2, 2], [2, 12], color='black')
ax[1].plot([2, 12], [12, 12], color='black')
ax[1].plot([12, 12], [12, 2], color='black')
ax[1].plot([12, 2], [2, 2], color='black')

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

plt.show()