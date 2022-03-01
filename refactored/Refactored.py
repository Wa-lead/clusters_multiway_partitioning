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


fig, ax = plt.subplots(2, figsize=(8, 14))
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

lambda0 = 1
numbPoints = (scipy.stats.poisson(lambda0*areaTotal).rvs())*numberOfClusters
x = np.random.uniform(size=numbPoints, low=xMin, high=xMax)
y = np.random.uniform(size=numbPoints, low=yMin,
                      high=yMax)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Dividing the set into several even-size clusters
groups = divideIntoEvenClusters(x, y, numberOfClusters)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Preparing The data point data base

arrayOfSubsets = []

for i in range(numberOfClusters):
    arrayOfSubsets.append(Group(i))

array_database = []

for i in range(numbPoints): #each point created adds it self to the group in belongs to
    array_database.append(Point(arrayOfSubsets[groups[i]], i, x[i], y[i], 1 if random.randint(0, 9) > 8 else 0, 0)
                          )




# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#         Prepare the edgeMatrix
#         Why? because we want to assign cost to each edge

edgeMatrix = [[0 for i in range(len(array_database))]
              for j in range(len(array_database))]

for index_first_point in range(numbPoints):  # x
    for index_second_point in range(index_first_point+1, numbPoints):
        point1 = array_database[index_first_point]
        point2 = array_database[index_second_point]
        if(point2 == point1):
            continue
        distance = sqrt((point1.x-point2.x)**2 +
                        (point1.y-point2.y)**2)
        if distance <= connectingDistance:
            point1.connect(point2)
            edgeMatrix[index_first_point][index_second_point] = 1
            edgeMatrix[index_second_point][index_first_point] = edgeMatrix[index_first_point][index_second_point]

# here we're finding the outer connected grouops for each and the also the outer points

for group in arrayOfSubsets:
    group.findConnectedGroup()


# # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
    startAgain = False
    for group in map(int,arrayOfSubsets[i].connectedGroup.keys()):
        indicator, points= twoWayPartitioningEdgePoint(
            arrayOfSubsets[i], arrayOfSubsets[group], edgeMatrix, groups)
        if indicator > 0:
            for point in points:
                point.updateOuterGroups()
            startAgain = True
    i = 0 if startAgain else i + 1    
            





# # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# edgeMatrixCopy = deepcopy(edgeMatrix)
# firewall = findFirewalls(arrayOfSubsets,edg)
# while firewall is not None:
#     for edge in edgeMatrixCopy[firewall.index]:
#         if edge == 1:
#             edge = 2
#     firewall.infected = 2
#     firewall = findFirewalls(arrayOfSubsets)

# ax[0].scatter(x, y, c=groups, cmap='rainbow')

# conver = []
# for point in firewalls1:
#     conver.append([point['x'],point['y']])

# X = np.array(conver)
# if len(X) != 0:
#     ax[0].scatter(X[:,0],X[:,1], c='green')

# ax[0].plot([(xMax+xMin)/4, (yMax+yMin)/4], [(xMax+xMin)/4, (yMax+yMin)/1.25], color='black')
# ax[0].plot([(xMax+xMin)/4, (yMax+yMin)/1.25], [(xMax+xMin)/1.25, (yMax+yMin)/1.25], color='black')
# ax[0].plot([(xMax+xMin)/1.25, (yMax+yMin)/1.25], [(xMax+xMin)/1.25, (yMax+yMin)/4], color='black')
# ax[0].plot([(xMax+xMin)/1.25, (yMax+yMin)/4], [(xMax+xMin)/4, (yMax+yMin)/4], color='black')

        # Plot the first graph before partioning
# for index_first_point in range(numbPoints):
#     for index_second_point in range(index_first_point,numbPoints):
#         point1 = array_database[index_first_point]
#         point2 = array_database[index_second_point]
#         edge = edgeMatrixCopy[index_first_point][index_second_point]
#         if edge > 0:
#             pointX = [point1.x, point2.x]
#             pointY = [point1.y, point2.y]
#             ax[0].plot(pointX, pointY, 'white' if edge ==1 else 'green')


# onePointInterchange(groups, arrayOfEdgePoints, arrayOfSubsets, edgeMatrix)
# arrayOfSubsets = getClustersPoints(numberOfClusters, array_database)
# arrayOfEdgePoints = findOuterEdges(arrayOfSubsets, array_database, edgeMatrix)
# firewalls = findFirewalls(arrayOfEdgePoints, array_database, edgeMatrix, arrayOfSubsets)

# print(len(firewalls))
# print(len(firewalls1))
# # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#         # Plotting the points on graph
#         # iterates through the the array and connects every node within distance 1 kilometer

# for index_first_point in range(numbPoints):
#     for index_second_point in range(index_first_point, numbPoints):
#         point1 = array_database[index_first_point]
#         point2 = array_database[index_second_point]
#         edge = edgeMatrix[index_first_point][index_second_point]
#         if edge ==2 :
#             pointX = [point1['x'], point2['x']]
#             pointY = [point1['y'], point2['y']]
#             ax[1].plot(pointX, pointY, 'white' if edge == 1 else 'green')

# # plots the points on the graph, if the point is healthy make it green, else red
# # array_database = sorted(array_database, key=lambda x: x['index'])
ax[1].scatter(x, y, c=groups, cmap='rainbow')

# conver = []
# for point in firewalls:
#     conver.append([point['x'],point['y']])

# X = np.array(conver)
# if len(X) != 0:
#     ax[1].scatter(X[:,0],X[:,1], c='green')


# # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Drawing the boundries of the range
ax[1].plot([(xMax+xMin)/4, (yMax+yMin)/4],
           [(xMax+xMin)/4, (yMax+yMin)/1.25], color='black')
ax[1].plot([(xMax+xMin)/4, (yMax+yMin)/1.25],
           [(xMax+xMin)/1.25, (yMax+yMin)/1.25], color='black')
ax[1].plot([(xMax+xMin)/1.25, (yMax+yMin)/1.25],
           [(xMax+xMin)/1.25, (yMax+yMin)/4], color='black')
ax[1].plot([(xMax+xMin)/1.25, (yMax+yMin)/4],
           [(xMax+xMin)/4, (yMax+yMin)/4], color='black')

# # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

plt.show()
