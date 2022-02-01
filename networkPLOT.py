from dis import dis
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import random
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from Algorithem import twoWayPartitioning, divideIntoEvenClusters

fig, ax = plt.subplots(2, figsize=(8, 14))
connectingDistance = 2
numberOfClusters = 5

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

# Point process parameters
lambda0 = 0.5  # intensity (ie mean density) of the Poisson process
# Simulate Poisson point process
# Poisson number of points # multiply by 2 to make it always even
numbPoints = (scipy.stats.poisson(lambda0*areaTotal).rvs())*numberOfClusters
# x = xDelta*scipy.stats.uniform.rvs(0,1,((numbPoints,1)))+xMin#x coordinates of Poisson points
# y = yDelta*scipy.stats.uniform.rvs(0,1,((numbPoints,1)))+yMin#y coordinates of Poisson points
# coordinates of Poisson points
x = np.random.uniform(size=numbPoints, low=xMin, high=xMax)
y = np.random.uniform(size=numbPoints, low=yMin,
                      high=yMax)  # y coordinates of Poisson
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Dividing the set into several even-size clusters
groups = divideIntoEvenClusters(x, y, numberOfClusters)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Preparing The data point data base

arrayOfSubsets = []
array_database = []

# create dictionary - associative array  - that holds all points as objects, the infected attribute is random (1= infected, 0= healthy)
for i in range(numbPoints):
    array_database.append({"index": i, "x": x[i], "y": y[i],
                          "infected": 1 if random.randint(0, 9) > 8 else 0, "Dvalue": 0})

for i in range(numberOfClusters):
    arrayOfSubsets.append({'group': i , 'subset' :[]})
    for j in range(len(groups)):
        if(groups[j] == i):
            arrayOfSubsets[i]['subset'].append(array_database[j])

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# sorts the array based on the infected attribute, infected points come first.
array_database = sorted(
    array_database, key=lambda x: x['infected'], reverse=True)

# iterated through the array and make every infected node "infects" its neighbors
for index_first_point in range(numbPoints):  # x
    for index_second_point in range(numbPoints):
        point1 = array_database[index_first_point]
        point2 = array_database[index_second_point]
        distance = sqrt((point1['x']-point2['x'])**2 +
                        (point1['y']-point2['y'])**2)
        if distance <= connectingDistance:  # the initial assumption is that only nodes within 1 kilometers of each other can connect
            if point1['infected'] == 1 or point2['infected'] == 1:
                point1['infected'] = 1
                point2['infected'] = 1
        array_database = sorted(
            array_database, key=lambda x: x['infected'], reverse=True)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#         Prepare the edgeMatrix
#         Why? because we want to assign cost to each edge

# sort it based on index to uilize the edge matrix
array_database = sorted(array_database, key=lambda x: x['index'])


edgeMatrix = [[0 for i in range(len(array_database))]
              for j in range(len(array_database))]

for index_first_point in range(numbPoints):  # x
    for index_second_point in range(index_first_point, numbPoints):
        point1 = array_database[index_first_point]
        point2 = array_database[index_second_point]
        distance = sqrt((point1['x']-point2['x'])**2 +
                        (point1['y']-point2['y'])**2)
        edgeMatrix[index_first_point][index_second_point] = distance if distance <= connectingDistance else 0
        edgeMatrix[index_second_point][index_first_point] = edgeMatrix[index_first_point][index_second_point]

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Plot the first graph before partioning
# for index_first_point in range(numbPoints):
#     for index_second_point in range(numbPoints):
#         point1 = array_database[index_first_point]
#         point2 = array_database[index_second_point]
#         # the initial assumption is that only nodes within 1 kilometers of each other can connect
#         if edgeMatrix[index_first_point][index_second_point] != 0:
#             pointX = [point1['x'], point2['x']]
#             pointY = [point1['y'], point2['y']]
#             # isInA = point1 in arrayOfSubsets[0]
#             # isInB = point1 in arrayOfSubsets[1]
#             ax[0].plot(pointX, pointY)

ax[0].scatter(x,y, c=groups, cmap='rainbow')

ax[0].plot([2, 2], [2, 12], color='black')
ax[0].plot([2, 12], [12, 12], color='black')
ax[0].plot([12, 12], [12, 2], color='black')
ax[0].plot([12, 2], [2, 2], color='black')

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
i = 0
while i < numberOfClusters:
    j = i + 1
    while j < numberOfClusters:
        arrayOfSubsets[i]['subset'], arrayOfSubsets[j]['subset'], indicator, groups = twoWayPartitioning(
            arrayOfSubsets[i], arrayOfSubsets[j], edgeMatrix, groups)
        print(indicator)
        j = j + 1
        if indicator > 0:
            i = 0
            j = i + 1
    i = i + 1

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                   # Plotting the points on graph

# iterates through the the array and connects every node within distance 1 kilometer
# for index_first_point in range(numbPoints):
#     for index_second_point in range(index_first_point, numbPoints):
#         point1 = array_database[index_first_point]
#         point2 = array_database[index_second_point]
#         # the initial assumption is that only nodes within 1 kilometers of each other can connect
#         if edgeMatrix[index_first_point][index_second_point] != 0:
#             pointX = [point1['x'], point2['x']]
#             pointY = [point1['y'], point2['y']]
#             # isInA = point1 in arrayOfSubsets[0]
#             # isInB = point1 in arrayOfSubsets[1]
#             ax[1].plot(pointX, pointY,
#                        'black')

# plots the points on the graph, if the point is healthy make it green, else red
array_database = sorted(array_database, key=lambda x: x['index'])
ax[1].scatter(x,y, c=groups, cmap='rainbow')

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
       # Drawing the boundries of the range
ax[1].plot([2, 2], [2, 12], color='black')
ax[1].plot([2, 12], [12, 12], color='black')
ax[1].plot([12, 12], [12, 2], color='black')
ax[1].plot([12, 2], [2, 2], color='black')

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

plt.show()
