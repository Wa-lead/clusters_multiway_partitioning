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
import sys

sys.setrecursionlimit(10000)



fig, ax = plt.subplots(2, figsize=(8, 14))
# fig, ax = plt.subplots(2, figsize=(4,7))
connectingDistance = 3
numberOfClusters = 4

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
    array_database.append(Point(groups[i], i, x[i], y[i], 1 if random.randint(0, 9) > 8 else 0, 0))



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


GroupCopy = deepcopy(arrayOfSubsets[0].outerPoints)