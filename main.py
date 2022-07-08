import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import scipy.stats
import matplotlib.pyplot as plt
from research_module import *
from copy import copy, deepcopy
from point import Point
from plot import *


fig, ax = plt.subplots(ncols = 2, figsize=(10, 5))
# fig, ax = plt.subplots(2, figsize=(4,7))
connect_distance = 1
num_of_clusters = 3

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Preparing the poisson distribution pibts
# Simulation window parameters
xMin = 0
xMax = 20
yMin = 0
yMax = 20
xDelta = xMax-xMin
yDelta = yMax-yMin  # rectangle dimensions
areaTotal = xDelta*yDelta

lambda0 = 1
# numbPoints = (scipy.stats.poisson(lambda0*areaTotal).rvs())
# np.save('point.npy',numbPoints)
numbPoints = np.load('point.npy', allow_pickle=True)

# x = np.random.uniform(size=numbPoints, low=xMin, high=xMax)
# np.save('x.npy',x)
x = np.load('x.npy', allow_pickle=True)

# y = np.random.uniform(size=numbPoints, low=yMin, high=yMax)
# np.save('y.npy',y)
y = np.load('y.npy', allow_pickle=True)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Dividing the set into several even-size clusters
# groups_ = divide_even_clusters(x, y, num_of_clusters)
# np.save('test.npy',groups_)
groups_ = np.load('test.npy', allow_pickle=True)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Preparing The data point data base

array_of_points = []


for i in range(numbPoints):
    """
    The last index of the point object indecates the state of the point:
    0: Firewall
    1: Normal 
    2: Protected --> protects itself and it's edges (not a firewall) <---- this is used in the second approach of firewalls
    """
    array_of_points.append(Point(groups_[i], i, x[i], y[i], 1, 0))

array_of_groups = get_cluster_points(num_of_clusters, array_of_points)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Prepare the edge_matrix
edge_matrix = form_edge_matrix(array_of_points, connect_distance)

# Here we're finding the outer connected grouops for each and the also the outer points
for group in array_of_groups:
    group.find_connected_groups()

plot_points(ax[0], x, y, groups_, array_of_groups)
plot_firewalls(ax[0], array_of_groups, protect='self')
plot_borders(ax[0], xMax, xMin, yMax, yMin)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Multi-way Partioning
i = 0
while i < num_of_clusters:
    startAgain = False
    for group in array_of_groups[i].connectedGroup:
        indicator = two_way_partitioning_enhanced(
            array_of_groups[i], array_of_groups[group], groups_)
        if indicator > 0:
            startAgain = True
            array_of_groups[i].find_connected_groups()
            array_of_groups[group].find_connected_groups()
    i = 0 if startAgain else i + 1

edgeMatrixCopy = deepcopy(edge_matrix)


# # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ---------- enforcing the new changes before the one way partioning
for group in array_of_groups:
    group.find_connected_groups()
# one point interchange
i = 0
while i < num_of_clusters:
    startAgain = False
    for group in array_of_groups[i].connectedGroup:
        indicator = one_way_partioning_enhanced(
            array_of_groups[i], array_of_groups[group], groups_, numbPoints/num_of_clusters, 0.1)
        if indicator > 0:
            startAgain = True
        array_of_groups[i].find_connected_groups()
        array_of_groups[group].find_connected_groups()
    i = 0 if startAgain else i + 1

plot_points(ax[1], x, y, groups_, array_of_groups)
plot_firewalls(ax[1], array_of_groups, protect='self')
plot_borders(ax[1], xMax, xMin, yMax, yMin)

# # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

for group in array_of_groups:
    if group.name == 1:
        print(group.outerPoints)

for group in array_of_groups:
    x_ = group.points[floor(len(group.points)/2)].x
    y_ = group.points[floor(len(group.points)/2)].y
    print(group.name, len(group.points), (int(numbPoints /
          num_of_clusters) - len(group.points)), (group.connectedGroup))
    ax[1].annotate(f'{group.name}', xy=(x_, y_), color='red',
                   xytext=(10, 10), textcoords="offset points")

print(numbPoints, int(int(numbPoints/num_of_clusters)*1.1),
      int((numbPoints/num_of_clusters)*0.9))

plt.show()
