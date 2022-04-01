import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import scipy.stats
import matplotlib.pyplot as plt
from research_module import *
from copy import copy, deepcopy
from point import Point


fig, ax = plt.subplots(2, figsize=(7,12))
# fig, ax = plt.subplots(2, figsize=(4,7))
connect_distance = 1
num_of_clusters = 2

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

lambda0 = 0.7
numbPoints = (scipy.stats.poisson(lambda0*areaTotal).rvs())
x = np.random.uniform(size=numbPoints, low=xMin, high=xMax)
y = np.random.uniform(size=numbPoints, low=yMin,
                      high=yMax)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Dividing the set into several even-size clusters
groups_ = divide_even_clusters(x, y, num_of_clusters)
print(numbPoints)
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
    array_of_points.append(Point(groups_[i], i, x[i], y[i], 1, 0)  
                          )

array_of_groups = get_cluster_points(num_of_clusters, array_of_points)



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#         Prepare the edge_matrix
#         Why? because we want to assign cost to each edge

edge_matrix = [[0 for i in range(len(array_of_points))]
              for j in range(len(array_of_points))]

for index_first_point in range(numbPoints):  # x
    for index_second_point in range(index_first_point, numbPoints):
        point1 = array_of_points[index_first_point]
        point2 = array_of_points[index_second_point]
        distance = sqrt((point1.x-point2.x)**2 +
                        (point1.y-point2.y)**2)
        if distance <= connect_distance and distance != 0:
            point1.connect(point2)
            edge_matrix[index_first_point][index_second_point] = 1
            edge_matrix[index_second_point][index_first_point] = edge_matrix[index_first_point][index_second_point]

# here we're finding the outer connected grouops for each and the also the outer points
for group in array_of_groups:
    group.find_connected_groups()

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Multi-way Partioning
i = 0
while i < num_of_clusters:
    startAgain = False
    for group in array_of_groups[i].connectedGroup:
        indicator = two_way_partitioning(
            array_of_groups[i], array_of_groups[group], edge_matrix, groups_)
        if indicator > 0:
            array_of_groups[i].find_connected_groups()
            array_of_groups[group].find_connected_groups()
            startAgain = True
    i = 0 if startAgain else i + 1    
            


for group in array_of_groups:
    group.find_connected_groups()


# # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

ax[0].scatter(x, y, c=groups_, cmap='rainbow')



ax[0].plot([(xMax+xMin)/4, (yMax+yMin)/4], [(xMax+xMin)/4, (yMax+yMin)/1.25], color='black')
ax[0].plot([(xMax+xMin)/4, (yMax+yMin)/1.25], [(xMax+xMin)/1.25, (yMax+yMin)/1.25], color='black')
ax[0].plot([(xMax+xMin)/1.25, (yMax+yMin)/1.25], [(xMax+xMin)/1.25, (yMax+yMin)/4], color='black')
ax[0].plot([(xMax+xMin)/1.25, (yMax+yMin)/4], [(xMax+xMin)/4, (yMax+yMin)/4], color='black')

edgeMatrixCopy = deepcopy(edge_matrix)
firewalls, edges = find_firewalls(array_of_groups, protect='connected')
print(len(firewalls))


## to plot edges
for index_first_point in range(numbPoints):
    for index_second_point in range(index_first_point,numbPoints):
        point1 = array_of_points[index_first_point]
        point2 = array_of_points[index_second_point]
        if edgeMatrixCopy[index_first_point][index_second_point] ==1:
            pointX = [point1.x, point2.x]
            pointY = [point1.y, point2.y]
            ax[0].plot(pointX, pointY, 'black')

#plot the protected edges
for edge in edges:
    point_x = [edge[0][0],edge[1][0]]
    point_y = [edge[0][1],edge[1][1]]
    ax[0].plot(point_x, point_y, 'green')

conver = []
for point in firewalls:
    conver.append([point.x,point.y])

X = np.array(conver)
if len(X) != 0:
    ax[0].scatter(X[:,0],X[:,1], c='green')






## one point interchange
i=0
while i < num_of_clusters:
    startAgain = False
    for group in array_of_groups[i].connectedGroup:
        indicator = one_way_partioning(
            array_of_groups[i], array_of_groups[group], edge_matrix, groups_, numbPoints/num_of_clusters, 0.1)
        if indicator > 0:
            array_of_groups[i].find_connected_groups()
            array_of_groups[group].find_connected_groups()
            startAgain = True
    i = 0 if startAgain else i + 1    
            
ax[1].scatter(x, y, c=groups_, cmap='rainbow')

firewalls, edges = find_firewalls(array_of_groups, protect='connected')
print(len(firewalls))


ax[1].plot([(xMax+xMin)/4, (yMax+yMin)/4], [(xMax+xMin)/4, (yMax+yMin)/1.25], color='black')
ax[1].plot([(xMax+xMin)/4, (yMax+yMin)/1.25], [(xMax+xMin)/1.25, (yMax+yMin)/1.25], color='black')
ax[1].plot([(xMax+xMin)/1.25, (yMax+yMin)/1.25], [(xMax+xMin)/1.25, (yMax+yMin)/4], color='black')
ax[1].plot([(xMax+xMin)/1.25, (yMax+yMin)/4], [(xMax+xMin)/4, (yMax+yMin)/4], color='black')






## to plot edges
for index_first_point in range(numbPoints):
    for index_second_point in range(index_first_point,numbPoints):
        point1 = array_of_points[index_first_point]
        point2 = array_of_points[index_second_point]
        if edge_matrix[index_second_point][index_first_point] !=0 and edge_matrix[index_first_point][index_second_point] !=2 :
            pointX = [point1.x, point2.x]
            pointY = [point1.y, point2.y]
            ax[1].plot(pointX, pointY, 'black')



for edge in edges:
    point_x = [edge[0][0],edge[1][0]]
    point_y = [edge[0][1],edge[1][1]]
    ax[1].plot(point_x, point_y, 'green')


conver = []
for point in firewalls:
    conver.append([point.x,point.y])

X = np.array(conver)
if len(X) != 0:
    ax[1].scatter(X[:,0],X[:,1], c='green')




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
