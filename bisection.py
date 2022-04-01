import numpy as np
import matplotlib.pyplot as plt
from math import floor, sqrt
import scipy.stats
import matplotlib.pyplot as plt
from research_module import *
from copy import copy, deepcopy
from point import Point


fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(30, 5))
# fig, ax = plt.subplots(2, figsize=(4,7))
connect_distance = 1

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
numbPoints = (scipy.stats.poisson(lambda0*areaTotal).rvs())
x = np.random.uniform(size=numbPoints, low=xMin, high=xMax)
y = np.random.uniform(size=numbPoints, low=yMin,
                      high=yMax)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Dividing the set into several even-size clusters
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Preparing The data point data base

array_of_points = []

for i in range(numbPoints):
    """
    The last index of the point object indecates the state of the point:
    0: Firewall
    1: Normal 
    2: Protected --> protects itself and it's edges (not a firewall) <---- this is used in the second approach of firewalls
    -----------------------------------
    Initially we'll initilize the poitns with no grouops
    """
    array_of_points.append(Point(None, i, x[i], y[i], 1, 0)
                           )



#--------------------------- create edge matrix
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

#-------------------------------------- prepare vars for bisestion
desired_number_of_firewalls = 20 
possible_num_of_clusters = [i for i in range(2, 17)] #this sets the range of acceptable clusters

#bestion vars -- similar to binary tree
_high = len(possible_num_of_clusters)-1 
_low = 0

#we need the buffer for comparison purposes
cluster_to_firewalls_buffer = []

# start of besction
blah = 0 #<--- just for plutting purposes
while _high > _low:

    mid = floor((_low+_high)/2) #calculate the mid for splitting purposes
    num_of_clusters = mid

    groups_ = divide_even_clusters(x, y, num_of_clusters)# divide the current points into clusters

    #now we assign the groups to each point
    for i in range(numbPoints):
        array_of_points[i].group = groups_[i]

    #we prepare the array that holds all the groups with their points
    array_of_groups = get_cluster_points(num_of_clusters, array_of_points)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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

    # one point interchange
    i = 0
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

    #This finds the firewalls of the current distribution along with the protected edges based on the selected variation
    firewalls, edges = find_firewalls(array_of_groups, protect='self')
    cluster_to_firewalls_buffer.append((num_of_clusters, len(firewalls)))

    ax[blah].scatter(x, y, c=groups_, cmap='rainbow')
    for edge in edges:
        point_x = [edge[0][0], edge[1][0]]
        point_y = [edge[0][1], edge[1][1]]
        ax[blah].plot(point_x, point_y, 'green')

    conver = []
    for point in firewalls:
        conver.append([point.x, point.y])

    X = np.array(conver)
    if len(X) != 0:
        ax[blah].scatter(X[:, 0], X[:, 1], c='green')

    blah += 1

    #if the number of firewalls is met, then break, NOT SURE ABOUT THIS POINT
    if(len(firewalls)) == desired_number_of_firewalls:
        break
    
    #typical binary tree logic
    elif len(firewalls) > desired_number_of_firewalls:
        _high = mid - 1

    else:
        _low = mid + 1


print(cluster_to_firewalls_buffer)
plt.show()
