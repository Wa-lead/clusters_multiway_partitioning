import numpy as np
import matplotlib.pyplot as plt
from math import floor
import scipy.stats
import matplotlib.pyplot as plt
from research_module import *
from point import Point
from plot import*


# fig, ax = plt.subplots(2, figsize=(4,7))
connect_distance = 1
fig, ax = plt.subplots()

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

lambda0 = 2
numbPoints = (scipy.stats.poisson(lambda0*areaTotal).rvs())
x = np.load('x.npy', allow_pickle=True)
y = np.load('y.npy', allow_pickle=True)
numbPoints = len(x)

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

#create edge matrix --> this also creates which point is connected to which
edge_matrix = form_edge_matrix(array_of_points, connect_distance)

#hard-coded values
desired_number_of_firewalls = 30
min_points_per_cluster = 20

#possible values start from 2 groups to n groups of 20 points each
possible_number_of_clusters = [i for i in range(
    2, int(numbPoints/min_points_per_cluster))]


#binary tree logic starts here
high = len(possible_number_of_clusters)-1
low = 0

cluster_to_firewalls_buffer = []
array_of_groups = []
groups_ = []

while low <= high:

    print('ss')

    #select the candidate clusters for bisection
    mid = int((low+high)/2)
    cnadidate_number_of_clusters = possible_number_of_clusters[mid]

    #-----------------------------------------------------------------------------------------
    #set up the the groups so we can find the firewalls required
    groups_ = divide_even_clusters(x, y, cnadidate_number_of_clusters)
    for i in range(numbPoints):
        array_of_points[i].group = groups_[i]

    # we prepare the array that holds all the groups with their points
    array_of_groups = get_cluster_points(
        cnadidate_number_of_clusters, array_of_points)
    for group in array_of_groups:
        group.find_connected_groups()
    #-----------------------------------------------------------------------------------------

    #start multiway partitioning
    i = 0
    while i < cnadidate_number_of_clusters:
        startAgain = False
        for group in array_of_groups[i].connectedGroup:
            indicator = two_way_partitioning_enhanced(
                array_of_groups[i], array_of_groups[group], groups_)
            if indicator > 0:
                startAgain = True
                array_of_groups[i].find_connected_groups()
                array_of_groups[group].find_connected_groups()
        i = 0 if startAgain else i + 1

    #update the current graph formulation
    for group in array_of_groups:
        group.find_connected_groups()

    #start one way partitioning
    i = 0
    while i < cnadidate_number_of_clusters:
        print('br')
        startAgain = False
        for group in array_of_groups[i].connectedGroup:
            indicator = one_way_partioning_enhanced(
                array_of_groups[i], array_of_groups[group], groups_, numbPoints/cnadidate_number_of_clusters, 0.1)
            if indicator > 0:
                array_of_groups[i].find_connected_groups()
                array_of_groups[group].find_connected_groups()
                startAgain = True
        i = 0 if startAgain else i + 1

    firewalls, edges = find_firewalls(array_of_groups, protect='self')
    cluster_to_firewalls_buffer.append(
        (f'clusters: {cnadidate_number_of_clusters}', f'firewalls: {len(firewalls)}'))

    # binary tree updates
    if len(firewalls) >= desired_number_of_firewalls:
        high = mid - 1
    else:
        low = mid + 1

plot_points(ax, x, y, groups_, array_of_groups)
plot_firewalls(ax, array_of_groups, protect='self')
print(cluster_to_firewalls_buffer)

plt.show()
