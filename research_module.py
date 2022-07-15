import numpy as np
from k_means_constrained import KMeansConstrained
from group import Group
import sys
from copy import deepcopy
from math import *

sys.setrecursionlimit(1000000)

#---- setup the problem formulation

def form_edge_matrix(array_of_points, connect_distance):
    numbPoints = len(array_of_points)
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

    return edge_matrix

def divide_even_clusters(x, y, num_of_clusters):  # done
    numbPoints = len(x)
    changeFormatArray = []
    for i in range(len(x)):  # here is merely changing the format of the array
        changeFormatArray.append([x[i], y[i]])
    X = np.array(changeFormatArray)
    kmeans = KMeansConstrained(n_clusters=num_of_clusters, size_min=int((numbPoints /
                               num_of_clusters)*0.9), size_max=int((numbPoints/num_of_clusters)*1.1), max_iter=10000)
    kmeans.fit(X)
    return kmeans.labels_

def get_cluster_points(num_of_clusters, array_of_points):  # done
    array_of_subsets = []
    for i in range(num_of_clusters):
        array_of_subsets.append(Group(i))
        for point in array_of_points:
            if(point.group == i):
                array_of_subsets[i].add(point)
    return array_of_subsets

#---- paritioing

def two_way_partitioning(A, B, edge_matrix, groups_):
    subsetA = A
    subsetB = B
    groupA = A.name
    groupB = B.name
    indicator = 0
    while True:
        for point in subsetA.outerPoints:
            ineternalCost = 0
            externalCost = 0
            # length of both subsets are the same
            for point2 in subsetA.points:
                ineternalCost += edge_matrix[point.index
                                             ][point2.index]

            for point2 in subsetB.points:
                externalCost += edge_matrix[point.index
                                            ][point2.index]

            point.Dvalue = externalCost - ineternalCost

        for point in subsetB.outerPoints:
            ineternalCost = 0
            externalCost = 0
            # length of both subsets are the same
            for point2 in subsetB.points:
                ineternalCost += edge_matrix[point.index
                                             ][point2.index]

            for point2 in subsetA.points:
                externalCost += edge_matrix[point.index
                                            ][point2.index]

            point.Dvalue = externalCost - ineternalCost

        subsetA.outerPoints = sorted(
            subsetA.outerPoints, key=lambda x: x.Dvalue, reverse=True)
        subsetB.outerPoints = sorted(
            subsetB.outerPoints, key=lambda x: x.Dvalue, reverse=True)

        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        G = 0

        Xstar = []
        Ystar = []
        GBuffer = []

        # copies are used to maintain the state of the original sets
        subsetACopy = subsetA.outerPoints.copy()
        subsetBCopy = subsetB.outerPoints.copy()
        smallerSet = subsetACopy if len(subsetACopy) < len(
            subsetBCopy) else subsetBCopy
        for i in range(len(smallerSet)):
            gain = -10000
            indexOfa1 = 0
            indexOfb1 = 0
            # find the highest gain
            for j in range(len(subsetACopy)):
                for k in range(len(subsetBCopy)):
                    newGain = subsetACopy[j].Dvalue + subsetBCopy[k].Dvalue - \
                        2*edge_matrix[subsetACopy[j].index
                                      ][subsetBCopy[k].index]
                    if newGain > gain:
                        gain = newGain
                        indexOfa1 = j
                        indexOfb1 = k

            G += gain
            # Recalculating the Dvalue of each in the session
            for j in range(len(subsetACopy)):
                subsetACopy[j].Dvalue = subsetACopy[j].Dvalue + 2*edge_matrix[subsetACopy[j].index
                                                                              ][subsetACopy[indexOfa1].index] - 2*edge_matrix[subsetACopy[j].index][subsetBCopy[indexOfb1].index]

            for j in range(len(subsetBCopy)):
                subsetBCopy[j].Dvalue = subsetBCopy[j].Dvalue + 2*edge_matrix[subsetBCopy[j].index
                                                                              ][subsetBCopy[indexOfb1].index] - 2*edge_matrix[subsetBCopy[j].index][subsetACopy[indexOfa1].index]
            # ------------------------------------------- the mistake is here, fix the popping problem

            # remove the highest gain points from the set
            Xstar.append(subsetACopy.pop(indexOfa1))
            Ystar.append(subsetBCopy.pop(indexOfb1))

            # buffers the gain
            GBuffer.append(
                {"index": i, "G": G, "Xstar": Xstar.copy(), "Ystar": Ystar.copy()})

            # sorting based on Dvalue again
            subsetACopy = sorted(
                subsetACopy, key=lambda x: x.Dvalue, reverse=True)
            subsetBCopy = sorted(
                subsetBCopy, key=lambda x: x.Dvalue, reverse=True)

        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # remove highest gain exchanges from the subsetA and subsetB
        GBuffer = sorted(GBuffer, key=lambda x: x['G'], reverse=True)

        if(GBuffer[0]['G'] <= 0.000000000001):
            break
            # create a function that deletes outer points and delete it from points too
            for point in GBuffer[0]['Xstar']:
                subsetA.remove_outer_point(point)
                subsetB.append_outer_point(point)

            for point in GBuffer[0]['Ystar']:
                subsetB.remove_outer_point(point)
                subsetA.append_outer_point(point)

            # add the interchanged points to their new sets
            indicator = indicator + 1

            for point in subsetA.outerPoints:  # update the group array
                groups_[point.index] = groupA

            for point in subsetB.outerPoints:  # update the group array
                groups_[point.index] = groupB

        return indicator

def two_way_partitioning_enhanced(A, B, groups_):
    subsetA = A
    subsetB = B
    groupA = A.name
    groupB = B.name
    indicator = 0

    while True:
        # compute the d values of both sets' points (block 1 in the flowchart)
        for point in subsetA.outerPoints:
            ineternalCost = 0
            externalCost = 0
            # length of both subsets are the same
            for point2 in point.connected_points:
                if point2.group == groupA:
                    ineternalCost += 1 
                elif point2.group == groupB:
                    externalCost += 1
            point.Dvalue = externalCost - ineternalCost

        for point in subsetB.outerPoints:
            ineternalCost = 0
            externalCost = 0
            # length of both subsets are the same
            for point2 in point.connected_points:
                if point2.group == groupB:
                    ineternalCost += 1 
                elif point2.group == groupA:
                    externalCost += 1
            point.Dvalue = externalCost - ineternalCost


        #this changes the set to a list for order purposes
        subsetA.outerPoints = sorted(
            subsetA.outerPoints, key=lambda x: x.Dvalue, reverse=True)
        subsetB.outerPoints = sorted(
            subsetB.outerPoints, key=lambda x: x.Dvalue, reverse=True)

        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        G = 0

        Xstar = []
        Ystar = []
        GBuffer = []

        # copies are used to maintain the state of the original sets
        subsetACopy = subsetA.outerPoints.copy()
        subsetBCopy = subsetB.outerPoints.copy()
        smallerSet = subsetACopy if len(subsetACopy) < len(
            subsetBCopy) else subsetBCopy

        # select aj and bj such that gain is maximized (block 2 in the flowachrt)
        for i in range(len(smallerSet)):
            gain = float('-inf')
            indexOfa1 = 0
            indexOfb1 = 0
            for j, point1 in enumerate(subsetACopy):
                for k, point2 in enumerate(subsetBCopy):
                    newGain = point1.Dvalue + point2.Dvalue - \
                        2* (1 if point2 in point1.connected_points else 0)
                    if newGain > gain:
                        gain = newGain
                        indexOfa1 = j
                        indexOfb1 = k

            G += gain

            A_to_B = subsetACopy[indexOfa1]
            B_to_A = subsetBCopy[indexOfb1]
            # # Recalculating the Dvalue of each in the session
          
            for point in subsetACopy:
                point.Dvalue = point.Dvalue + 2*(1 if A_to_B in point.connected_points else 0) - 2*(1 if B_to_A in point.connected_points else 0)

            for point in subsetBCopy:
                point.Dvalue = point.Dvalue + 2*(1 if B_to_A in point.connected_points else 0) - 2*(1 if A_to_B in point.connected_points else 0)

            # remove the highest gain points from the set
            Xstar.append(subsetACopy.pop(indexOfa1))
            Ystar.append(subsetBCopy.pop(indexOfb1))

            # buffers the gain
            GBuffer.append(
                {"index": i, "G": G, "Xstar": Xstar.copy(), "Ystar": Ystar.copy()})

            # sorting based on Dvalue again
            subsetACopy = sorted(
                subsetACopy, key=lambda x: x.Dvalue, reverse=True)
            subsetBCopy = sorted(
                subsetBCopy, key=lambda x: x.Dvalue, reverse=True)

        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # remove highest gain exchanges from the subsetA and subsetB
        if GBuffer:
            GBuffer = max(GBuffer, key=lambda x: x['G'])

            if(GBuffer['G'] <= 0.000000000001):
                break

            for point in GBuffer['Xstar']:
                #convert back to set
                subsetB.outerPoints = set(subsetB.outerPoints)
                subsetA.outerPoints = set(subsetA.outerPoints)

                subsetA.remove_outer_point(point)
                subsetB.append_outer_point(point)
                groups_[point.index] = groupB


            for point in GBuffer['Ystar']:
                #convert back to set
                subsetB.outerPoints = set(subsetB.outerPoints)
                subsetA.outerPoints = set(subsetA.outerPoints)

                subsetB.remove_outer_point(point)
                subsetA.append_outer_point(point)
                groups_[point.index] = groupA


            # add the interchanged points to their new sets
            indicator = indicator + 1

    
    return indicator

def one_way_partioning(A, B, edge_matrix, groups_, def_size, legal_increaserese):
    UPPER_BOUND = def_size + def_size * legal_increaserese
    LOWER_BOUND = def_size - def_size * legal_increaserese
    groupB = B.name
    indicator = 0

    # create deepcopies so we can modify those copies in the iteration withougt affecting the original sets
    subsetACopy = deepcopy(A)
    subsetBCopy = deepcopy(B)

    GBuffer = []
    A_to_B_candidates = []
    G = 0
    loopThreshold = int(min(len(A.points) - LOWER_BOUND,
                            UPPER_BOUND - len(B.points)))
    i = 0
    while i < loopThreshold and subsetACopy.outerPoints and subsetBCopy.outerPoints:
        A_B_connections = set()
        B_A_connections = set()
        

        # detect the common conncections betweent the two groups
        for p in subsetACopy.outerPoints:
            for j in p.connected_points:
                if j.group == groupB:
                    A_B_connections.add(p)
                    B_A_connections.add(j)

        # If there are no connections between the two groups then end the process
        if not A_B_connections or not B_A_connections:
            break

        for point in A_B_connections:
            ineternalCost = 0
            externalCost = 0
            # calculate the Dvalues of each set
            for point2 in subsetACopy.points:
                ineternalCost += edge_matrix[point.index
                                             ][point2.index]

            for point2 in B_A_connections:
                externalCost += edge_matrix[point.index
                                            ][point2.index]
            point.Dvalue = externalCost - ineternalCost

        # find the highest candidate for interchange
        # candidate = max(subsetACopy.outerPoints, key= lambda x: (x.Dvalue,len(x.connected_points)))
        candidate = max(subsetACopy.outerPoints, key=lambda x: x.Dvalue)
        G += candidate.Dvalue  # add it to the gain
        # this buffers the current interchanged points
        A_to_B_candidates.append(candidate)
        subsetACopy.remove_outer_point(candidate)
        subsetBCopy.append_outer_point(candidate)
        # buffers both gain and the current interchanged points
        GBuffer.append(
            {'G': G, 'A_to_B_candidates': A_to_B_candidates.copy()})
        subsetBCopy.find_connected_groups()  # must update the groups_
        subsetACopy.find_connected_groups()
        i += 1

    if(GBuffer):  # if the above process ocurred GBuffer would've stored something
        bestGain = max(GBuffer, key=lambda x: x['G'])
        if bestGain['G'] > 0:  # if the highest buffered gain is positive
            # the reason behind creating this is that the current interchanged points are deep copies and are not stored by reference.
            for point in bestGain['A_to_B_candidates']:
                for point2 in A.outerPoints:
                    if point.index == point2.index:
                        B.append_outer_point(point2)
                        groups_[point2.index] = groupB
                        A.remove_outer_point(point2)

            indicator += 1  # indicates the change has happened

    return indicator


def cluster_size(point):
    seen_points = set()
    cluster_content = []
    def dfs(point):
        for p in point.connected_points:
            if p in seen_points or p.group != point.group:
                continue
            seen_points.add(p)
            cluster_content.append(p)
            dfs(p)

    dfs(point)
    return len(cluster_content)

def selection_hueristic1(current_candidates,threshold):   

    def hueristic(point):
        size = cluster_size(point) 
        if size > threshold:
            return float('-inf')
        else:
            return point.Dvalue

    return max(current_candidates, key= hueristic)

def selection_hueristic2(current_candidates):    
    return max(current_candidates, key = lambda x : x.Dvalue)

def selection_hueristic3(current_candidates, buffered_candidates):
    # (Dvalue, common points between its connectect points and already selected ones)
    return max(current_candidates, key= lambda x: (x.Dvalue,len(set(x.connected_points) & set(buffered_candidates))))

def one_way_partioning_enhanced(A, B, groups_, def_size, legal_increaserese):
    """
    from A to B
    """

    #Define the coundaries
    UPPER_BOUND = int(def_size + def_size * legal_increaserese)
    LOWER_BOUND = int(def_size - def_size * legal_increaserese)

    groupA = A.name
    groupB = B.name

    # used to show if an interchanged occured between the two groups
    indicator = 0

    # create deepcopies so we can modify those copies in the iteration withougt affecting the original sets
    subsetACopy = deepcopy(A)
    subsetBCopy = deepcopy(B)

    #buffers each iteration
    GBuffer = []
    A_to_B_candidates = []
    #Total gain
    G = 0

    loopThreshold = int(min(len(A.points) - LOWER_BOUND,
                            UPPER_BOUND - len(B.points)))
    i = 0
    while i <= loopThreshold and subsetACopy.outerPoints and subsetBCopy.outerPoints:
        A_B_connections = set()
        

        # detect the conncections betweent the two groups
        for p in subsetACopy.outerPoints:
            for j in p.connected_points:
                if j.group == groupB:
                    A_B_connections.add(p)

        # If there are no connections between the two groups then end the process
        if not A_B_connections:
            break
        
        #calculates the D-value for the connected points
        for point in A_B_connections:
            ineternalCost = 0
            externalCost = 0
            # length of both subsets are the same
            for point2 in point.connected_points:
                if point2.group == groupA:
                    ineternalCost += 1 
                elif point2.group == groupB:
                    externalCost += 1
            point.Dvalue = externalCost - ineternalCost

        # find the highest candidate for interchange
        # candidate = selection_hueristic1(A_B_connections, loopThreshold - i )
        candidate = selection_hueristic2(A_B_connections)
        # candidate = selection_hueristic3(A_B_connections, GBuffer[-1]['A_to_B_candidates'] if  GBuffer else [])

        
        G += candidate.Dvalue  # add it to the gain
        # this buffers the current interchanged points
        A_to_B_candidates.append(candidate)
        subsetACopy.remove_outer_point(candidate)
        subsetBCopy.append_outer_point(candidate)
        # buffers both gain and the current interchanged points
        
        GBuffer.append(
            {'G': G, 'A_to_B_candidates': A_to_B_candidates.copy()})
        subsetBCopy.find_connected_groups()  # must update the groups_
        subsetACopy.find_connected_groups()
        i += 1

    if(GBuffer):  # if the above process ocurred GBuffer would've stored something
        bestGain = max(GBuffer, key=lambda x: x['G'])
        if bestGain['G'] > 0:  # if the highest buffered gain is positive
            # the reason behind creating this is that the current interchanged points are deep copies and are not stored by reference.
            for point in bestGain['A_to_B_candidates']:
                for point2 in list(A.outerPoints):
                    if point.index == point2.index:
                        B.append_outer_point(point2)
                        groups_[point2.index] = groupB
                        A.remove_outer_point(point2)

            indicator += 1  # indicates the change has happened

    return indicator



    

