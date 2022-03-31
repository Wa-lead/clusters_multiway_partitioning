from operator import attrgetter
import numpy as np
from k_means_constrained import KMeansConstrained
from group import Group
import sys
from copy import deepcopy

sys.setrecursionlimit(1000000)


def two_way_partitioning(A, B, edge_matrix, groups):
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
            groups[point.index] = groupA

        for point in subsetB.outerPoints:  # update the group array
            groups[point.index] = groupB

    return indicator


def divide_even_clusters(x, y, num_of_clusters):  # done
    numbPoints = len(x)
    changeFormatArray = []
    for i in range(len(x)):  # here is merely changing the format of the array
        changeFormatArray.append([x[i], y[i]])
    X = np.array(changeFormatArray)
    kmeans = KMeansConstrained(n_clusters=num_of_clusters, size_min=int((numbPoints /
                               num_of_clusters)*0.9), size_max=int((numbPoints/num_of_clusters)*1.1)
                               )
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


def one_way_partioning(A, B, edge_matrix, groups, def_size, legal_increaserese):
    legalSizeUpperBoundery = def_size + def_size * legal_increaserese
    legalSizeLowerBoundery = def_size - def_size * legal_increaserese
    subsetA = A
    subsetB = B
    # groupA = A.name
    groupB = B.name
    indicator = 0

    # create deepcopies so we can modify those copies in the iteration withougt affecting the original sets
    subsetACopy = deepcopy(subsetA)
    subsetBCopy = deepcopy(subsetB)

    GBuffer = []
    A_to_B_candidates = []
    G = 0
    loopThreshold = int(min(len(subsetA.points) - legalSizeLowerBoundery,
                            legalSizeUpperBoundery - len(subsetB.points)))
    i = 0
    while i < loopThreshold and subsetACopy.outerPoints and subsetBCopy.outerPoints:
        for point in subsetACopy.outerPoints:
            ineternalCost = 0
            externalCost = 0
            # calculate the Dvalues of each set
            for point2 in subsetACopy.points:
                ineternalCost += edge_matrix[point.index
                                            ][point2.index]

            for point2 in subsetBCopy.points:
                externalCost += edge_matrix[point.index
                                           ][point2.index]
            point.Dvalue = externalCost - ineternalCost

        # find the highest candidate for interchange
        candidate = max(subsetACopy.outerPoints, key=attrgetter('Dvalue'))
        G += candidate.Dvalue  # add it to the gain
        # this buffers the current interchanged points
        A_to_B_candidates.append(candidate)
        subsetACopy.remove_outer_point(candidate)
        subsetBCopy.append_outer_point(candidate)
        # buffers both gain and the current interchanged points
        GBuffer.append(
            {'G': G, 'A_to_B_candidates': A_to_B_candidates.copy()})
        subsetBCopy.find_connected_groups()  # must update the groups
        subsetACopy.find_connected_groups()
        i += 1

    if(GBuffer):  # if the above process ocurred GBuffer would've stored something
        bestGain = sorted(GBuffer, key=lambda x: x['G'], reverse=True)[0]
        if bestGain['G'] > 0:  # if the highest buffered gain is positive
            # the reason behind creating this is that the current interchanged points are deep copies and are not stored by reference.
            B_to_add = []
            for point in bestGain['A_to_B_candidates']:
                for point2 in subsetA.outerPoints:
                    if point.index == point2.index:
                        B_to_add.append(point2)
                        subsetA.remove_outer_point(point2)

            for point in B_to_add:
                subsetB.append_outer_point(point)
                point.group = groupB
                groups[point.index] = groupB

            indicator += 1  # indicates the change has happened

    return indicator


def find_potential_firewalls(array_of_subsets):
    arrayOfEdgePoints = []
    for group in array_of_subsets:
        if(group.outerPoints):
            arrayOfEdgePoints = arrayOfEdgePoints + group.outerPoints

    potentialFirewalls = []
    if arrayOfEdgePoints:
        for point in arrayOfEdgePoints:
            outerEdges = [
                p for p in point.connected_points if p.group != point.group]
            potentialFirewalls.append(
                {'point': point, 'outerEdges': outerEdges})

    return potentialFirewalls


def find_firewalls(array_of_subsets, protect=None):
    if protect == 'self' or protect == None:
        actualFirewalls = []
        protectedEdges = []
        # This finds all the potential ( outer points ) firewalls, then sort them based NUMBER of outer edges
        potentialFirewalls = sorted(find_potential_firewalls(
            array_of_subsets), key=lambda x: len(x['outerEdges']), reverse=True)

        if potentialFirewalls:  # if there are outerpoints:
            """
            The following code is to remove the the current selected candidate from the count of other potentials
            outer edge, why?
            because once we assign a firewall, all its edges will be protected, and hence, must be removed
            from the count of the following selections.
            """
            for potential in potentialFirewalls:  # this segemnts is to remove the
                for potential2 in potentialFirewalls:
                    if potential['point'] in potential2['outerEdges']:
                        potential2['outerEdges'].remove(potential['point'])
                # if the potential still has edges the above removal of already-protected edges
                if(potential['outerEdges']):
                    current_selected_firewall = potential['point']
                    current_selected_firewall.state =1
                    actualFirewalls.append(current_selected_firewall)

                    # this assignes the value 2 which means protected to the points connected to the current firewall
                    for point in current_selected_firewall.connected_points:
                        protectedEdges.append([(current_selected_firewall.x, current_selected_firewall.y), (point.x, point.y)])
                    

                # Sort the current potientials based on number of outer edges again
                potentialFirewalls = sorted(
                    potentialFirewalls, key=lambda x: len(x['outerEdges']), reverse=True)

        return actualFirewalls, protectedEdges
        # ------------------------------- this is another variation that protects the connected nodes
    elif protect == 'connected':
        actualFirewalls = []
        protectedEdges = []
        # This finds all the potential ( outer points ) firewalls, then sort them based NUMBER of outer edges
        potentialFirewalls = sorted(find_potential_firewalls(
            array_of_subsets), key=lambda x: len(x['outerEdges']), reverse=True)

        if potentialFirewalls:  # if there are outerpoints:
            """
            The following code is to remove the the current selected candidate from the count of other potentials
            outer edge, why?
            because once we assign a firewall, all its edges will be protected, and hence, must be removed
            from the count of the following selections.
            """
            for potential in potentialFirewalls:  
                for potential2 in potentialFirewalls:
                    if potential['point'] in potential2['outerEdges']:
                        potential2['outerEdges'].remove(potential['point'])
                # if the potential still has edges the above removal of already-protected edges
                if(potential['outerEdges']):
                    current_selected_firewall = potential['point']
                    if current_selected_firewall.state != 2:
                        current_selected_firewall.state =1
                        actualFirewalls.append(current_selected_firewall)

                    # this assignes the value 2 which means protected to the points connected to the current firewall
                    for point in current_selected_firewall.connected_points:
                        """
                        For future reference: if applying a simulation about infection, we will need also to update the edge matrix tha ease the process of tracking protected edges.
                        """
                        point.state = 2
                        protectedEdges.append([(current_selected_firewall.x, current_selected_firewall.y), (point.x, point.y)])
                        #now we also protect the edges of the protected (state: 2)  points
                        for point2 in point.connected_points:
                            protectedEdges.append([(point.x, point.y), (point2.x, point2.y)])
                

                potentialFirewalls = [firewall for firewall in potentialFirewalls if firewall['point'].state!=2] #<-- keeps only the unprotected potential candidates

                # Sort the current potientials based on number of outer edges again
                potentialFirewalls = sorted(
                    potentialFirewalls, key=lambda x: len(x['outerEdges']), reverse=True)

        return actualFirewalls, protectedEdges
