from asyncore import loop
from operator import attrgetter
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from k_means_constrained import KMeansConstrained
from group import Group
import sys
from copy import deepcopy

sys.setrecursionlimit(1000000)


def twoWayPartitioningEdgePoint(A, B, edgeMatrix, groups):
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
                ineternalCost += edgeMatrix[point.index
                                            ][point2.index]

            for point2 in subsetB.points:
                externalCost += edgeMatrix[point.index
                                           ][point2.index]

            point.Dvalue = externalCost - ineternalCost

        for point in subsetB.outerPoints:
            ineternalCost = 0
            externalCost = 0
            # length of both subsets are the same
            for point2 in subsetB.points:
                ineternalCost += edgeMatrix[point.index
                                            ][point2.index]

            for point2 in subsetA.points:
                externalCost += edgeMatrix[point.index
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
                        2*edgeMatrix[subsetACopy[j].index
                                     ][subsetBCopy[k].index]
                    if newGain > gain:
                        gain = newGain
                        indexOfa1 = j
                        indexOfb1 = k

            G += gain
            # Recalculating the Dvalue of each in the session
            for j in range(len(subsetACopy)):
                subsetACopy[j].Dvalue = subsetACopy[j].Dvalue + 2*edgeMatrix[subsetACopy[j].index
                                                                             ][subsetACopy[indexOfa1].index] - 2*edgeMatrix[subsetACopy[j].index][subsetBCopy[indexOfb1].index]

            for j in range(len(subsetBCopy)):
                subsetBCopy[j].Dvalue = subsetBCopy[j].Dvalue + 2*edgeMatrix[subsetBCopy[j].index
                                                                             ][subsetBCopy[indexOfb1].index] - 2*edgeMatrix[subsetBCopy[j].index][subsetACopy[indexOfa1].index]
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
            subsetA.removeOuterPoint(point)
            subsetB.appendOuterPoint(point)

        for point in GBuffer[0]['Ystar']:
            subsetB.removeOuterPoint(point)
            subsetA.appendOuterPoint(point)

        # add the interchanged points to their new sets
        indicator = indicator + 1

        for point in subsetA.outerPoints:  # update the group array
            groups[point.index] = groupA

        for point in subsetB.outerPoints:  # update the group array
            groups[point.index] = groupB

    return indicator


def divideIntoEvenClusters(x, y, numberOfClusters):  # done
    numbPoints = len(x)
    changeFormatArray = []
    for i in range(len(x)):  # here is merely changing the format of the array
        changeFormatArray.append([x[i], y[i]])
    X = np.array(changeFormatArray)
    # kmeans = KMeansConstrained(n_clusters=numberOfClusters, size_min=(numbPoints /
    #                            numberOfClusters)*0.9, size_max=(numbPoints/numberOfClusters)
    #                            * 1.10)
    kmeans = KMeansConstrained(n_clusters=numberOfClusters, size_min=int((numbPoints /
                               numberOfClusters)*0.9), size_max=int((numbPoints/numberOfClusters)*1.1)
                               )
    kmeans.fit(X)
    return kmeans.labels_


def getClustersPoints(numberOfClusters, array_database):  # done
    arrayOfSubsets = []
    for i in range(numberOfClusters):
        arrayOfSubsets.append(Group(i))
        for point in array_database:
            if(point.group == i):
                arrayOfSubsets[i].add(point)
    return arrayOfSubsets


def onePointInterchange(A, B, edgeMatrix, groups, defaultSize, legalIncrese):
    legalSize = defaultSize + defaultSize * legalIncrese
    subsetA = A
    subsetB = B
    groupA = A.name
    groupB = B.name
    indicator = 0
    while subsetA.outerPoints and subsetB.outerPoints:
        for point in subsetA.outerPoints:
            ineternalCost = 0
            externalCost = 0
            # length of both subsets are the same
            for point2 in subsetA.points:
                ineternalCost += edgeMatrix[point.index
                                            ][point2.index]

            for point2 in subsetB.points:
                externalCost += edgeMatrix[point.index
                                           ][point2.index]

            point.Dvalue = externalCost - ineternalCost

        for point in subsetB.outerPoints:
            ineternalCost = 0
            externalCost = 0
            # length of both subsets are the same
            for point2 in subsetB.points:
                ineternalCost += edgeMatrix[point.index
                                            ][point2.index]

            for point2 in subsetA.points:
                externalCost += edgeMatrix[point.index
                                           ][point2.index]

            point.Dvalue = externalCost - ineternalCost

        subsetA.outerPoints = sorted(
            subsetA.outerPoints, key=lambda x: x.Dvalue, reverse=True)
        subsetB.outerPoints = sorted(
            subsetB.outerPoints, key=lambda x: x.Dvalue, reverse=True)

        subsetACandidate = subsetA.outerPoints[0]
        subsetBCandidate = subsetB.outerPoints[0]

        # subsetACandidate = min(subsetA.outerPoints, key=attrgetter('Dvalue'))
        # subsetBCandidate = min(subsetB.outerPoints, key=attrgetter('Dvalue'))

        if len(subsetA.points) <= legalSize and len(subsetB.points) <= legalSize and (subsetACandidate.Dvalue > 0 or subsetBCandidate.Dvalue > 0):
            indicator += 1
            if(subsetACandidate.Dvalue > subsetBCandidate.Dvalue):
                subsetB.appendOuterPoint(subsetACandidate)
                subsetA.removeOuterPoint(subsetACandidate)
                groups[subsetACandidate.index] = groupB
            else:
                subsetA.appendOuterPoint(subsetBCandidate)
                subsetB.removeOuterPoint(subsetBCandidate)
                groups[subsetBCandidate.index] = groupA

        else:
            break

    return indicator


def onePointInterchangeEnhanced(A, B, edgeMatrix, groups, defaultSize, legalIncrese):
    legalSizeUpperBoundery = defaultSize + defaultSize * legalIncrese
    legalSizeLowerBoundery = defaultSize - defaultSize * legalIncrese
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
                ineternalCost += edgeMatrix[point.index
                                            ][point2.index]

            for point2 in subsetBCopy.points:
                externalCost += edgeMatrix[point.index
                                           ][point2.index]
            point.Dvalue = externalCost - ineternalCost

        # find the highest candidate for interchange
        candidate = max(subsetACopy.outerPoints, key=attrgetter('Dvalue'))
        G += candidate.Dvalue  # add it to the gain
        # this buffers the current interchanged points
        A_to_B_candidates.append(candidate)
        subsetACopy.removeOuterPoint(candidate)
        subsetBCopy.appendOuterPoint(candidate)
        # buffers both gain and the current interchanged points
        GBuffer.append(
            {'G': G, 'A_to_B_candidates': A_to_B_candidates.copy()})
        subsetBCopy.findConnectedGroup()  # must update the groups
        subsetACopy.findConnectedGroup()
        i+=1

    if(GBuffer):  # if the above process ocurred GBuffer would've stored something
        bestGain = sorted(GBuffer, key=lambda x: x['G'], reverse=True)[0]
        if bestGain['G'] > 0:  # if the highest buffered gain is positive
            # the reason behind creating this is that the current interchanged points are deep copies and are not stored by reference.
            B_to_add = []
            for point in bestGain['A_to_B_candidates']:
                for point2 in subsetA.outerPoints:
                    if point.index == point2.index:
                        B_to_add.append(point2)
                        subsetA.removeOuterPoint(point2)

            for point in B_to_add:
                subsetB.appendOuterPoint(point)
                point.group = groupB
                groups[point.index] = groupB

            indicator += 1  # indicates the change has happened

    return indicator


def findPotentialFireWalls(arrayOfSubsets):
    arrayOfEdgePoints = []
    for group in arrayOfSubsets:
        if(group.outerPoints):
            arrayOfEdgePoints = arrayOfEdgePoints + group.outerPoints

    potentialFirewalls = []
    if arrayOfEdgePoints:
        for point in arrayOfEdgePoints:
            outerEdges = [
                p for p in point.connectedWith if p.group != point.group]
            potentialFirewalls.append(
                {'point': point, 'outerEdges': outerEdges})

    return potentialFirewalls


def findFirewalls(arrayOfSubsets, edgeMatrix):
    actualFirewalls = []
    potentialFirewalls = sorted(findPotentialFireWalls(
        arrayOfSubsets), key=lambda x: len(x['outerEdges']), reverse=True)

    if potentialFirewalls:
        for potential in potentialFirewalls:
            for potential2 in potentialFirewalls:
                if potential['point'] in potential2['outerEdges']:
                    potential2['outerEdges'].remove(potential['point'])

            if(potential['outerEdges']):
                actualFirewalls.append(potential['point'])
                edgeMatrix[potential['point'].index] = [
                    2 if item == 1 else item for item in edgeMatrix[potential['point'].index]]
            potentialFirewalls = sorted(
                potentialFirewalls, key=lambda x: len(x['outerEdges']), reverse=True)

    return actualFirewalls
