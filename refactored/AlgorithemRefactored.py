import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from k_means_constrained import KMeansConstrained
from group import Group
from point import Point


def twoWayPartitioning(A, B, edgeMatrix, groups):
    subsetA = A.outerPoints
    subsetB = B.outerPoints
    groupA = A.name
    groupB = B.name
    indicator = 0
    while True:
        for index1 in range(len(subsetA)):
            ineternalCost = 0
            externalCost = 0
            # length of both subsets are the same
            for index2 in range(len(subsetA)):
                ineternalCost += edgeMatrix[subsetA[index1].index
                                            ][subsetA[index2].index]
                externalCost += edgeMatrix[subsetA[index1].index
                                           ][subsetB[index2].index]
            subsetA[index1].Dvalue = externalCost - ineternalCost

        for index1 in range(len(subsetB)):
            ineternalCost = 0
            externalCost = 0
            # length of both subsets are the same
            for index2 in range(len(subsetB)):
                ineternalCost += edgeMatrix[subsetB[index1]
                                            .index][subsetB[index2].index]
                externalCost += edgeMatrix[subsetB[index1]
                                           .index][subsetA[index2].index]
            subsetB[index1].Dvalue = externalCost - ineternalCost

        subsetA = sorted(subsetA, key=lambda x: x.Dvalue, reverse=True)
        subsetB = sorted(subsetB, key=lambda x: x.Dvalue, reverse=True)

        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        G = 0

        Xstar = []
        Ystar = []
        GBuffer = []

        # copies are used to maintain the state of the original sets
        subsetACopy = subsetA.copy()
        subsetBCopy = subsetB.copy()

        for i in range(len(subsetACopy)):
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

        for point in GBuffer[0]['Xstar']:
            subsetA.remove(point)

        for point in GBuffer[0]['Ystar']:
            subsetB.remove(point)

        # add the interchanged points to their new sets
        subsetA = subsetA + GBuffer[0]['Ystar']
        subsetB = subsetB + GBuffer[0]['Xstar']
        indicator = indicator + 1

        for i in range(len(subsetA)):  # update the group array
            subsetA[i].group = groupA
            groups[subsetA[i].index] = groupA
            subsetB[i].group = groupB
            groups[subsetB[i].index] = groupB

    return subsetA, subsetB, indicator, groups


def twoWayPartitioningEdgePoint(A, B, edgeMatrix, groups):
    subsetA = A.outerPoints
    subsetB = B.outerPoints
    groupA = A.name
    groupB = B.name
    indicator = 0
    while True:
        for index1 in range(len(subsetA)):
            ineternalCost = 0
            externalCost = 0
            # length of both subsets are the same
            for index2 in range(len(subsetA)):
                ineternalCost += edgeMatrix[subsetA[index1].index
                                            ][subsetA[index2].index]

            for index2 in range(len(subsetB)):
                externalCost += edgeMatrix[subsetA[index1].index
                                           ][subsetB[index2].index]

            subsetA[index1].Dvalue = externalCost - ineternalCost

        for index1 in range(len(subsetB)):
            ineternalCost = 0
            externalCost = 0
            # length of both subsets are the same
            for index2 in range(len(subsetB)):
                ineternalCost += edgeMatrix[subsetB[index1].index
                                            ][subsetB[index2].index]

            for index2 in range(len(subsetA)):
                externalCost += edgeMatrix[subsetB[index1].index
                                           ][subsetA[index2].index]

            subsetB[index1].Dvalue = externalCost - ineternalCost

        subsetA = sorted(subsetA, key=lambda x: x.Dvalue, reverse=True)
        subsetB = sorted(subsetB, key=lambda x: x.Dvalue, reverse=True)

        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        G = 0

        Xstar = []
        Ystar = []
        GBuffer = []

        # copies are used to maintain the state of the original sets
        subsetACopy = subsetA.copy()
        subsetBCopy = subsetB.copy()

        for i in range(len(subsetACopy)):
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

        for point in GBuffer[0]['Xstar']:
            subsetA.remove(point)

        for point in GBuffer[0]['Ystar']:
            subsetB.remove(point)

        # add the interchanged points to their new sets
        subsetA = subsetA + GBuffer[0]['Ystar']
        subsetB = subsetB + GBuffer[0]['Xstar']
        indicator = indicator + 1

        for i in range(len(subsetA)):  # update the group array

            subsetA[i].group = groupA
            groups[subsetA[i].index] = groupA
        

        for i in range(len(subsetB)):  # update the group array
            subsetB[i].group = groupB
            groups[subsetB[i].index] = groupB

    return subsetA, subsetB, indicator, groups


def divideIntoEvenClusters(x, y, numberOfClusters):  # done
    numbPoints = len(x)
    changeFormatArray = []
    for i in range(len(x)):  # here is merely changing the format of the array
        changeFormatArray.append([x[i], y[i]])
    X = np.array(changeFormatArray)
    kmeans = KMeansConstrained(n_clusters=numberOfClusters, size_min=numbPoints /
                               numberOfClusters, size_max=numbPoints/numberOfClusters)
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


def findOuterEdges(arrayOfSubsets, array_database, edgeMatrix):
    numbPoints = len(array_database)
    numberOfClusters = len(arrayOfSubsets)
    arrayOfEdgePoints = []
    for i in range(numberOfClusters):
        groupArray = arrayOfSubsets[i]['subset']
        for j in range(len(groupArray)):
            externalEdgesCounter = 0
            point1 = groupArray[j]
            whatGropus = []
            for k in range(numbPoints):
                point2 = array_database[k]
                differntGroups = not (point1['group'] == point2['group'])
                if edgeMatrix[point1['index']][point2['index']] == 1 and differntGroups:
                    if not (point2['group'] in whatGropus):
                        whatGropus.append(point2['group'])
                    externalEdgesCounter = externalEdgesCounter + 1
            if(externalEdgesCounter > 0):
                arrayOfEdgePoints.append(
                    {'point': point1, 'outerEdges': externalEdgesCounter, 'outerGroups': whatGropus})
    arrayOfEdgePoints = sorted(
        arrayOfEdgePoints, key=lambda x: x['outerEdges'], reverse=True)
    return 0 if len(arrayOfEdgePoints) == 0 else arrayOfEdgePoints


def onePointInterchange(groups, arrayOfEdgePoints, arrayOfSubsets, edgeMatrix):
    numbPoints = len(groups)
    numberOfClusters = len(arrayOfSubsets)
    for point in arrayOfEdgePoints:
        internalCost = 0
        for point2 in arrayOfSubsets[point['point']['group']]['subset']:
            if edgeMatrix[point['point']['index']][point2['index']] == 1:
                internalCost = internalCost + 1
        externalCostBuffer = []
        for group in point['outerGroups']:
            if(len(arrayOfSubsets[group]['subset']) > 1.05*(numbPoints/numberOfClusters)):
                break

            externalCost = 0
            for point2 in arrayOfSubsets[group]['subset']:
                if edgeMatrix[point['point']['index']][point2['index']] == 1:
                    externalCost = externalCost + 1
                    print(group)

            externalCostBuffer.append(
                {'externalCost': externalCost, 'group': group})
        externalCostBuffer = sorted(
            externalCostBuffer, key=lambda x: x['externalCost'], reverse=True)
        if externalCostBuffer[0]['externalCost'] - internalCost > 0:
            oldGroup = point['point']['group']
            newGroup = externalCostBuffer[0]['group']
            arrayOfSubsets[newGroup]['subset'].append(point['point'])
            arrayOfSubsets[oldGroup]['subset'].remove(point['point'])
            groups[point['point']['index']] = newGroup
            point['point']['group'] = newGroup


def findFirewalls(arrayOfEdgePoints, array_database, edgeMatrix, arrayOfSubsets):
    while arrayOfEdgePoints:  # finds firewalls and mark them with 2
        for point in array_database:
            firstOuterPoint = arrayOfEdgePoints[0]['point']['index']
            secondOuterPoint = point['index']
            if edgeMatrix[firstOuterPoint][secondOuterPoint] == 1:
                edgeMatrix[firstOuterPoint][secondOuterPoint] = 2
                edgeMatrix[secondOuterPoint][firstOuterPoint] = 2
        array_database[arrayOfEdgePoints.pop(
            0)['point']['index']]['infected'] = 2
        arrayOfEdgePoints = findOuterEdges(
            arrayOfSubsets, array_database, edgeMatrix)

    firewalls = []
    for point in array_database:
        if point['infected'] == 2:
            firewalls.append(point)

    return firewalls
