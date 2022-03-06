import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from k_means_constrained import KMeansConstrained
from group import Group
from point import Point
import sys
sys.setrecursionlimit(10000)



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

        subsetA.outerPoints = sorted(subsetA.outerPoints, key=lambda x: x.Dvalue, reverse=True)
        subsetB.outerPoints = sorted(subsetB.outerPoints, key=lambda x: x.Dvalue, reverse=True)

        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        G = 0

        Xstar = []
        Ystar = []
        GBuffer = []

        # copies are used to maintain the state of the original sets
        subsetACopy = subsetA.outerPoints.copy()
        subsetBCopy = subsetB.outerPoints.copy()
        smallerSet = subsetACopy if len(subsetACopy) < len(subsetBCopy) else subsetBCopy
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

        subsetA.outerPoints = sorted(subsetA.outerPoints, key=lambda x: x.Dvalue, reverse=True)
        subsetB.outerPoints = sorted(subsetB.outerPoints, key=lambda x: x.Dvalue, reverse=True)

        subsetACandidate = subsetA.outerPoints[0]
        subsetBCandidate = subsetB.outerPoints[0]

        if len(subsetA.points) <= legalSize and len(subsetB.points) <= legalSize and (subsetACandidate.Dvalue>0 or subsetBCandidate.Dvalue>0):
            indicator += 1
            if(subsetACandidate.Dvalue>subsetBCandidate.Dvalue):
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
    

def findPotentialFireWalls(arrayOfSubsets):
    arrayOfEdgePoints = []
    for group in arrayOfSubsets:
        if(group.outerPoints):
            arrayOfEdgePoints = arrayOfEdgePoints + group.outerPoints

    potentialFirewalls = []
    if arrayOfEdgePoints:
        for point in arrayOfEdgePoints:
            outerEdges = [p for p in point.connectedWith if p.group != point.group]
            potentialFirewalls.append({'point': point, 'outerEdges': outerEdges})

    return potentialFirewalls


def findFirewalls(arrayOfSubsets,edgeMatrix):
    actualFirewalls = []
    potentialFirewalls = sorted(findPotentialFireWalls(arrayOfSubsets), key=lambda x: len(x['outerEdges']), reverse=True)
    # for p in potentialFirewalls:
    #     print(len(p['outerEdges']))

    
    if potentialFirewalls:
        for potential in potentialFirewalls:
            for potential2 in potentialFirewalls:
                if potential['point'] in potential2['outerEdges']:
                    potential2['outerEdges'].remove(potential['point'])

            if(potential['outerEdges']):
                actualFirewalls.append(potential['point'])
                edgeMatrix[potential['point'].index] = [2 if item == 1 else item for item in edgeMatrix[potential['point'].index]]
            potentialFirewalls = sorted(potentialFirewalls, key=lambda x: len(x['outerEdges']), reverse=True)

    
    return actualFirewalls

    


