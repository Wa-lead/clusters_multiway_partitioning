from dis import dis
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import random
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from k_means_constrained import KMeansConstrained

def twoWayPartitioning(A,B,edgeMatrix,groups):
    subsetA = A['subset']
    subsetB = B['subset']
    groupA = A['group']
    groupB = B['group']
    indicator = 0
    while True:
        for index1 in range(len(subsetA)):
            ineternalCost = 0
            externalCost = 0
            for index2 in range(len(subsetA)): ## length of both subsets are the same
                ineternalCost += edgeMatrix[subsetA[index1]['index']][subsetA[index2]['index']]
                externalCost += edgeMatrix[subsetA[index1]['index']][subsetB[index2]['index']]
            subsetA[index1]['Dvalue'] = externalCost - ineternalCost
            

        for index1 in range(len(subsetB)):
            ineternalCost = 0
            externalCost =0
            for index2 in range(len(subsetB)): ## length of both subsets are the same
                ineternalCost += edgeMatrix[subsetB[index1]['index']][subsetB[index2]['index']]
                externalCost += edgeMatrix[subsetB[index1]['index']][subsetA[index2]['index']]
            subsetB[index1]['Dvalue'] = externalCost - ineternalCost

        subsetA= sorted(subsetA, key=lambda x:x['Dvalue'], reverse=True)
        subsetB= sorted(subsetB, key=lambda x:x['Dvalue'], reverse=True)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        G = 0

        Xstar = []
        Ystar = []
        GBuffer = []

        #copies are used to maintain the state of the original sets
        subsetACopy = subsetA.copy()
        subsetBCopy = subsetB.copy()

        for i in range(len(subsetACopy)):
            gain = -10000
            indexOfa1 = 0
            indexOfb1 = 0
            #find the highest gain
            for j in range(len(subsetACopy)):
                for k in range(len(subsetACopy)):
                    newGain = subsetACopy[j]['Dvalue'] + subsetBCopy[k]['Dvalue'] - 2*edgeMatrix[subsetACopy[j]['index']][subsetBCopy[k]['index']]
                    if newGain > gain:
                        gain = newGain
                        indexOfa1 = j
                        indexOfb1 = k
            
            G+= gain
        
            #Recalculating the Dvalue of each in the session
            for j in range(len(subsetACopy)):
                subsetACopy[j]['Dvalue'] = subsetACopy[j]['Dvalue'] +  2*edgeMatrix[subsetACopy[j]['index']][subsetACopy[indexOfa1]['index']] - 2*edgeMatrix[subsetACopy[j]['index']][subsetBCopy[indexOfb1]['index']]
                subsetBCopy[j]['Dvalue'] = subsetBCopy[j]['Dvalue'] +  2*edgeMatrix[subsetBCopy[j]['index']][subsetBCopy[indexOfb1]['index']] - 2*edgeMatrix[subsetBCopy[j]['index']][subsetACopy[indexOfa1]['index']]
            
            ##------------------------------------------- the mistake is here, fix the popping problem


            #remove the highest gain points from the set
            Xstar.append(subsetACopy.pop(indexOfa1))
            Ystar.append(subsetBCopy.pop(indexOfb1))


            #buffers the gain 
            GBuffer.append({"index" :i, "G": G, "Xstar": Xstar.copy(), "Ystar": Ystar.copy()})

            #sorting based on Dvalue again
            subsetACopy= sorted(subsetACopy, key=lambda x:x['Dvalue'], reverse=True)
            subsetBCopy= sorted(subsetBCopy, key=lambda x:x['Dvalue'], reverse=True)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #remove highest gain exchanges from the subsetA and subsetB
        GBuffer= sorted(GBuffer, key=lambda x:x['G'], reverse=True)

        if(GBuffer[0]['G']<=0.000000000001):
            break

        for point in GBuffer[0]['Xstar']:
            subsetA.remove(point)

        for point in GBuffer[0]['Ystar']:
            subsetB.remove(point)   

        #add the interchanged points to their new sets
        subsetA = subsetA + GBuffer[0]['Ystar']
        subsetB = subsetB + GBuffer[0]['Xstar']
        indicator = indicator + 1

        for i in range(len(subsetA)): # update the group array
            groups[subsetA[i]['index']] = groupA
            groups[subsetB[i]['index']] = groupB


    return subsetA ,subsetB, indicator, groups


def divideIntoEvenClusters(x,y, numberOfClusters):
    numbPoints = len(x)
    changeFormatArray =[] 
    for i in range(len(x)): #here is merely changing the format of the array
        changeFormatArray.append([x[i],y[i]])
    X = np.array(changeFormatArray)
    kmeans = KMeansConstrained(n_clusters=numberOfClusters, size_min=numbPoints/numberOfClusters, size_max=numbPoints/numberOfClusters)
    kmeans.fit(X)
    return kmeans.labels_

