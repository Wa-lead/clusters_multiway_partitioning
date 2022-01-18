from dis import dis
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import random
#creates multiple random points in "np" format, then I convert them to normal list form.
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


array_database = []

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                            #Preparing the poisson distribution pibts
#Simulation window parameters
xMin=0;xMax=14
yMin=0;yMax=14
xDelta=xMax-xMin
yDelta=yMax-yMin; #rectangle dimensions
areaTotal=xDelta*yDelta
 
#Point process parameters
lambda0=0.4; #intensity (ie mean density) of the Poisson process
 
#Simulate Poisson point process
numbPoints = (scipy.stats.poisson( lambda0*areaTotal ).rvs())*2#Poisson number of points # multiply by 2 to make it always even
x = xDelta*scipy.stats.uniform.rvs(0,1,((numbPoints,1)))+xMin#x coordinates of Poisson points
y = yDelta*scipy.stats.uniform.rvs(0,1,((numbPoints,1)))+yMin#y coordinates of Poisson points

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                            #Preparing The data point data base

#create dictionary - associative array  - that holds all points as objects, the infected attribute is random (1= infected, 0= healthy)

subsetA = []
subsetB = []

for i in range(numbPoints):
    array_database.append({ "index": i, "x":x[i][0], "y":y[i][0], "infected": 1 if random.randint(0, 9)>8 else 0, "Dvalue": 0})
    if(i%2 == 0):
        subsetA.append(array_database[i])
    else:
        subsetB.append(array_database[i])
 

#sorts the array based on the infected attribute, infected points come first.
array_database= sorted(array_database, key=lambda x:x['infected'], reverse=True)

#iterated through the array and make every infected node "infects" its neighbors 
for index_first_point in range(numbPoints):#x
    for index_second_point in range(numbPoints):
        point1= array_database[index_first_point]
        point2= array_database[index_second_point]
        distance = sqrt((point1['x']-point2['x'])**2+(point1['y']-point2['y'])**2)
        if distance <= 1: # the initial assumption is that only nodes within 1 kilometers of each other can connect
            if point1['infected']==1 or point2['infected']==1:
                point1['infected'] = 1
                point2['infected'] = 1
        array_database= sorted(array_database, key=lambda x:x['infected'], reverse=True)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                            #Prepare the edgeMatrix
                                                            #Why? because we want to assign cost to each edge

array_database= sorted(array_database, key=lambda x:x['index']) ## sort it based on index to uilize the edge matrix
edgeMatrix = []

for index_first_point in range(numbPoints):#x
    pointEdges = [0]*numbPoints
    for index_second_point in range(numbPoints):
        point1= array_database[index_first_point]
        point2= array_database[index_second_point]
        distance = sqrt((point1['x']-point2['x'])**2+(point1['y']-point2['y'])**2)
        pointEdges[index_second_point] = 0 if index_second_point == index_first_point else distance
    edgeMatrix.append(pointEdges)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                            #Calculating the the D value
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
G = 0

Xstar = []
Ystar = []
for i in range(len(subsetA)):
    gain = -10000
    indexOfa1 = 0
    indexOfb1 = 0
    for o in range(len(subsetA)):
        for k in range(len(subsetA)): #Finding the maximum gain resulted from exchange
            newGain=subsetA[o]['Dvalue'] + subsetB[k]['Dvalue'] - 2*edgeMatrix[subsetA[o]['index']][subsetB[k]['index']]
            if newGain > gain:
                gain = newGain
                indexOfb1 = k
                indexOfa1 = o
    G+= gain    
    for j in range(len(subsetA)):#Recalculating the Dvalue of each in the session
        subsetA[j]['Dvalue'] = subsetA[j]['Dvalue'] +  2*edgeMatrix[subsetA[j]['index']][subsetA[indexOfa1]['index']] - 2*edgeMatrix[subsetA[j]['index']][subsetB[indexOfb1]['index']]
        subsetB[j]['Dvalue'] = subsetB[j]['Dvalue'] +  2*edgeMatrix[subsetB[j]['index']][subsetB[indexOfb1]['index']] - 2*edgeMatrix[subsetB[j]['index']][subsetA[indexOfa1]['index']]
    #excluding the interchanged values
    Xstar.append(subsetA.pop(indexOfa1))
    Ystar.append(subsetB.pop(indexOfb1))

    #sorting based on Dvalue again
    subsetA= sorted(subsetA, key=lambda x:x['Dvalue'], reverse=True)
    subsetB= sorted(subsetB, key=lambda x:x['Dvalue'], reverse=True)

subsetA.extend(Ystar)
subsetB.extend(Xstar)

print (int(G))
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                            #Plotting the points on graph

#plots the points on the graph, if the point is healthy make it green, else red
for index in range(numbPoints):
    color='black'
    if array_database[index]['x']>12 or array_database[index]['x']<2 or array_database[index]['y']>12 or array_database[index]['y']<2:
        color='grey'
    elif array_database[index] in subsetA:
        color='green'
    else:
        color='purple'

    plt.scatter(array_database[index]['x'],array_database[index]['y'], c= color)



#iterates through the the array and connects every node within distance 1 kilometer
for index_first_point in range(numbPoints):
    for index_second_point in range(numbPoints):
        point1= array_database[index_first_point]
        point2= array_database[index_second_point]
        if edgeMatrix[index_first_point][index_second_point] <= 1: # the initial assumption is that only nodes within 1 kilometers of each other can connect
            pointX = [point1['x'], point2['x']]
            pointY = [point1['y'], point2['y']]
            plt.plot(pointX, pointY, color='black')

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                            #Drawing the boundries of the range
plt.plot([2,2], [2,12],color='black')
plt.plot([2,12], [12,12],color='black')
plt.plot([12,12], [12,2],color='black')
plt.plot([12,2], [2,2], color='black')

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------


plt.show()

