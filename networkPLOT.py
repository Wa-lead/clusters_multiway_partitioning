from dis import dis
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import random
#creates multiple random points in "np" format, then I convert them to normal list form.
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


fig, ax = plt.subplots(2, figsize=(8, 14))

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
lambda0=1.5; #intensity (ie mean density) of the Poisson process
 
#Simulate Poisson point process
numbPoints = (scipy.stats.poisson( lambda0*areaTotal ).rvs())*2#Poisson number of points # multiply by 2 to make it always even
x = xDelta*scipy.stats.uniform.rvs(0,1,((numbPoints,1)))+xMin#x coordinates of Poisson points
y = yDelta*scipy.stats.uniform.rvs(0,1,((numbPoints,1)))+yMin#y coordinates of Poisson points

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                            #Preparing The data point data base

# create the two subsets A and B
subsetA = [] #A
subsetB = [] #B

#create dictionary - associative array  - that holds all points as objects, the infected attribute is random (1= infected, 0= healthy)
for i in range(numbPoints):
    array_database.append({ "index": i, "x":x[i][0], "y":y[i][0], "infected": 1 if random.randint(0, 9)>8 else 0, "Dvalue": 0})
    if(x[i][0]>=7 and len(subsetA)<numbPoints/2) or (len(subsetB) == numbPoints/2):
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
edgeMatrix = [[0 for i in range(len(array_database))] for j in range(len(array_database))]


for index_first_point in range(numbPoints):#x
    index_second_point = index_first_point +1
    for index_second_point in range(numbPoints):
        point1= array_database[index_first_point]
        point2= array_database[index_second_point]
        distance = sqrt((point1['x']-point2['x'])**2+(point1['y']-point2['y'])**2)
        edgeMatrix[index_first_point][index_second_point] = distance if distance <= 1 else 0
        edgeMatrix[index_second_point][index_first_point] = edgeMatrix[index_first_point][index_second_point]

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                            #Plot the girst graph before partioning
for index in range(numbPoints):
    color='black'
    if array_database[index]['x']>12 or array_database[index]['x']<2 or array_database[index]['y']>12 or array_database[index]['y']<2:
        color='grey'
    elif array_database[index] in subsetA:
        color='green'
    else:
        color='purple'

    ax[0].scatter(array_database[index]['x'],array_database[index]['y'], c= color)

for index_first_point in range(numbPoints):
    for index_second_point in range(numbPoints):
        point1= array_database[index_first_point]
        point2= array_database[index_second_point]
        if edgeMatrix[index_first_point][index_second_point] != 0: # the initial assumption is that only nodes within 1 kilometers of each other can connect
            pointX = [point1['x'], point2['x']]
            pointY = [point1['y'], point2['y']]
            isInSubsetA= point1 in subsetA
            ax[0].plot(pointX, pointY,'green' if isInSubsetA  else 'purple')

ax[0].plot([2,2], [2,12],color='black')
ax[0].plot([2,12], [12,12],color='black')
ax[0].plot([12,12], [12,2],color='black')
ax[0].plot([12,2], [2,2], color='black')

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                          #Calculating the the D value

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

    ax[1].scatter(array_database[index]['x'],array_database[index]['y'], c= color)



# iterates through the the array and connects every node within distance 1 kilometer
for index_first_point in range(numbPoints):
    for index_second_point in range(numbPoints):
        point1= array_database[index_first_point]
        point2= array_database[index_second_point]
        if edgeMatrix[index_first_point][index_second_point] != 0: # the initial assumption is that only nodes within 1 kilometers of each other can connect
            pointX = [point1['x'], point2['x']]
            pointY = [point1['y'], point2['y']]
            isInSubsetA= point1 in subsetA
            ax[1].plot(pointX, pointY,'green' if isInSubsetA  else 'purple')

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                            #Drawing the boundries of the range
ax[1].plot([2,2], [2,12],color='black')
ax[1].plot([2,12], [12,12],color='black')
ax[1].plot([12,12], [12,2],color='black')
ax[1].plot([12,2], [2,2], color='black')

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------


plt.show()

