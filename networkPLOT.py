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
lambda0=2; #intensity (ie mean density) of the Poisson process
 
#Simulate Poisson point process
numbPoints = (scipy.stats.poisson( lambda0*areaTotal ).rvs())#Poisson number of points
x = xDelta*scipy.stats.uniform.rvs(0,1,((numbPoints,1)))+xMin#x coordinates of Poisson points
y = yDelta*scipy.stats.uniform.rvs(0,1,((numbPoints,1)))+yMin#y coordinates of Poisson points

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                            #Preparing The data point data base

#create dictionary - associative array  - that holds all points as objects, the infected attribute is random (1= infected, 0= healthy)
for i in range(numbPoints):
    array_database.append({ "x":x[i][0], "y":y[i][0], "infected": 1 if random.randint(0, 9)>8 else 0}) 

#sorts the array based on the infected attribute, infected points come first.
array_database= sorted(array_database, key=lambda x:x['infected'], reverse=True)

#iterated through the array and make every infected node "infects" its neighbors 
for index_first_point in range(numbPoints):#x
    pointEdges = [0]*numbPoints
    for index_second_point in range(numbPoints):
        point1= array_database[index_first_point]
        point2= array_database[index_second_point]
        distance = sqrt((point1['x']-point2['x'])**2+(point1['y']-point2['y'])**2)
        if distance <= 1: # the initial assumption is that only nodes within 1 kilometers of each other can connect
            pointEdges[index_second_point] = 1
            if point1['infected']==1 or point2['infected']==1:
                point1['infected'] = 1
                point2['infected'] = 1
        array_database= sorted(array_database, key=lambda x:x['infected'], reverse=True)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                            #Prepare the edgeMatrix
                                                            #Why? because we want to assign cost to each edge
edgeMatrix = []

for index_first_point in range(numbPoints):#x
    pointEdges = [0]*numbPoints
    for index_second_point in range(numbPoints):
        point1= array_database[index_first_point]
        point2= array_database[index_second_point]
        distance = sqrt((point1['x']-point2['x'])**2+(point1['y']-point2['y'])**2)
        if distance <= 1: # the initial assumption is that only nodes within 1 kilometers of each other can connect
            pointEdges[index_second_point] = 1
    edgeMatrix.append(pointEdges)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                            #Plotting the points on graph

#plots the points on the graph, if the point is healthy make it green, else red
for index in range(numbPoints):
    color='black'
    if array_database[index]['x']>12 or array_database[index]['x']<2 or array_database[index]['y']>12 or array_database[index]['y']<2:
        color='grey'
    elif array_database[index]['infected']==0:
        color='green'
    else:
        color='red'

    plt.scatter(array_database[index]['x'],array_database[index]['y'], c= color)
    

#iterates through the the array and connects every node within distance 1 kilometer
for index_first_point in range(numbPoints):
    for index_second_point in range(numbPoints):
        point1= array_database[index_first_point]
        point2= array_database[index_second_point]
        if edgeMatrix[index_first_point][index_second_point] == 1: # the initial assumption is that only nodes within 1 kilometers of each other can connect
            pointX = [point1['x'], point2['x']]
            pointY = [point1['y'], point2['y']]
            plt.plot(pointX, pointY)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                            #Drawing the boundries of the range
plt.plot([2,2], [2,12],color='black')
plt.plot([2,12], [12,12],color='black')
plt.plot([12,12], [12,2],color='black')
plt.plot([12,2], [2,2], color='black')

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------


plt.show()

