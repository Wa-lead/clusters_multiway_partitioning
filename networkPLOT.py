import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import random

array_database = []
numberOfNodes = int(input("Enter the number of nodes: "))

#creates multiple random points in "np" format, then I convert them to normal list form.
x = (np.random.rand(numberOfNodes)*10).tolist() # ( 10 ) represnets the range of the scattering
y = (np.random.rand(numberOfNodes)*10).tolist() 

#create dictionary - associative array  - that holds all points as objects, the infected attribute is random (1= infected, 0= healthy)
for i in range(numberOfNodes):
    array_database.append({ "x":x[i], "y":y[i], "infected": 1 if random.randint(0, 9)>7 else 0}) 

#sorts the array based on the infected attribute, infected points come first.
array_database= sorted(array_database, key=lambda x:x['infected'], reverse=True)

#iterated through the array and make every infected node "infects" its neighbors 
for index_first_point in range(numberOfNodes):
    for index_second_point in range(numberOfNodes):
        point1= array_database[index_first_point]
        point2= array_database[index_second_point]
        distance = sqrt((point1['x']-point2['x'])**2+(point1['y']-point2['y'])**2)
        if distance <= 1: # the initial assumption is that only nodes within 1 kilometers of each other can connect
            if point1['infected']==1 or point2['infected']==1:
                point1['infected'] = 1
                point2['infected'] = 1


#plots the points on the graph, if the point is healthy make it green, else red
for index in range(numberOfNodes):
    plt.scatter(array_database[index]['x'],array_database[index]['y'], c='green' if array_database[index]['infected']==0 else 'red')

#iterates through the the array and connects every node within distance 1 kilometer
for index_first_point in range(numberOfNodes):
    for index_second_point in range(numberOfNodes):
        point1= array_database[index_first_point]
        point2= array_database[index_second_point]
        distance = sqrt((point1['x']-point2['x'])**2+(point1['y']-point2['y'])**2)
        if distance <= 1: # the initial assumption is that only nodes within 1 kilometers of each other can connect
            pointX = [point1['x'], point2['x']]
            pointY = [point1['y'], point2['y']]
            plt.plot(pointX, pointY)

plt.show()

