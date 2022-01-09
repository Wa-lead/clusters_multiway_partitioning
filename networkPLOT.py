import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

numberOfNodes = int(input("Enter the number of nodes: "))
x = np.random.rand(numberOfNodes)*10 # ( 10 ) represnets the range of the scattering
y = np.random.rand(numberOfNodes)*10

plt.scatter(x, y)

for index_first_point in range(numberOfNodes):
    for index_second_point in range(numberOfNodes):
        distance = sqrt((x[index_first_point]-x[index_second_point])**2+(y[index_first_point]-y[index_second_point])**2)
        if distance <= 1: # the initial assumption is that only nodes within 1 kilometers of each other can connect
            pointX = [x[index_first_point], x[index_second_point]]
            pointY = [y[index_first_point], y[index_second_point]]
            plt.plot(pointX, pointY)
plt.show()


