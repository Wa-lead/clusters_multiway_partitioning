import scipy.stats
xMin=0;xMax=14
yMin=0;yMax=14
xDelta=xMax-xMin
yDelta=yMax-yMin; #rectangle dimensions
areaTotal=xDelta*yDelta
 
#Point process parameters
lambda0=0.5; #intensity (ie mean density) of the Poisson process
 
#Simulate Poisson point process
numbPoints = (scipy.stats.poisson( lambda0*areaTotal ).rvs())#Poisson number of points
print(numbPoints)