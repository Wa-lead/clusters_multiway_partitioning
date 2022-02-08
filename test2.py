
import math
index = 0
def binarySearch(list, x):
    low = 0
    high = len(list) -1 

    while low<= high:
        mid = math.floor((low+high)/2)
        if list[mid] == x:
             return mid
        elif x < list[mid]:
            high = mid - 1
        else:
            low = mid + 1

        
print(binarySearch([1,2,3,6,8,9,11,20], 11))