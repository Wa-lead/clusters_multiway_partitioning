from copy import copy, deepcopy

class Point():
    def __init__(self, group, index, x, y, infected, Dvalue):
        self.group = group
        group.points.append(self)
        self.index = index
        self.x = x
        self.y = y
        self.infected = infected
        self.Dvalue = Dvalue
        self.connectedWith = []
        self.outerGroups = [] ## change name later
    
    def connect(self, point): 
        self.connectedWith.append(point)
        point.connectedWith.append(self)
        if(self.group != point.group):
            self.outerGroups.append(point.group)
            point.outerGroups.append(self.group)


    # def __repr__(self):
    #         return f'group: {self.group}, index: {self.index},x: {self.x},y: {self.y},infected? {self.infected}, Dvalue: {self.Dvalue}'

    def __repr__(self):
            return f'{self.index}\n' 

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result
    
    def updateOuterGroups(self):
        self.outerGroups = []
        for point in self.connectedWith:
            if point.group != self.group:
                self.outerGroups.append(point.group)