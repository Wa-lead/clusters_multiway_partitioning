from copy import copy, deepcopy


class Group():
    def __init__(self, name):
        self.name = name
        self.points = []
        self.connectedGroup = {}
        self.outerPoints = []

    def add(self, point):
        self.points.append(point)

    def __repr__(self):
            return f'{self.name}: {self.points}'

    """
    this function itertates through all the points in this group and detects points that links it with other groups and stores it in (outerPoints)
    also, it adds it to another array of Group (connectedGroup) to optimize the process of multi-way partioning -- to make only connected group do the two way partioning
    One great feature is that is stores the connectedGroup as a dictionary where the key is the group name and the value is an array of all points that connects this group to connectedGroup
    this way, it allows dynamic identification of connected groups without the need to run this algorithm after each iteration of the two-way partioning
    """

    # def findConnectedGroup(self):
    #     for point1 in self.points:
    #         for point2 in point1.connectedWith:
    #             if point2.group != self:
    #                 if not (point1 in self.outerPoints):
    #                     self.outerPoints.append(point1)
    #                 self.connectedGroup[f'{point2.group.name}'] = [point1]

    def findConnectedGroup(self):
        for point1 in self.points:
            for group in point1.outerGroups:
                self.connectedGroup[f'{group.name}'] =  self.connectedGroup[f'{group.name}'] + [point1] if f'{group.name}' in self.connectedGroup.keys() else [point1]
                self.outerPoints.append(point1)

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

    def removeOuterPoint(self,point):
        self.outerPoints.remove(point)
        self.points.remove(point)

        """
        This part checks if
        """
        if point.group.name in self.connectedGroup.keys():
            removedPointGroup = self.connectedGroup[f'{point.group.name}']
            removedPointGroup.remove(point)
            if len(removedPointGroup) == 0:
                self.connectedGroup.pop(f'{point.group.name}')

    # def appendOuterPoint(self,point):
    #     self.outerPoints.append(point)
    #     self.points.append(point)
    #     point.group = self
    #     for point2 in point.connectedWith:
    #         if point2.group != point.group:
    #             self.connectedGroup[f'{point2.group.name}'] = self.connectedGroup[f'{point2.group.name}'] + [point]

    
    def appendOuterPoint(self,point):
        self.outerPoints.append(point)
        self.points.append(point)
        point.group = self
        point.updateOuterGroups()
        for group in point.outerGroups:
            self.connectedGroup[f'{group.name}'] = self.connectedGroup[f'{group.name}'] + [point] if f'{group.name}' in self.connectedGroup.keys() else [point]

    