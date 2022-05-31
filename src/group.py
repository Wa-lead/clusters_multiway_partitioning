from copy import copy, deepcopy

class Group():
    def __init__(self, name):
        self.name = name
        self.points = []
        self.connectedGroup = []
        self.outerPoints = []
    
    def add(self, point):
        self.points.append(point)


    def __repr__(self):
            return f'{self.name}: {self.points}'
    
    def find_connected_groups(self):
        self.connectedGroup = []
        self.outerPoints = []
        for point1 in self.points:
            for point2 in point1.connected_points:
                if point2.group != self.name:
                    if not (point2.group  in self.connectedGroup):
                        self.connectedGroup.append(point2.group)
                    if not (point1 in self.outerPoints):
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

    def remove_outer_point(self,point):
        self.outerPoints.remove(point)
        self.points.remove(point)

    def append_outer_point(self,point):
        self.outerPoints.append(point)
        self.points.append(point)
        point.group = self.name