from copy import copy, deepcopy

class Group():
    def __init__(self, name):
        self.name = name
        self.points = set()
        self.connectedGroup = set()
        self.outerPoints = set()
    
    def add(self, point):
        self.points.add(point)


    def __repr__(self):
            return f'{self.name}: {self.points}'
    
    def find_connected_groups(self):
        self.connectedGroup = set()
        self.outerPoints = set()
        for point1 in self.points:
            for point2 in point1.connected_points:
                if point2.group != self.name:
                        self.connectedGroup.add(point2.group)
                        self.outerPoints.add(point1)


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
        self.outerPoints.add(point)
        self.points.add(point)
        point.group = self.name