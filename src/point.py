from copy import copy, deepcopy

class Point():
    def __init__(self, group, index, x, y, state, Dvalue):
        self.group = group
        self.index = index
        self.x = x
        self.y = y
        self.state = state
        self.Dvalue = Dvalue
        self.connected_points = []
    
    def connect(self, point): 
        self.connected_points.append(point)
        point.connected_points.append(self)

    def __repr__(self):
            return f'{(int(self.x),int(self.y))}'

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