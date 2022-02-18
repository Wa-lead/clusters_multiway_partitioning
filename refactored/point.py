class Point():
    def __init__(self, group, index, x, y, infected, Dvalue):
        self.group = group
        self.index = index
        self.x = x
        self.y = y
        self.infected = infected
        self.Dvalue = Dvalue
        self.connectedWith = []
    
    def connect(self, point): 
        self.connectedWith.append(point)
        point.connectedWith.append(self)

    def __repr__(self):
            return f'group: {self.group}, index: {self.index},x: {self.x},y: {self.y},infected? {self.infected}, Dvalue: {self.Dvalue}'