from pointtest import* 
from group import* 
from copy import copy, deepcopy

s = [Point()]
d= s.copy()
s[0].group = Group(1)

d.remove(s[0])

