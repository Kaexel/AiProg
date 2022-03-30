class SetUnit:
    def __init__(self, value: tuple):
        self.value = value
        self.parent = None
        self.size = 0
        self.rank = 0

    def __str__(self):
        return f"Value: {self.value}, Parent:  {self.parent}"

    def __repr__(self):
        return f"Value: {self.value}, Parent:  {self.parent.value}"

class DisjointSetForest:
    def __init__(self):
        self.forest = {}  # Map of array index tuples to sets

    def make_set(self, coords):
        if coords not in self.forest.keys():
            x = SetUnit(coords)
            x.parent = x
            x.size = 1
            x.rank = 0
            self.forest[x.value] = x

    def find(self, x):  # Traverses set to root element
        if x.parent != x:
            x.parent = self.find(x.parent)
            return x.parent
        else:
            return x

    def union(self, x, y):  # Connects two sets
        x = self.find(x)
        y = self.find(y)

        if x == y:
            return  # Both elements in same set

        if x.size < y.size:
            x, y = y, x

        y.parent = x
        x.size = x.size + y.size

    def is_connected(self, x, y):  # Checks whether two elements are in same set (same parent)
        x = self.find(x)
        y = self.find(y)
        return x == y


