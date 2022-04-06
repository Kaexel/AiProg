import copy

"""
A solution used to determine very quickly whether a game is over.
Replaced by DFS through a np array due to deecopy() being slow.
"""

class HexSetUnit:
    def __init__(self, value: tuple, parent=None):
        self.value = value
        self.parent = parent

    def __str__(self):
        return f"Value: {self.value}, Parent:  {self.parent}"

    def __repr__(self):
        return f"Value: {self.value}, Parent:  {self.parent.value}\n"

    def __copy__(self):
        return HexSetUnit(self.value, copy.copy(self.parent))


class DisjointSetForest:
    """
    Works very quickly to determine whether game is over. Agonizingly slow to copy in python however. Went over to use DFS on a numpy board instead.
    """
    def __init__(self):
        self.forest = {}  # Map of array index tuples to sets

    def make_set(self, coords):
        if coords not in self.forest.keys():
            x = HexSetUnit(coords)
            x.parent = x
            x.size = 1
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

    def clone(self):
        dsf = DisjointSetForest()
        dsf.forest = {val: copy.deepcopy(hsu) for val, hsu in self.forest.items()}
        return dsf


