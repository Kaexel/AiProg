from disjoint_set import DisjointSetForest
from hex import Hex

hex = Hex(4, 4)

ds = DisjointSetForest()

ac = hex.get_legal_actions()

hex.play_action((0, 0))
hex.play_action((2, 2))
hex.play_action((2, 1))
print(hex.forest.forest)
print(hex.forest.is_connected(hex.forest.forest[(0,0)], hex.forest.forest[(2, 2)]))
hex.play_action((1, 1))
print(hex.forest.is_connected(hex.forest.forest[(0,0)], hex.forest.forest[(2, 2)]))
hex.print_board()
print(hex.forest.forest)
