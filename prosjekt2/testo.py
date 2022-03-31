from disjoint_set import DisjointSetForest
from sim_worlds.hex import Hex
from plotting import plot_board

hex = Hex(4, 4)

ds = DisjointSetForest()

ac = hex.get_legal_actions()

hex.play_action((0, 0))
hex.play_action((0, 3))
hex.play_action((3, 1))
#hex.play_action((0, 2))
hex.play_action((3, 0))
hex.play_action((1, 1))
print(hex.forest.forest)
#print(hex.forest.is_connected(hex.forest.forest[(0,0)], hex.forest.forest[(2, 2)]))
#hex.play_action((1, 1))
#print(hex.forest.is_connected(hex.forest.forest[(3,1)], hex.forest.forest[(2, 2)]))

plot_board(hex)
print(hex.channels())
