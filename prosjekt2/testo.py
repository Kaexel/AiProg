import cProfile

import numpy as np

from disjoint_set import DisjointSetForest
from nn import make_keras_model, PolicyNetwork
from opmc import OnPolicyMonteCarlo
from sim_worlds.hex import Hex
from plotting import plot_board

import tensorflow as tf

hex = Hex(4, 4)

ds = DisjointSetForest()

ac = hex.get_legal_actions()

hex.play_action((0, 0))
hex.play_action((0, 3))
hex.play_action((3, 1))
#hex.play_action((0, 2))
hex.play_action((3, 0))
hex.play_action((1, 1))
#print(hex.forest.is_connected(hex.forest.forest[(0,0)], hex.forest.forest[(2, 2)]))
#hex.play_action((1, 1))
#print(hex.forest.is_connected(hex.forest.forest[(3,1)], hex.forest.forest[(2, 2)]))

opmc = OnPolicyMonteCarlo(Hex(5, 5))
#pr = cProfile.Profile()
#pr.enable()
opmc.run_games()
#pr.disable()
#pr.print_stats()
"""
q = hex.channels()
mike = make_keras_model(20, 4, 4)
p_net = PolicyNetwork(20, 4, 4)
print(q.shape)
y = p_net.get_action(hex)
print(y)
"""


plot_board(hex)
#print(hex.channels())
