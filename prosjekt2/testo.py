import cProfile

import numpy as np

from disjoint_set import DisjointSetForest
from nn import make_keras_model, PolicyNetwork
from opmc import OnPolicyMonteCarlo
from sim_worlds.hex import Hex
from plotting import plot_board
import configparser
import tensorflow as tf


config = configparser.ConfigParser()
config.read('config.ini')
num_actual = config["PRIMARY"].getint('NUM_ACTUAL_GAMES')
num_rollout = config["PRIMARY"].getint('MCTS_NUM_ROLLOUTS')
board_k = config["HEX"].getint('BOARD_K')

opmc = OnPolicyMonteCarlo(Hex.initialize_state(board_k, board_k), 50, actual_games=num_actual, search_games=num_rollout)
pr = cProfile.Profile()
pr.enable()
opmc.run_games()
pr.disable()
pr.print_stats()
"""
q = hex.channels()
mike = make_keras_model(20, 4, 4)
p_net = PolicyNetwork(20, 4, 4)
print(q.shape)
y = p_net.get_action(hex)
print(y)
"""

#print(hex.channels())
