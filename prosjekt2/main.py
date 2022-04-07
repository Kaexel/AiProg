import ast
import cProfile
import configparser
import glob
import time
import torch

import gui
import nn
from game_managers.hex_manager import HexManager
from opmc import OnPolicyMonteCarlo
from sim_worlds.hex import Hex
from mcts import MCTS
from sim_worlds.nim import Nim
from sim_worlds.old_gold import OldGold
import random
import tensorflow as tf

random.seed(42)

AVAILABLE_ACTIVATIONS = ['relu', 'tanh', 'linear', 'sigmoid']
AVAILABLE_OPTIMIZERS = {'adam': tf.keras.optimizers.Adam, 'adagrad': tf.keras.optimizers.Adagrad, 'sgd': tf.keras.optimizers.SGD, 'rmsprop': tf.keras.optimizers.RMSprop}

"""
Setting up some primary variables
"""
available_sim_worlds = {0: Nim, 1: OldGold, 2: Hex}
config = configparser.ConfigParser()
config.read('config.ini')

num_actual = config["PRIMARY"].getint('NUM_ACTUAL_GAMES')
num_rollout = config["PRIMARY"].getint('MCTS_NUM_ROLLOUTS')
rbuf_len = config["PRIMARY"].getint('RBUF_LENGTH')
rbuf_num_sample = config["PRIMARY"].getint('RBUF_NUM_SAMPLE')
board_k = config["HEX"].getint('BOARD_K')
sim_type = config["PRIMARY"].getint('SIM_WORLD_TYPE')

learning_rate = config["ANET"].getfloat('LEARNING_RATE')
conv_layers = ast.literal_eval(config["ANET"]['CONV_LAYERS'])
dense_layers = ast.literal_eval(config["ANET"]['DENSE_LAYERS'])
activation_function = config["ANET"]["ACTIVATION_FUNCTION"]
assert activation_function in AVAILABLE_ACTIVATIONS
optimizer_string = config["ANET"]["OPTIMIZER"]
assert optimizer_string in AVAILABLE_OPTIMIZERS
optimizer = AVAILABLE_OPTIMIZERS[optimizer_string](learning_rate=learning_rate)
num_cached_nets = config["ANET"].getint("NUM_CACHED_NETS")
num_game_tournament = config["ANET"].getint("NUM_GAME_TOURNAMENT")

interval_save = num_actual // num_cached_nets if num_cached_nets > 0 else num_actual + 1


"""Below used for selecting sim world. No longer relevant"""
#params = {}
#for k, v in config[str(available_sim_worlds[sim_type].__name__).upper()].items():
#    params[k] = int(v)

#assert (sim_type in range(0, len(available_sim_worlds)))
#sim_world = available_sim_worlds[sim_type](**params)
#nim = OldGold(8)

# Instantiate On-Policy Monte Carlo object with config params and run games
#model = nn.make_keras_model(filters=conv_layers, dense=dense_layers, rows=board_k, cols=board_k, activation_function=activation_function, optimizer=optimizer)
#opmc = OnPolicyMonteCarlo(mgr=HexManager(board_k), i_s=interval_save, actual_games=num_actual, search_games=num_rollout, model=model, max_rbuf=rbuf_len, sample_rbuf=rbuf_num_sample, gui=gui.GameGUI())
#opmc.run_games()
models = glob.glob(f"models\\model_{board_k}_*")
t_gui = gui.TournamentGUI(num_game_tournament, models)
t_gui.root.mainloop()
