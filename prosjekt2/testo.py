import ast
import cProfile
import time

import gui
import nn
from gui import GameGUI
from mcts import MCTS
from opmc import OnPolicyMonteCarlo
import configparser
import tensorflow as tf
from game_managers.hex_manager import HexManager
AVAILABLE_ACTIVATIONS = ['relu', 'tanh', 'linear', 'sigmoid']
AVAILABLE_OPTIMIZERS = {'adam': tf.keras.optimizers.Adam, 'adagrad': tf.keras.optimizers.Adagrad, 'sgd': tf.keras.optimizers.SGD, 'rmsprop': tf.keras.optimizers.RMSprop}

"""
Setting up some primary variables
"""
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


model = nn.make_keras_model(filters=conv_layers, dense=dense_layers, rows=board_k, cols=board_k, activation_function=activation_function, optimizer=optimizer)

testo = MCTS(game_manager=HexManager(5), policy_object=nn.LiteModel.from_keras_model(model))
t = time.time()

pr = cProfile.Profile()
pr.enable()
momo = testo.search(1000)
pr.disable()
pr.print_stats()

print(f"{time.time() - t} seconds")
exit()

pr = cProfile.Profile()
pr.enable()
optimizer = tf.keras.optimizers.Adam()
model = nn.make_keras_model(filters=(64, 64), dense=(64,), rows=5, cols=5, activation_function='relu', optimizer=optimizer)
opmc = OnPolicyMonteCarlo(mgr=HexManager(5), i_s=50, actual_games=2, search_games=1000, model=model, max_rbuf=1000, sample_rbuf=160, gui=gui.GameGUI())
opmc.run_games()
pr.disable()
pr.print_stats()
