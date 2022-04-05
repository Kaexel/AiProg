import cProfile

import nn
from opmc import OnPolicyMonteCarlo
import configparser
import tensorflow as tf
from game_managers.hex_manager import HexManager

config = configparser.ConfigParser()
config.read('config.ini')
num_actual = config["PRIMARY"].getint('NUM_ACTUAL_GAMES')
num_rollout = config["PRIMARY"].getint('MCTS_NUM_ROLLOUTS')
board_k = config["HEX"].getint('BOARD_K')
#pr = cProfile.Profile()

#pr.enable()
#testo = MCTS()
#t = time.time()
#momo = testo.search(1000)
#pr.disable()
#pr.print_stats()
#print(f"{time.time() - t} seconds")
#exit()

#pr = cProfile.Profile()
#pr.enable()
optimizer = tf.keras.optimizers.Adam()
model = nn.make_keras_model(filters=(64, 64), dense=(64,), rows=5, cols=5, activation_function='relu', optimizer=optimizer)
opmc = OnPolicyMonteCarlo(mgr=HexManager(5), i_s=50, actual_games=2, search_games=1000, model=model, max_rbuf=1000, sample_rbuf=160)
opmc.run_games()
#pr.disable()
#pr.print_stats()
