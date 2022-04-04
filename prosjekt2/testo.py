import cProfile

from opmc import OnPolicyMonteCarlo
import configparser

from game_managers.hex_manager import HexManager

config = configparser.ConfigParser()
config.read('config.ini')
num_actual = config["PRIMARY"].getint('NUM_ACTUAL_GAMES')
num_rollout = config["PRIMARY"].getint('MCTS_NUM_ROLLOUTS')
board_k = config["HEX"].getint('BOARD_K')
pr = cProfile.Profile()

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
opmc = OnPolicyMonteCarlo(mgr=HexManager(board_k), i_s=50, actual_games=num_actual, search_games=num_rollout)
opmc.run_games()
#pr.disable()
#pr.print_stats()
