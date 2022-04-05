from tensorflow import keras

import nn
import plotting
from game_managers.hex_manager import HexManager
from sim_worlds.hex import Hex
import glob
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def run_series(model_l, model_r, num_games):
    win_count_1 = 0
    win_count_2 = 0
    manager = HexManager(7)
    for _ in range(num_games):
        state = manager.generate_initial_state()
        while not manager.is_state_final(state):
            nn_move = model_l.get_action(state, manager)
            manager.play_action(nn_move, state, True)
            if manager.is_state_final(state):
                win_count_1 += 1
                break
            nn_2_move = model_r.get_action(state, manager)
            manager.play_action(nn_2_move, state, True)
            if manager.is_state_final(state):
                win_count_2 += 1
                break

    return win_count_1, win_count_2


def topp(models: list, num_games: int):
    # Every model plays two series of games against every other. One as player 1 and one as player 2
    while models:
        model_1 = models.pop()
        for model_2 in models:
            win_1 = run_series(model_1, model_2, num_games)
            win_2 = run_series(model_2, model_1, num_games)

model_1 = keras.models.load_model('models/model_7_149')
model_2 = keras.models.load_model('models/model_7_100_old')
BOARD_SIZE = 7
models = glob.glob(f"models/model_{BOARD_SIZE}_*")
print(models)
lite_1 = nn.LiteModel.from_keras_model(model_1)
lite_2 = nn.LiteModel.from_keras_model(model_2)

lite_1.epsilon = 0.05
lite_2.epsilon = 0.05
models = [lite_1, lite_2]
print(run_series(lite_1, lite_2, 500))
print(run_series(lite_2, lite_1, 500))



