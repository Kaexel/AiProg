from tensorflow import keras

import gui
import nn
import plotting
from game_managers.hex_manager import HexManager
from sim_worlds.hex import Hex
import glob
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def run_series(model_l, model_r, num_games, manager):
    win_count_1 = 0
    win_count_2 = 0
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


def topp(models: list, num_games: int, manager):
    # Every model plays two series of games against every other. One as player 1 and one as player 2
    tourneys = {}
    while models:
        model_1 = models.pop()
        for model_2 in models:
            win_1 = run_series(model_1[1], model_2[1], num_games, manager)
            win_2 = run_series(model_2[1], model_1[1], num_games, manager)
            tourneys[f"{model_1[0]}-{model_2[0]}"] = win_1
            tourneys[f"{model_2[0]}-{model_1[0]}"] = win_2
    return tourneys


class Tournament:
    def __init__(self, num_games):
        self.num_games = num_games
        self.manager = HexManager(5)

    def load_models(self, ):
        models = glob.glob(f"models\\model_{BOARD_SIZE}_*")
        loaded_models = [nn.LiteModel.from_keras_model(keras.models.load_model(model)) for model in models]
        models = [model.split("\\")[-1] for model in models]
        models_list = list(zip(models, loaded_models))
        return models_list

    def run_tourney(self):
        pass

    def run_series(self, model_l, model_r):
        win_count_1 = 0
        win_count_2 = 0
        for _ in range(self.num_games):
            state = self.manager.generate_initial_state()
            while not self.manager.is_state_final(state):
                nn_move = self.get_move(state, self.manager)

                self.manager.play_action(nn_move, state, True)
                if self.manager.is_state_final(state):
                    win_count_1 += 1
                    break
                nn_2_move = model_r.get_action(state, self.manager)
                self.manager.play_action(nn_2_move, state, True)
                if self.manager.is_state_final(state):
                    win_count_2 += 1
                    break

        return win_count_1, win_count_2

    def get_move(self, model, state):
        return model.get_action(state, self.manager)


BOARD_SIZE = 5
#models = glob.glob(f"models\\fullconv_5\\model_{BOARD_SIZE}_*")
models = glob.glob(f"models\\model_{BOARD_SIZE}_*")
#models.append("models\\model_5_10")
#models = ["models\\model_7_50", "models\\model_7_49"]
#models = ["models\\model_5_50", "models\\model_5_150"]
#print(models)
loaded_models = [nn.LiteModel.from_keras_model(keras.models.load_model(model)) for model in models]
models = [model.split("\\")[-1]for model in models]
models_list = list(zip(models, loaded_models))

wins = topp(models_list, 100, HexManager(BOARD_SIZE))
print(wins)


#lite_1 = nn.LiteModel.from_keras_model(model_1)
#lite_2 = nn.LiteModel.from_keras_model(model_2)

#lite_1.epsilon = 1
#lite_2.epsilon = 1
#models = [lite_1, lite_2]
#print(run_series(lite_1, lite_2, 500))
#print(run_series(lite_2, lite_1, 500))



