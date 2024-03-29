import numpy as np
from tensorflow import keras

import nn
import plotting
from game_managers.hex_manager import HexManager
"""
Helper functions to play against a trained net
"""


model = keras.models.load_model('models\\5_dense_double_empties\\model_5_50')
lite = nn.LiteModel.from_keras_model(model)
manager = HexManager(5)
state = manager.generate_initial_state()

lite.epsilon = 0


def start_first():
    plotting.plot_board(state)
    while 1:
        actions = manager.get_legal_actions(state)
        user_input = None
        while user_input not in actions:
            a = input("move!")
            user_input = tuple(int(x) for x in a.split(","))
        manager.play_action(user_input, state, True)
        plotting.plot_board(state)
        if manager.is_state_final(state):
            break

        nn_move = lite.get_action(state=state, manager=manager)
        manager.play_action(nn_move, state, True)
        plotting.plot_board(state)
        if manager.is_state_final(state):
            break


def start_second():
    plotting.plot_board(state)
    while 1:
        nn_move = lite.get_action(state=state, manager=manager)
        manager.play_action(nn_move, state, True)
        plotting.plot_board(state)
        if manager.is_state_final(state):
            break
        actions = manager.get_legal_actions(state)
        user_input = None
        while user_input not in actions:
            a = input("move!")
            user_input = tuple(int(x) for x in a.split(","))
        manager.play_action(user_input, state, True)
        plotting.plot_board(state)
        if manager.is_state_final(state):
            break


#start_first()
start_second()
