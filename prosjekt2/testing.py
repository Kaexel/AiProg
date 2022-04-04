import numpy as np
from tensorflow import keras

from nn import LiteModel
from game_managers.hex_manager import HexManager, HexState
from sim_worlds.sim_world import Players

mgr = HexManager(4)

board = np.array([
    [1,-1,1,1],
    [0,1,1,-1],
    [1,0,1,1],
    [-1,-1,-1,-1]])
state = HexState(board, player_turn=Players.BLACK)
state2 = HexState(board, player_turn=Players.WHITE)
model = LiteModel.from_keras_model(keras.models.load_model('models/model_200'))

print(state.board)
print(mgr.nn_state_representation(state))
print(mgr.nn_state_representation(state2))
model.get_action(state, mgr)
q = mgr.is_state_final(state)
print(q)
