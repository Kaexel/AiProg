import numpy as np
from tensorflow import keras

import nn
import plotting
from sim_worlds.hex import Hex

model = keras.models.load_model('models/model_50')

testo = nn.make_keras_model(32, 5, 5)
lite = nn.LiteModel.from_keras_model(model)
sim_world = Hex.initialize_state(5, 5)
plotting.plot_board(sim_world)
while not sim_world.is_current_state_final():

    nn_move = lite.get_action(sim_world)
    sim_world.play_action(nn_move, True)
    plotting.plot_board(sim_world)
    print(sim_world.board)
    print(np.rot90(sim_world.board))
    print(sim_world.board)

    actions = sim_world.get_legal_actions()
    user_input = None
    while user_input not in actions:
        a = input("move!")
        user_input = tuple(int(x) for x in a.split(","))
    sim_world.play_action(user_input, True)
    plotting.plot_board(sim_world)

