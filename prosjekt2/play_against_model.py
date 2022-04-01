from tensorflow import keras

import nn
import plotting
from sim_worlds.hex import Hex

model = keras.models.load_model('models')

lite = nn.LiteModel.from_keras_model(model)
sim_world = Hex(5, 5)

while not sim_world.is_current_state_final():

    plotting.plot_board(sim_world)
    actions = sim_world.get_legal_actions()
    user_input = None
    while user_input not in actions:
        a = input("move!")
        user_input = tuple(int(x) for x in a.split(","))
    sim_world.play_action(user_input, True)
    plotting.plot_board(sim_world)

    nn_move = lite.get_action(sim_world)
    sim_world.play_action(nn_move, True)

