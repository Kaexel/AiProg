import numpy as np
from tensorflow import keras

import nn
import plotting
from sim_worlds.hex import Hex

sim_world = Hex.initialize_state(5, 5)
plotting.plot_board(sim_world)
while not sim_world.is_current_state_final():

    actions = sim_world.get_legal_actions()
    user_input = None
    while user_input not in actions:
        a = input("move!")
        user_input = tuple(int(x) for x in a.split(","))
    sim_world.play_action(user_input, True)
    plotting.plot_board(sim_world)

