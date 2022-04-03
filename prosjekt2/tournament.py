from tensorflow import keras

import nn
import plotting
from sim_worlds.hex import Hex

model_1 = keras.models.load_model('models/model_50')
model_2 = keras.models.load_model('models/model_200')
lite_1 = nn.LiteModel.from_keras_model(model_1)
lite_2 = nn.LiteModel.from_keras_model(model_2)
sim_world = Hex.initialize_state(5, 5)
plotting.plot_board(sim_world)
lite_1.epsilon = 0.05
lite_2.epsilon = 0.05

win_count_1 = 0
win_count_2 = 0
for i in range(500):
    sim_world = Hex.initialize_state(5, 5)
    while not sim_world.is_current_state_final():

        nn_move = lite_2.get_action(sim_world)
        sim_world.play_action(nn_move, True)
        if sim_world.is_current_state_final():
            win_count_1 += 1
            break
        nn_2_move = lite_1.get_action(sim_world)
        sim_world.play_action(nn_2_move, True)
        if sim_world.is_current_state_final():
            win_count_2 += 1
            break
        #u = input("popp")
        #plotting.plot_board(sim_world)

print(f"Win count 1: {win_count_1}")
print(f"Win count 2: {win_count_2}")


