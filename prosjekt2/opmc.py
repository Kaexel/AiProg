from configparser import ConfigParser
from copy import deepcopy
from random import choice
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import torch
import time
import nn
from sim_worlds.hex import Hex
from mcts import MCTS
from sim_worlds.sim_world import SimWorld


class OnPolicyMonteCarlo:
    def __init__(self, sim_world: SimWorld):
        self.sim_world = sim_world
        self.board_size = sim_world.board_size

        self.i_s = 50  # Save interval for ANET parameters
        self.rbuf = {'states': [], 'dists': []}
        self.model = nn.make_keras_model(32, self.board_size[1], self.board_size[0])  # TODO: set params
        self.num_actual_games = 10
        self.num_search_games = 1000

    def run_games(self):

        for g_a in range(self.num_actual_games):
            t = time.time()
            move_count = 0
            sim_world = deepcopy(self.sim_world)
            mcts = MCTS(deepcopy(sim_world), policy_object=nn.LiteModel.from_keras_model(self.model))
            while not sim_world.is_current_state_final():
                moves_to_consider = mcts.search(self.num_search_games)
                action = self.choose_action(moves_to_consider)  # TODO: fix this
                distribution = self.gen_distribution(moves_to_consider)

                self.rbuf['states'].append(sim_world.nn_state_representation())
                self.rbuf['dists'].append(distribution)
                move_count += 1
                mcts.update_root(action)
                sim_world = sim_world.play_action(action)

            print(f"{(time.time() - t):.3} seconds. Game # {g_a}")
            self.model.fit(np.array(self.rbuf['states']), np.array(self.rbuf['dists']), epochs=10, batch_size=1)
            if g_a % self.i_s == 0:
                pass
                # Save keras model
        self.model.save("models")

    def gen_distribution(self, nodes):
        dist = np.zeros(shape=self.board_size, dtype='f')
        for move, value in nodes.items():
            dist[move] = value / self.num_search_games
        return dist.flatten()

    def choose_action(self, distribution):
        max_actions = []
        best_value = 0
        for action, value in distribution.items():
            if value > best_value:
                max_actions = [action]
                best_value = value
            elif value == best_value:
                max_actions.append(action)

        return choice(max_actions)


