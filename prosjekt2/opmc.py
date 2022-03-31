from copy import deepcopy

import torch

import nn
from sim_worlds.hex import Hex
from mcts import MCTS


class OnPolicyMonteCarlo:
    def __init__(self):
        self.i_s = 50  # Save interval for ANET parameters
        self.rbuf = []
        self.model = nn.NeuralNetwork(1, 2, 3)  # TODO: set params
        self.num_actual_games = 50
        self.num_search_games = 50

    def run_games(self):
        for g_a in range(self.num_actual_games):
            sim_world = Hex(4, 4)  # TODO: set params
            mcts = MCTS(deepcopy(sim_world))
            while not sim_world.is_current_state_final():
                mc_game_board = MCTS(deepcopy(sim_world))
                mc_game_board.search(self.num_search_games)
                distribution = []  # TODO: fix this
                self.rbuf.append((deepcopy(sim_world), distribution))
                action = max(distribution)  # TODO: fix this
                sim_world = sim_world.play_action(action)
                mcts.update_root(action)
            # train model on minibatch
            if g_a % self.i_s == 0:
                torch.save(self.model.state_dict(), f"models/model_{g_a}")



