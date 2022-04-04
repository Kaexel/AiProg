from random import choice
import os

from game_managers.game_manager import GameManager

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import time
import nn
from mcts import MCTS


class OnPolicyMonteCarlo:
    def __init__(self, mgr: GameManager, i_s: int, actual_games: int, search_games: int):
        self.manager = mgr
        self.board_size = mgr.get_size()  # TODO: last inn fra config fil

        self.i_s = i_s  # Save interval for ANET parameters
        self.rbuf = {'states': [], 'dists': []}
        self.model = nn.make_keras_model(32, self.board_size, self.board_size)  # TODO: set params
        self.num_actual_games = actual_games
        self.num_search_games = search_games

    def run_games(self):

        for g_a in range(self.num_actual_games):
            t = time.time()
            move_count = 0
            actual_state = self.manager.generate_initial_state()
            mcts = MCTS(self.manager, policy_object=nn.LiteModel.from_keras_model(self.model))
            while not self.manager.is_state_final(actual_state):
                moves_to_consider = mcts.search(self.num_search_games)
                action = self.choose_action(moves_to_consider)  # TODO: fix this
                distribution = self.gen_distribution(moves_to_consider)

                self.rbuf['states'].append(self.manager.nn_state_representation(actual_state))
                self.rbuf['dists'].append(distribution)
                move_count += 1
                mcts.update_root(action)
                self.manager.play_action(action, actual_state, inplace=True)

            print(f"{(time.time() - t):.3} seconds. Game # {g_a}")
            self.model.fit(np.array(self.rbuf['states']), np.array(self.rbuf['dists']), epochs=10, batch_size=8)
            if g_a % self.i_s == 0:
                self.model.save(f"models/model_{g_a}")
                # Save keras model


    def gen_distribution(self, nodes):
        dist = np.zeros(shape=(self.board_size, self.board_size), dtype='f')
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


