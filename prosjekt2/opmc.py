import random
from random import choice
import os
import multiprocessing as mp

import gui
from game_managers.game_manager import GameManager
from sim_worlds.sim_world import Players

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import time
import nn
from mcts import MCTS


class OnPolicyMonteCarlo:
    """
    Class connecting the game manager, the neural network, and the MCTS
    """

    def __init__(self, mgr: GameManager, i_s: int, actual_games: int, search_games: int, model, max_rbuf: int,  sample_rbuf: int, gui: gui.GameGUI = None):
        self.manager = mgr
        self.board_size = mgr.get_size()

        self.i_s = i_s  # Save interval for ANET parameters
        self.rbuf = {'states': [], 'dists': []}
        self.model = model
        self.num_actual_games = actual_games
        self.num_search_games = search_games
        self.max_rbuf_size = max_rbuf
        self.num_sample_rbuf = sample_rbuf
        self.gui = gui

    def run_games(self):
        start_time = time.time()
        # Caching an untrained net
        self.model.save(f"models/model_{self.board_size}_{0}")

        for g_a in range(self.num_actual_games):
            t = time.time()
            move_count = 0
            actual_state = self.manager.generate_initial_state()
            mcts = MCTS(self.manager, policy_object=nn.LiteModel.from_keras_model(self.model))
            if self.gui:
                self.gui.update_title(f"Game # {g_a}")
            while not self.manager.is_state_final(actual_state):

                moves_to_consider = mcts.search(self.num_search_games)
                # moves_to_consider = mcts.parallel_search(self.num_search_games)  # TODO: make parallelization work maybe
                action = self.choose_action(moves_to_consider)
                distribution = self.gen_distribution(moves_to_consider)

                self.rbuf['states'].append(self.manager.nn_state_representation(actual_state))
                self.rbuf['dists'].append(distribution)

                move_count += 1
                mcts.update_root(action)
                self.manager.play_action(action, actual_state, inplace=True)
                if self.gui:
                    if self.gui.get_board_state():
                        self.gui.update_plot(actual_state)
                    self.gui.update_gui()

            print(f"{(time.time() - t):.4} seconds. Game # {g_a}")

            # Remove oldest elements from replay buffer if too long
            if len(self.rbuf['states']) > self.max_rbuf_size:
                diff = len(self.rbuf['states']) - self.max_rbuf_size
                self.rbuf['states'] = self.rbuf['states'][diff:]
                self.rbuf['dists'] = self.rbuf['dists'][diff:]

            # Random sampling of states and distributions
            states, dists = zip(*random.sample(list(zip(self.rbuf['states'], self.rbuf['dists'])),
                                               min(len(self.rbuf['states']), self.num_sample_rbuf)))
            # Fitting model with 20 epochs and bs 8
            self.model.fit(np.array(states), np.array(dists), epochs=20, batch_size=4)

            # Saving model according to save interval
            if (g_a + 1) % self.i_s == 0:
                self.model.save(f"models/model_{self.board_size}_{g_a + 1}")

        print(f"Ran {self.num_actual_games}  episodes with {self.num_search_games} rollouts per move \n"
              f"Time spent: {(time.time() - start_time):.4} seconds")

    # Generates a distribution over the visit counts from MCTS
    def gen_distribution(self, nodes):
        dist = np.zeros(shape=(self.board_size, self.board_size), dtype='f')
        for move, value in nodes.items():
            dist[move] = value / self.num_search_games

        return dist.flatten()

    # Chooses action based on visit counts of nodes returned from MCTS
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
