import math
import random
import configparser
import ast

import numpy as np

from SimWorld import SimWorld


class PoleBalancing(SimWorld):
    def __init__(self):
        super().__init__()
        config = configparser.ConfigParser()
        config.read('config.ini')
        self.pole_length = config['POLEBALANCING'].getfloat('POLE_LENGTH')
        self.pole_mass = config['POLEBALANCING'].getfloat('POLE_MASS')
        self.cart_mass = 1
        self.gravity = config['POLEBALANCING'].getfloat('GRAVITY')
        self.theta_m = 0.21
        self.x_minus = -2.4
        self.x_plus = 2.4
        self.tau = config['POLEBALANCING'].getfloat('TIMESTEP')
        self.episode_length = 300

        self.x_bins = ([-3.1, -1.5], [-1.5, -1], [-1, -0.5], [-0.5, 0], [0, 0.5], [0.5, 1], [1, 1.5], [1.5, 3.1])
        self.theta_bins = ([-5.21, -0.15], [-0.15, -0.1], [-0.1, -0.05], [-0.05, 0], [0, 0.05], [0.05, 0.1], [0.10, 0.15], [0.15, 5.21])
        self.x_d1_bins = ([-4.1, -1.5], [-1.5, -1], [-1, -0.5], [-0.5, 0], [0, 0.5], [0.5, 1], [1, 1.5], [1.5, 4.1])
        self.theta_d1_bins = ([-5, -1.5], [-1.5, -1], [-1, -0.5], [-0.5, 0], [0, 0.5], [0.5, 1], [1, 1.5], [1.5, 5])

        self.bin_dict = {'x_bins': self.x_bins, 'theta_bins': self.theta_bins, 'x_d1_bins': self.x_d1_bins, 'theta_d1_bins': self.theta_d1_bins}

        """ Below vars need reset each episode """
        self.theta = random.uniform(-self.theta_m, self.theta_m)
        self.theta_d1 = 0
        self.theta_d2 = 0
        self.x = 0
        self.x_d1 = 0
        self.x_d2 = 0
        self.bang_bang = 10

        self.current_episode = 0
        self.reward = 0

    def reset_world(self):
        self.theta = random.uniform(-self.theta_m, self.theta_m)
        self.theta_d1 = 0
        self.theta_d2 = 0
        self.x = 0
        self.x_d1 = 0
        self.x_d2 = 0
        self.current_episode = 0
        self.reward = 0

    def update_variables(self):
        self.theta_d2 = (self.gravity * math.sin(self.theta) + (math.cos(self.theta)*(-self.bang_bang - self.pole_mass * self.pole_length * (self.theta_d1**2) * math.sin(self.theta)) / (self.pole_mass + self.cart_mass))) \
                    / (self.pole_length*((4/3) - (self.pole_mass*math.cos(self.theta)**2)/(self.pole_mass + self.cart_mass)))
        self.x_d2 = (self.bang_bang + self.pole_mass * self.pole_length * (self.theta_d1**2 * math.sin(self.theta) - self.theta_d2*math.cos(self.theta))) / (self.pole_mass + self.cart_mass)

        self.theta_d1 += self.tau * self.theta_d2
        self.x_d1 += self.tau * self.x_d2
        self.theta += self.tau * self.theta_d1
        self.x += self.tau * self.x_d1

    def produce_state_definition(self):
        #c_theta, c_theta_d1, c_x, c_x_d1 = -1, -1, -1, -1
        state = {'x': self.x, 'theta': self.theta, 'x_d1': self.x, 'theta_d1': self.theta_d1}
        for v, b in zip(state, self.bin_dict):
            for i in range(len(self.bin_dict[b])):
                if self.bin_dict[b][i][0] <= state[v] < self.bin_dict[b][i][1]:
                    state[v] = i
                    break
        return tuple([k for k in state.values()])

    def get_reward(self):
        if self.current_episode >= self.episode_length:
            return 100
        if self.is_current_state_final() and self.current_episode < self.episode_length:
            return -1000 * self.episode_length / self.current_episode
        else:
            return self.current_episode
            return self.current_episode / self.episode_length


    """ SimWorld functions """

    def is_current_state_final(self):
        if abs(self.theta) > self.theta_m:
            return True
        if abs(self.x) > self.x_plus:
            return True
        if self.current_episode >= self.episode_length:
            return True
        return False

    def produce_initial_state(self):
        state = self.x, self.x_d1, self.theta, self.theta_d1
        return self.produce_state_definition()

    def get_legal_actions(self):
        return [-10, 10]

    def plot_world_state(self):
        return

    def apply_action(self, action):
        self.bang_bang = action
        self.update_variables()
        self.current_episode += 1
        self.reward = self.current_episode * 10
        """ If current angle greater than threshold, state is final. """

        # state = (self.x, self.x_d1, self.x_d2, self.theta, self.theta_d1, self.theta_d2)
        state = (self.x, self.x_d1, self.theta, self.theta_d1)
        #state = self.x_d1
        #test = tuple([float("{0:.2f}".format(n)) for n in state])
        return self.produce_state_definition(), self.get_reward()

    def get_value(self):
        return self.current_episode

