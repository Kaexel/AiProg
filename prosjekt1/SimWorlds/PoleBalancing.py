import math
import random
import configparser

import matplotlib.axes
import numpy as np
from matplotlib import pyplot as plt

from SimWorlds.SimWorld import SimWorld


def produce_bins(limits, num_bins):
    """
    Returns num_bins evenly spaced bins given limits "
    """
    diff = np.sum(np.abs(limits))
    step = diff/num_bins
    bins = [[limits[0] + i*step, limits[0] + i*step + step] for i in range(num_bins)]
    bins.insert(0, [-float('inf'), limits[0]])
    bins.append([limits[1], float('inf')])
    print(bins)
    return bins


class PoleBalancing(SimWorld):
    def __init__(self):
        super().__init__()
        self.name = "Pole balancing"
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
        self.num_bins = config['POLEBALANCING'].getint('NUM_BINS')

        """
        Splitting state space into bins
        """
        self.x_bins = produce_bins((-2.4, 2.4), self.num_bins)
        self.theta_bins = produce_bins((-0.21, 0.21), self.num_bins)
        self.x_d1_bins = produce_bins((-2.5, 2.5), self.num_bins)
        self.theta_d1_bins = produce_bins((-3, 3), self.num_bins)

        self.bin_dict = {'x_bins': self.x_bins, 'theta_bins': self.theta_bins, 'x_d1_bins': self.x_d1_bins, 'theta_d1_bins': self.theta_d1_bins}

        """ Below vars need reset each episode """
        self.theta = random.uniform(-self.theta_m, self.theta_m)
        self.theta_d1 = 0
        self.theta_d2 = 0
        self.x = 0
        self.x_d1 = 0
        self.x_d2 = 0
        self.bang_bang = 10
        self.angle_history = []
        self.best_angle_history = []

        self.current_episode = 0
        self.reward = 0

    def reset_world(self):
        self.theta = random.uniform(-self.theta_m, self.theta_m)
        if len(self.angle_history) >= len(self.best_angle_history):
            self.best_angle_history = self.angle_history
        self.angle_history = []
        self.theta_d1 = 0
        self.theta_d2 = 0
        self.x = 0
        self.x_d1 = 0
        self.x_d2 = 0
        self.current_episode = 0
        self.reward = 0
        self.angle_history.append(self.theta)

    def update_variables(self):
        self.theta_d2 = (self.gravity * math.sin(self.theta) + (math.cos(self.theta)*(-self.bang_bang - self.pole_mass * self.pole_length * (self.theta_d1**2) * math.sin(self.theta)) / (self.pole_mass + self.cart_mass))) \
                    / (self.pole_length*((4/3) - (self.pole_mass*math.cos(self.theta)**2)/(self.pole_mass + self.cart_mass)))
        self.x_d2 = (self.bang_bang + self.pole_mass * self.pole_length * (self.theta_d1**2 * math.sin(self.theta) - self.theta_d2*math.cos(self.theta))) / (self.pole_mass + self.cart_mass)

        self.theta_d1 += self.tau * self.theta_d2
        self.x_d1 += self.tau * self.x_d2
        self.theta += self.tau * self.theta_d1
        self.x += self.tau * self.x_d1
        self.angle_history.append(self.theta)

    def get_reward(self):
        """
        Trying some different reward schemes. Seems like string negative reward works best
        Also try to give better reward when acceleration (both theta and x) moves toward middle
        """
        if self.current_episode >= self.episode_length:
            return 10000
        if self.is_current_state_final() and self.current_episode < self.episode_length:
            return -10000
        if (self.theta_d2 < 0 and self.theta < 0) or (self.theta_d2 > 0 and self.theta > 0):
            return self.current_episode * 0.6
        if (self.x_d2 < 0 and self.x < 0) or (self.x_d2 > 0 and self.x > 0):
            return self.current_episode * 0.67
        else:
            return self.current_episode * 1

    """ SimWorld functions """
    def get_sim_world_name(self):
        return self.name

    def is_current_state_final(self):
        """ If current angle greater than threshold, state is final. Also if cart outside of X-bounds"""
        if (abs(self.theta) > self.theta_m) or (abs(self.x) > self.x_plus):
            return True
        if self.current_episode >= self.episode_length:
            return True
        return False

    def produce_initial_state(self):
        return self.produce_state_definition()

    def get_legal_actions(self):
        return [-10, 10]

    def apply_action(self, action):
        self.bang_bang = action
        self.update_variables()
        self.current_episode += 1
        self.reward = self.current_episode * 10
        return self.produce_state_definition(), self.get_reward()

    def plot_world_state(self):
        return

    def visualize_best(self):
        fig = plt.figure()
        ax: matplotlib.axes.Axes = fig.add_subplot(111)
        ax.set_title("")
        ax.set_ylim(-0.21, 0.21)
        ax.set_title("Best plot")
        ax.set_xlabel("Angle (Radians)")
        ax.set_ylabel("Timestep")
        ax.plot(self.best_angle_history)
        plt.show()

    def produce_state_definition(self):
        state = {'x': self.x, 'theta': self.theta, 'x_d1': self.x, 'theta_d1': self.theta_d1}
        for v, b in zip(state, self.bin_dict):
            for i in range(len(self.bin_dict[b])):
                if self.bin_dict[b][i][0] <= state[v] < self.bin_dict[b][i][1]:
                    state[v] = i
                    break
        return tuple([k for k in state.values()])



