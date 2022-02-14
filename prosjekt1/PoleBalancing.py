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
        config.read('config')
        self.pole_length = config['POLEBALANCING'].getfloat('POLE_LENGTH')
        self.pole_mass = config['POLEBALANCING'].getfloat('POLE_MASS')
        self.cart_mass = 1
        self.gravity = config['POLEBALANCING'].getfloat('GRAVITY')
        self.theta_m = 0.21
        self.x_minus = -2.4
        self.x_plus = 2.4
        self.tau = config['POLEBALANCING'].getfloat('TIMESTEP')
        self.episode_length = 300

        """ Needs reset each episode """
        self.theta = random.uniform(-self.theta_m, self.theta_m)
        self.theta_d1 = 0
        self.theta_d2 = 0
        self.x = 0
        self.x_d1 = 0
        self.x_d2 = 0
        self.bang_bang = 10

        self.final_state = False
        self.current_episode = 0
        self.reward = 0

    def reset_world(self):
        self.theta = random.uniform(-self.theta_m, self.theta_m)
        self.theta_d1 = 0
        self.theta_d2 = 0
        self.x = 0
        self.x_d1 = 0
        self.x_d2 = 0
        self.final_state = False
        self.current_episode = 0
        self.reward = 0

    def update_variables(self):
        self.theta_d2 = (self.gravity * math.sin(self.theta) + (math.cos(self.theta)*(-self.bang_bang - self.pole_mass * self.pole_length * (self.theta_d1**2) * math.sin(self.theta)) / (self.pole_mass + self.cart_mass))) \
                    / (self.pole_length*((4/3)-(self.pole_mass*math.cos(self.theta)**2)/(self.pole_mass + self.cart_mass)))
        self.x_d2 = (self.bang_bang + self.pole_mass * self.pole_length * (self.theta_d1**2 * math.sin(self.theta) - self.theta_d2*math.cos(self.theta))) / (self.pole_mass + self.cart_mass)
        self.theta_d1 += self.tau * self.theta_d2
        self.x_d1 += self.tau * self.x_d2
        self.theta += self.tau * self.theta_d1
        self.x += self.tau * self.x_d1

    """ SimWorld functions """

    def is_current_state_final(self):
        return self.final_state

    def produce_initial_state(self):
        state = self.x, self.x_d1, self.x_d2, self.theta, self.theta_d1, self.theta_d2
        test = tuple([float("{0:.2f}".format(n)) for n in state])
        # print(test)
        return test

    def get_legal_actions(self):
        return [-10, 10]

    def apply_action(self, action):
        self.bang_bang = action
        self.update_variables()
        self.current_episode += 1
        self.reward = self.current_episode * 10

        """ If current angle greater than threshold, state is final. """
        if abs(self.theta) > self.theta_m:
            self.final_state = True
            self.reward = 0
        if self.current_episode >= self.episode_length:
            print("jippi!")
            self.final_state = True
            self.reward = 1000
        state = (self.x, self.x_d1, self.x_d2, self.theta, self.theta_d1, self.theta_d2)
        test = tuple([float("{0:.2f}".format(n)) for n in state])
        #print(test)
        return test, self.reward

    def get_value(self):
        return self.current_episode

