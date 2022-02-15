import random
import numpy as np
from Hanoi import Hanoi
import configparser
import ast
from SimWorld import SimWorld
from collections import defaultdict


class Actor:

    def __init__(self, sim_world: SimWorld):
        assert isinstance(sim_world, SimWorld)

        config = configparser.ConfigParser()
        config.read('config.ini')
        self.discount_factor = config["PRIMARY"].getfloat('DISCOUNT_ACTOR')
        self.lr = config["PRIMARY"].getfloat('ACTOR_LR')
        self.edr = config["PRIMARY"].getfloat('EDR_ACTOR')
        self.epsilon = config["PRIMARY"].getfloat('EPSILON_INITIAL')
        self.epsilon_decay = config["PRIMARY"].getfloat('EPSILON_DECAY')
        self.sim_world = sim_world
        """ Policy and eligibilities initialized with 0 """
        self.policy = defaultdict(int)
        self.eligibilities = defaultdict(int)

    def max_greed(self):
        self.epsilon = 0

    def reset_e(self):
        self.eligibilities = defaultdict(int)

    def get_action(self, state, possible_actions):
        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        keys = [(state, k) for k in possible_actions]
        temp = {k: self.policy[k] for k in keys}
        return max(temp, key=lambda k: temp[k])[1]


    def set_eligibility(self, state, action, value):
        self.eligibilities[(state, action)] = value

    def update_state_action(self, current_sa, td_error):
        self.policy.update({k: self.policy[k] + self.lr*td_error*self.eligibilities[k] for k in current_sa})

    def update_e(self, current_sa):
        self.eligibilities.update({k: self.discount_factor * self.edr * self.eligibilities[k] for k in current_sa})

    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay

