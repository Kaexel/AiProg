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
        config.read('config')
        self.discount_factor = config["PRIMARY"].getfloat('DISCOUNT_ACTOR')
        self.lr = config["PRIMARY"].getfloat('ACTOR_LR')
        self.edr = config["PRIMARY"].getfloat('EDR_ACTOR')
        self.epsilon = config["PRIMARY"].getfloat('EPSILON_INITIAL')
        self.epsilon_decay = config["PRIMARY"].getfloat('EPSILON_DECAY')
        self.sim_world = sim_world
        """ Policy and eligibilities initialized with 0 """
        self.policy = defaultdict(int)
        self.eligibilities = defaultdict(int)

    def reset_e(self):
        self.eligibilities = defaultdict(int)

    def get_action(self, state, possible_actions):
        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        print(possible_actions)
        keys = [(state, k) for k in possible_actions]
        temp = {k: self.policy[k] for k in keys}
        return max(temp, key=lambda k: temp[k])[1]

    def set_eligibility(self, state, action, value):
        self.eligibilities[(state, action)] = value

    def update_state_action(self, td_error):
        self.policy.update({k: v + self.lr*td_error*self.eligibilities[k] for k, v in self.policy.items()})

    def update_e(self):
        self.eligibilities.update({k: self.discount_factor * self.edr * v for k, v in self.eligibilities.items()})

    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay


class Critic:

    def __init__(self, sim_world: SimWorld):
        assert isinstance(sim_world, SimWorld)
        config = configparser.ConfigParser()
        config.read('config')
        self.discount_factor = config["PRIMARY"].getfloat('DISCOUNT_CRITIC')
        self.lr = config["PRIMARY"].getfloat('CRITIC_LR')
        self.edr = config["PRIMARY"].getfloat('EDR_CRITIC')
        self.nn_dims = ast.literal_eval(config["PRIMARY"]['CRITIC_NN_DIMS'])
        self.sim_world = sim_world
        self.td_error = 1

        """ Values initialized with small random values, eligibilites with 0"""
        self.values = defaultdict(lambda: random.random())
        self.eligibilities = defaultdict(int)

    def reset_e(self):
        self.eligibilities = defaultdict(int)

    def set_eligibility(self, state, value):
        self.eligibilities[state] = value

    def get_td_error(self):
        return self.td_error

    def update_td_error(self, reward, state, state_next):
        self.td_error = reward + self.discount_factor * self.values[state_next] - self.values[state]

    def update_state_action(self):
        self.values.update({k: v + self.lr*self.td_error*self.eligibilities[k] for k, v in self.values.items()})
        #TODO: usikker på om eligibilites riktig satt

    def update_e(self):
        self.eligibilities.update({k: self.discount_factor * self.edr * v for k, v in self.eligibilities.items()})
