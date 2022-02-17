import random
import configparser
from collections import defaultdict


class Actor:
    """
    Class implementing Actor portion of Actor-Critic
    """
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('config.ini')
        self.discount_factor = config["PRIMARY"].getfloat('DISCOUNT_ACTOR')
        self.lr = config["PRIMARY"].getfloat('ACTOR_LR')
        self.edr = config["PRIMARY"].getfloat('EDR_ACTOR')
        self.epsilon = config["PRIMARY"].getfloat('EPSILON_INITIAL')
        self.epsilon_decay = config["PRIMARY"].getfloat('EPSILON_DECAY')
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
        keys = {self.policy[(state, a)]: a for a in possible_actions}

        return keys[max(keys)]

    def set_eligibility(self, state, action, value):
        self.eligibilities[(state, action)] = value

    def update_state_action(self, current_sa, td_error):
        self.policy.update({k: self.policy[k] + self.lr*td_error*self.eligibilities[k] for k in current_sa})

    def update_e(self, current_sa):
        self.eligibilities.update({k: self.discount_factor * self.edr * self.eligibilities[k] for k in current_sa})

    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay

