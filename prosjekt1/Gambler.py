import math
import random
import configparser
import ast

from SimWorld import SimWorld


class Gambler(SimWorld):
    def __init__(self):
        super().__init__()
        config = configparser.ConfigParser()
        config.read('config.ini')
        self.monetary_units = random.randint(1, 99)
        self.initial_money = self.monetary_units
        self.win_prob = config['GAMBLER'].getfloat('WIN_PROB')

    def is_current_state_final(self):
        return self.monetary_units <= 0 or self.monetary_units >= 100

    def produce_initial_state(self):
        return self.monetary_units

    def get_legal_actions(self):
        return list(range(0, min(self.monetary_units, 100 - self.monetary_units)))

    def apply_action(self, action):
        if random.random() <= self.win_prob:
            self.monetary_units += action
        else:
            self.monetary_units -= action

        return self.monetary_units,  self.monetary_units - self.initial_money

    def reset_world(self):
        self.monetary_units = random.randint(1, 99)

    def plot_world_state(self):
        pass



p = Gambler()
