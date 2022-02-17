import random
import configparser
from SimWorlds.SimWorld import SimWorld


class Gambler(SimWorld):
    """
    Implements the Gambler SimWorld
    """
    def __init__(self):
        super().__init__()
        self.name = "Gambler"
        config = configparser.ConfigParser()
        config.read('config.ini')
        self.monetary_units = random.randint(1, 99)
        self.initial_money = self.monetary_units
        self.win_prob = config['GAMBLER'].getfloat('WIN_PROB')

    def produce_state_definition(self):
        pass

    """ SimWorld Functions """
    def get_sim_world_name(self):
        return self.name

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
        # Using total win this episode as reward
        return self.monetary_units,  self.monetary_units - self.initial_money

    def reset_world(self):
        self.monetary_units = random.randint(1, 99)

    def plot_world_state(self):
        pass

    def visualize_best(self):
        pass
