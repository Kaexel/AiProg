from abc import ABC, abstractmethod

"""
Abstract base class defining functions all SimWorlds must implement.
"""


class SimWorld:
    def __init__(self):
        pass

    @abstractmethod
    def get_sim_world_name(self):
        raise NotImplementedError

    @abstractmethod
    def is_current_state_final(self):
        raise NotImplementedError

    @abstractmethod
    def produce_initial_state(self):
        raise NotImplementedError

    @abstractmethod
    def get_legal_actions(self):
        raise NotImplementedError

    @abstractmethod
    def apply_action(self, action):
        raise NotImplementedError

    @abstractmethod
    def reset_world(self):
        raise NotImplementedError

    @abstractmethod
    def plot_world_state(self):
        raise NotImplementedError

    @abstractmethod
    def visualize_best(self):
        raise NotImplementedError

    @abstractmethod
    def produce_state_definition(self):
        raise NotImplementedError
