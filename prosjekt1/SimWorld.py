from abc import ABC, abstractmethod


class SimWorld:
    def __init__(self):
        pass

    @abstractmethod
    def is_current_state_final(self):
        self.__impl_reminder()
        raise NotImplementedError

    @abstractmethod
    def produce_initial_state(self):
        self.__impl_reminder()
        raise NotImplementedError

    @abstractmethod
    def get_legal_actions(self):
        self.__impl_reminder()

    @abstractmethod
    def apply_action(self, action):
        self.__impl_reminder()
        raise NotImplementedError

    @abstractmethod
    def reset_world(self):
        self.__impl_reminder()
        raise NotImplementedError

    @abstractmethod
    def plot_world_state(self):
        raise NotImplementedError

    def produce_state_definition(self):
        self.__impl_reminder()
        raise NotImplementedError

    def __impl_reminder(self):
        print(f"husk å implementere dette for klasse {type(self)} axel!!")
