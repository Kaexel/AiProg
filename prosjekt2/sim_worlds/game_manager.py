from abc import ABC, abstractmethod


class GameManager(ABC):


    @abstractmethod
    def generate_initial_state(self):
        raise NotImplementedError

    @abstractmethod
    def get_legal_actions(self, state):
        raise NotImplementedError

    @abstractmethod
    def play_action(self, action, state, inplace=False):
        ...

    @abstractmethod
    def is_state_final(self, state) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_final_result(self, state) -> int:
        raise NotImplementedError

    @abstractmethod
    def nn_state_representation(self, state):
        raise NotImplementedError
