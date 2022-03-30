from copy import deepcopy

from sim_world import SimWorld


class Nim(SimWorld):
    """
    Class implementing the simple NIM game defined in the docs
    """

    def __init__(self, n, k):
        super().__init__((0, 0))
        self.n = n
        self.k = k

    def get_legal_actions(self):
        return [*range(1, min(self.k + 1, self.n + 1))]  # Allowed moves

    def play_action(self, action):
        state = deepcopy(self)
        state.n -= action
        if not state.is_current_state_final():
            state.switch_player_turn()
        return state

    def get_sim_world_name(self):
        return "Nim"

    def is_current_state_final(self):
        return self.n <= 0



