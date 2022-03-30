from copy import deepcopy

from sim_world import SimWorld
import random


class OldGold(SimWorld):
    """
    Class implementing Old Gold defined in docs
    """

    def __init__(self, board_length):
        super().__init__((1, board_length))
        self.board_length = 15
        self.board = [0] * self.board_length
        #self.board[random.randrange(0, self.board_length)] = 2
        self.board = [0, 0, 2, 0]
        self.game_over = False

    def get_legal_actions(self):
        actions = []
        last_occupied = -1
        for (i, t) in enumerate(self.board):           # Loop through board
            if t == 0:
                continue                               # Ignore empty spaces
            if i == 0:
                actions.append((t, 0, -1))             # Move off ledge (move type t from cell 0 to cell -1 (off-board))
            else:
                for j in range(last_occupied + 1, i):  # Loop every position between last occupied and current position
                    actions.append((t, i, j))          # Add every cell between last occupied and current position
            last_occupied = i                          # Update the last occupied cell
        return actions

    def is_current_state_final(self):
        return self.game_over

    def play_action(self, action):
        state = deepcopy(self)
        if action[2] == -1:
            if action == (2, 0, -1):                   # If gold moved off ledge, state is final
                state.game_over = True
                # TODO update game state
            state.board[0] = 0
        else:
            state.board[action[1]], state.board[action[2]] = state.board[action[2]], state.board[action[1]]
            state.switch_player_turn()
        return state

    def get_sim_world_name(self):
        pass


