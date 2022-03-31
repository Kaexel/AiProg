from copy import deepcopy

from disjoint_set import DisjointSetForest
from sim_worlds.sim_world import SimWorld
from sim_worlds.sim_world import Players
import numpy as np


neighbors = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]  # Possible neighbor offsets


class Hex(SimWorld):
    """
    Class implementing Hex as defined in docs
    """
    def __init__(self, board_x: int, board_y: int):
        super().__init__((board_x, board_y))
        self.forest = DisjointSetForest()
        self.board = np.array([[0 for i in range(board_y)] for j in range(board_x)])
        self.board = np.pad(self.board, 1, mode='constant', constant_values=2)
        self.board[0, :] = 1
        self.board[-1, :] = 1
        self.add_edges_to_forest(self.board)

    # TODO: make this less of a mess
    def add_edges_to_forest(self, board):
        # Top set
        last_val = None
        for row, val in enumerate(self.board[0, :]):
            if val == 1:
                cur_val = (0, row)
                self.forest.make_set(cur_val)
                if last_val is not None:
                    self.forest.union(self.forest.forest[last_val], self.forest.forest[cur_val])
                last_val = cur_val

        # Bottom set
        last_val = None
        for row, val in enumerate(self.board[-1, :]):
            if val == 1:
                cur_val = (self.board.shape[1] - 1, row)
                self.forest.make_set(cur_val)
                if last_val is not None:
                    self.forest.union(self.forest.forest[last_val], self.forest.forest[cur_val])
                last_val = cur_val

        # Left set
        last_val = None
        for row, val in enumerate(self.board[:, 0]):
            if val == 2:
                cur_val = (row, 0)
                self.forest.make_set(cur_val)
                if last_val is not None:
                    self.forest.union(self.forest.forest[last_val], self.forest.forest[cur_val])
                last_val = cur_val

        # Right set
        last_val = None
        for row, val in enumerate(self.board[:, -1]):
            if val == 2:
                cur_val = (row, self.board.shape[0] - 1)
                self.forest.make_set(cur_val)
                if last_val is not None:
                    self.forest.union(self.forest.forest[last_val], self.forest.forest[cur_val])
                last_val = cur_val

    def get_legal_actions(self):
        actions = []
        for row, tile_y in enumerate(self.board):
            for col, tile_x in enumerate(tile_y):
                if tile_x == 0:
                    actions.append((row - 1, col - 1))  # Abstracting away edges used for win check
        return actions

    def is_current_state_final(self):
        top_bot_connect = self.forest.is_connected(self.forest.forest[(0, 1)], self.forest.forest[(self.board.shape[0] - 1, 1)])
        left_right_connect = self.forest.is_connected(self.forest.forest[(1, 0)], self.forest.forest[(1, self.board.shape[1] - 1)])
        return top_bot_connect or left_right_connect

    def play_action(self, action, inplace=False):
        if inplace:
            state = self
        else:
            state = deepcopy(self)
        action = (action[0] + 1, action[1] + 1)     # Rejiggering into edge space
        placement = 1 if state.player_turn == Players.WHITE else 2  # Init placement indicator

        state.forest.make_set(action)
        for tile in neighbors:
            tile_coords = (action[0] + tile[0], action[1] + tile[1])
            if state.in_bounds(tile_coords):                                         # Check if tile is on board
                if state.board[tile_coords[0], tile_coords[1]] == placement:         # Check if neighbor tile belongs to player
                    #if tile_coords in self.forest.forest.keys():                   # Check if neighbor tile added to forest
                    state.forest.union(state.forest.forest[action], state.forest.forest[tile_coords])

        state.board[action[0]][action[1]] = placement
        state.switch_player_turn()
        return state
        # TODO: game over

    def get_sim_world_name(self):
        return "Hex"

    def channels(self):
        channels = []
        channel_empty = np.where(self.board == 0, 1, 0)
        channel_white = np.where(self.board == 1, 1, 0)
        channel_black = np.where(self.board == 2, 1, 0)
        channel_black_bridge = np.array((2, 2))
        channel_white_bridge = np.array((2, 2))
        channel_black_play = np.ones_like(self.board) if self.player_turn == Players.BLACK else np.zeros_like(self.board)
        channel_white_play = np.ones_like(self.board) if self.player_turn == Players.WHITE else np.zeros_like(self.board)
        channel_save_bridge = np.array((2, 2))
        channel_form_bridge = np.array((2, 2))

        return channel_empty


    def in_bounds(self, coords):
        size = self.board.shape
        if 0 <= coords[0] < size[0] and 0 <= coords[1] < size[1]:
            return True
        return False
