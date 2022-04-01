from copy import deepcopy

from disjoint_set import DisjointSetForest
from sim_worlds.sim_world import SimWorld
from sim_worlds.sim_world import Players
import numpy as np


neighbors = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]  # Possible neighbor offsets


# TODO: make this less of a mess
def add_edges_to_forest(board, forest):
    # Top set
    last_val = None
    for row, val in enumerate(board[0, :]):
        if val == 1:
            cur_val = (0, row)
            forest.make_set(cur_val)
            if last_val is not None:
                forest.union(forest.forest[last_val], forest.forest[cur_val])
            last_val = cur_val

    # Bottom set
    last_val = None
    for row, val in enumerate(board[-1, :]):
        if val == 1:
            cur_val = (board.shape[1] - 1, row)
            forest.make_set(cur_val)
            if last_val is not None:
                forest.union(forest.forest[last_val], forest.forest[cur_val])
            last_val = cur_val

    # Left set
    last_val = None
    for row, val in enumerate(board[:, 0]):
        if val == 2:
            cur_val = (row, 0)
            forest.make_set(cur_val)
            if last_val is not None:
                forest.union(forest.forest[last_val], forest.forest[cur_val])
            last_val = cur_val

    # Right set
    last_val = None
    for row, val in enumerate(board[:, -1]):
        if val == 2:
            cur_val = (row, board.shape[0] - 1)
            forest.make_set(cur_val)
            if last_val is not None:
                forest.union(forest.forest[last_val], forest.forest[cur_val])
            last_val = cur_val


class Hex(SimWorld):
    """
    Class implementing Hex as defined in docs
    """
    @classmethod
    def clone_state(cls, state):
        board = state.board.copy()
        forest = state.forest
        converter = tf.lite.TFLiteConverter.from_keras_model(kmodel)
        tflite_model = converter.convert()
        return LiteModel(tf.lite.Interpreter(model_content=tflite_model))

    @classmethod
    def initialize_state(cls, board_x, board_y):
        forest = DisjointSetForest()
        board = np.array([[0 for i in range(board_y)] for j in range(board_x)])
        board = np.pad(board, 1, mode='constant', constant_values=2)
        board[0, :] = 1
        board[-1, :] = 1
        add_edges_to_forest(forest, board)
        cls(board, forest, (board_x, board_y))

    def __init__(self, board, forest, board_size: tuple):
        super().__init__(board.shape)
        self.forest = forest
        self.board = board


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

    def nn_state_representation(self):
        channels = np.zeros((3, self.board.shape[1], self.board.shape[0]), dtype=int)

        channels[0] = np.where(self.board == 0, 1, 0)
        channels[1] = np.where(self.board == 1, 1, 0)
        channels[2] = np.where(self.board == 2, 1, 0)
        """
        channel_empty = np.where(self.board == 0, 1, 0)
        channel_white = np.where(self.board == 1, 1, 0)
        channel_black = np.where(self.board == 2, 1, 0)
        channel_black_bridge = np.array((2, 2))
        channel_white_bridge = np.array((2, 2))
        channel_black_play = np.ones_like(self.board) if self.player_turn == Players.BLACK else np.zeros_like(self.board)
        channel_white_play = np.ones_like(self.board) if self.player_turn == Players.WHITE else np.zeros_like(self.board)
        channel_save_bridge = np.array((2, 2))
        channel_form_bridge = np.array((2, 2))
        """

        return channels

    def in_bounds(self, coords):
        size = self.board.shape
        if 0 <= coords[0] < size[0] and 0 <= coords[1] < size[1]:
            return True
        return False

    def clone(self):

