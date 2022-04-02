from copy import deepcopy

from disjoint_set import DisjointSetForest
from sim_worlds.sim_world import SimWorld
from sim_worlds.sim_world import Players
import numpy as np


neighbors = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]  # Possible neighbor offsets
bridge_endpoints = [(-1, -1), (-2, 1), (-1, 2), (1, 1), (2, -1), (1, -2)] # Possible bridge points
bridge_dict = {(-1, -1): [(0, -1), (-1, 0)], (-2, 1): [], (-1, 2): [], (1, 1): [], (2, -1): [], (1, -2): []}


def add_edges_to_forest(board, forest, shape):
    top = (0, 1)
    bot = (shape[1] - 1, 1)
    left = (1, 0)
    right = (1, shape[0] - 1)

    forest.make_set(top); forest.make_set(bot); forest.make_set(left); forest.make_set(right)

    iterator = np.nditer(board, flags=["multi_index"])
    for element in iterator:
        if element == 1:
            forest.make_set(iterator.multi_index)
            if iterator.multi_index[0] == 0:
                forest.union(forest.forest[top], forest.forest[iterator.multi_index])
            elif iterator.multi_index[0] == shape[0] - 1:
                forest.union(forest.forest[bot], forest.forest[iterator.multi_index])

        elif element == 2:
            forest.make_set(iterator.multi_index)
            if iterator.multi_index[1] == 0:
                forest.union(forest.forest[left], forest.forest[iterator.multi_index])
            elif iterator.multi_index[1] == shape[1] - 1:
                forest.union(forest.forest[right], forest.forest[iterator.multi_index])


class Hex(SimWorld):
    """
    Class implementing Hex as defined in docs
    """
    @classmethod
    def clone_state(cls, state):
        board = state.board.copy()
        forest = state.forest.clone()
        forest = deepcopy(state.forest)
        return Hex(board, forest, board.shape, state.player_turn)

    @classmethod
    def initialize_state(cls, board_x, board_y):
        forest = DisjointSetForest()
        board = np.array([[0 for i in range(board_y)] for j in range(board_x)])
        board = np.pad(board, 1, mode='constant', constant_values=2)
        board[0, :] = 1
        board[-1, :] = 1
        board[0, 0] = 3
        add_edges_to_forest(board, forest, board.shape)
        return Hex(board, forest, (board_x, board_y))

    def __init__(self, board, forest, board_size: tuple, player_turn=Players.WHITE):
        super().__init__(board.shape)
        self.forest = forest
        self.board = board
        self.player_turn = player_turn

    def get_legal_actions(self):
        actions = []
        iterator = np.nditer(self.board, flags=["multi_index"])
        for tile in iterator:
            if tile == 0:
                actions.append((iterator.multi_index[0] - 1, iterator.multi_index[1] - 1)) 
        return actions

    def get_illegal_actions(self):
        actions = []
        for row, tile_y in enumerate(self.board):
            for col, tile_x in enumerate(tile_y):
                if tile_x != 0:
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
            state = Hex.clone_state(self)
        action = (action[0] + 1, action[1] + 1)     # Rejiggering into edge space

        self.loopy(action, state)

        state.switch_player_turn()
        return state
        # TODO: game over

    def loopy(self, action, state):
        placement = 1 if state.player_turn == Players.WHITE else 2  # Init placement indicator
        state.forest.make_set(action)
        for tile in neighbors:
            tile_coords = (action[0] + tile[0], action[1] + tile[1])
            if state.in_bounds(tile_coords):                                         # Check if tile is on board
                if state.board[tile_coords[0], tile_coords[1]] == placement:         # Check if neighbor tile belongs to player
                    #if tile_coords in self.forest.forest.keys():                   # Check if neighbor tile added to forest
                    state.forest.union(state.forest.forest[action], state.forest.forest[tile_coords])

        state.board[action[0]][action[1]] = placement

    def get_final_result(self):
        assert self.is_current_state_final()
        # Current player has been swapped after last tile placed. We need to return the opposite from current turn.
        return Players.WHITE.value if self.player_turn == Players.BLACK else Players.BLACK.value


    def get_sim_world_name(self):
        return "Hex"

    def get_board_shape(self):
        return self.board.shape[0] - 2, self.board.shape[1] - 2

    def nn_state_representation(self):
        channels = np.zeros((5, self.board.shape[1], self.board.shape[0]), dtype=int)

        channels[0] = np.where(self.board == 0, 1, 0)
        channels[1] = np.where(self.board == 1, 1, 0)
        channels[2] = np.where(self.board == 2, 1, 0)
        channels[3] = np.ones_like(self.board) if self.player_turn == Players.BLACK else np.zeros_like(self.board)
        channels[4] = np.ones_like(self.board) if self.player_turn == Players.WHITE else np.zeros_like(self.board)
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
        if self.player_turn == Players.BLACK:
            channels = np.rot90(channels, axes=(1, 2))

        return channels

    def in_bounds(self, coords):
        size = self.board.shape
        if 0 <= coords[0] < size[0] and 0 <= coords[1] < size[1]:
            return True
        return False

