from copy import deepcopy

from disjoint_set import DisjointSetForest
from sim_worlds.game_manager import GameManager
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


class HexState:
    """
    Class implementing a minimal state for Hex. Done to avoid copying the whole hex object each time.
    """
    def __init__(self, board: np.ndarray, player_turn=Players.WHITE):
        self.player_turn: Players = player_turn
        self.board = board

    def __copy__(self):
        return HexState(board=self.board.copy(), player_turn=self.player_turn)


class HexManager(GameManager):
    """
    Class implementing a manager for Hex as defined in docs
    """
    def __init__(self, board_size):
        self.board_size: int = board_size

    def generate_initial_state(self):
        return HexState(np.array([[0 for _ in range(self.board_size)] for _ in range(self.board_size)], dtype='i'))

    def get_legal_actions(self, state: HexState):
        actions = set()
        iterator = np.nditer(state.board, flags=["multi_index"])
        for tile in iterator:
            if tile == 0:
                actions.add((iterator.multi_index[0], iterator.multi_index[1]))
        return actions

    def board_traversal_dfs(self, current_tile, board, player_turn, visited=None):
        if visited is None:
            visited = set()
        if current_tile in visited:
            return False

        visited.add(current_tile)

        # If tile at other side, return true
        if (player_turn == Players.WHITE and current_tile[0] == board.shape[0] - 1) or (player_turn == Players.BLACK and current_tile[1] == board.shape[1] - 1):
            return True

        for neighbor in neighbors:
            tile_coords = (current_tile[0] + neighbor[0], current_tile[1] + neighbor[1])
            if self.in_bounds(tile_coords) and tile_coords not in visited and board[tile_coords[0], tile_coords[1]] == player_turn.value:  # Check if neighbor tile belongs to player
                if self.board_traversal_dfs(tile_coords, board, player_turn, visited):
                    return True
        return False

    # May be a better way to check final result? Not disjoint forest, too slow to deepcopy
    def is_state_final(self, state):
        board = state.board
        player_turn = self.switch_player_turn(state.player_turn)  # We only need to check if game is finished for the player who last placed a tile. We switch the current player in the state
        starts = ((0, x) for x in range(board.shape[1]) if board[0, x] == player_turn.value) if player_turn == Players.WHITE else ((y, 0) for y in range(board.shape[0]) if board[y, 0] == player_turn.value)
        for start in starts:
            if self.board_traversal_dfs(start, state.board, player_turn):
                return True
        return False

    def play_action(self, action, state, inplace=False):

        if inplace:
            state.board[action[0], action[1]] = state.player_turn.value
            state.player_turn = self.switch_player_turn(state.player_turn)
        else:
            board = state.board.copy()
            board[action[0], action[1]] = state.player_turn.value
            return HexState(board, self.switch_player_turn(state.player_turn))

    def get_final_result(self, state):
        # Need to return opposite of current turn, since player turn swapped after final move
        return Players.WHITE.value if state.player_turn == Players.BLACK else Players.BLACK.value

    def nn_state_representation(self, state):
        channels = np.zeros((5, state.board.shape[1], state.board.shape[0]), dtype=int)
        channels[0] = np.where(state.board == 0, 1, 0)
        channels[1] = np.where(state.board == 1, 1, 0)
        channels[2] = np.where(state.board == 2, 1, 0)
        channels[3] = np.ones_like(state.board) if state.player_turn == Players.BLACK else np.zeros_like(state.board)
        channels[4] = np.ones_like(state.board) if state.player_turn == Players.WHITE else np.zeros_like(state.board)
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
        if state.current_player == Players.BLACK:
            channels = np.rot90(channels, axes=(1, 2))

        return channels

    def switch_player_turn(self, player_turn: Players):
        return Players.WHITE if player_turn == Players.BLACK else Players.BLACK

    def in_bounds(self, coords: tuple):
        # Assuming board equal dimensions
        if 0 <= coords[0] < self.board_size and 0 <= coords[1] < self.board_size:
            return True
        return False

