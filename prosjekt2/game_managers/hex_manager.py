from game_managers.game_manager import GameManager
from sim_worlds.sim_world import Players
import numpy as np



neighbors = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]  # Possible neighbor offsets
# TODO: apply bridge information to representation
bridge_endpoints = [(-1, -1), (-2, 1), (-1, 2), (1, 1), (2, -1), (1, -2)]  # Possible bridge points
bridge_dict = {(-1, -1): [(0, -1), (-1, 0)], (-2, 1): [], (-1, 2): [], (1, 1): [], (2, -1): [], (1, -2): []}


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
        super().__init__(board_size)
        self.zero_board = np.zeros((board_size, board_size))
        self.ones_board = np.ones((board_size, board_size))


    def generate_initial_state(self):
        board = np.zeros((3, self.board_size, self.board_size))
        board[0] = np.ones((self.board_size, self.board_size))
        return HexState(board, Players.WHITE)

    def get_legal_actions(self, state: HexState):
        actions = np.nonzero(state.board[0])
        return list(zip(actions[0], actions[1]))

    # May be a better way to check final result? Not disjoint forest, too slow to deepcopy
    def is_state_final(self, state):

        player_turn = self.switch_player_turn(state.player_turn)  # We only need to check if game is finished for the player who last placed a tile. We switch the current player in the state
        board = state.board[1] if player_turn == Players.WHITE else state.board[2]
        starts = ((0, x) for x in range(board.shape[1]) if board[0, x] == 1) if player_turn == Players.WHITE else ((y, 0) for y in range(board.shape[0]) if board[y, 0] == 1)
        for start in starts:
            if self.board_traversal_dfs(start, board, player_turn):
                return True
        return False

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
            if self.in_bounds(tile_coords) and tile_coords not in visited and board[tile_coords[0], tile_coords[1]] == 1:  # Check if neighbor tile belongs to player
                if self.board_traversal_dfs(tile_coords, board, player_turn, visited):
                    return True
        return False

    def play_action(self, action, state, inplace=False):
        # We use inplace to avoid instantiating a new hex state during rollouts
        if inplace:
            state.board[0, action[0], action[1]] = 0
            if state.player_turn == Players.WHITE:
                state.board[1, action[0], action[1]] = 1
            else:
                state.board[2, action[0], action[1]] = 1
                #TODO: permanent rotated board may be more efficient
                #state.board[2, action[1], (self.board_size - 1 - action[0])] = 1
            state.player_turn = self.switch_player_turn(state.player_turn)
        else:
            board = state.board.copy()
            board[0, action[0], action[1]] = 0
            if state.player_turn == Players.WHITE:
                board[1, action[0], action[1]] = 1
            else:
                board[2, action[0], action[1]] = 1
            return HexState(board, self.switch_player_turn(state.player_turn))

    def get_final_result(self, state):
        # TODO: should probably assert game over here, but more efficient to not
        # Need to return opposite of current turn, since player turn swapped after final move
        return Players.WHITE.value if state.player_turn == Players.BLACK else Players.BLACK.value

    def nn_state_representation(self, state):
        # TODO: features for bridge points
        """
        channel_black_bridge = np.array((2, 2))
        channel_white_bridge = np.array((2, 2))
        channel_save_bridge = np.array((2, 2))
        channel_form_bridge = np.array((2, 2))
        """
        #return np.concatenate((np.rot90(state.board[], axes=(2,1)), np.array([self.ones_board, self.zero_board])), axis=0)
        #return np.concatenate((np.array([state.board[0], state.board[1], np.rot90(state.board[2], axes=(1, 0))]), np.array([self.ones_board, self.zero_board])), axis=0) if state.player_turn == Players.WHITE else np.concatenate((np.array([state.board[0], state.board[1], np.rot90(state.board[2], axes=(1, 0))]), np.array([self.zero_board, self.ones_board])), axis=0)
        #return np.concatenate((np.array([state.board[1], np.rot90(state.board[2], axes=(1, 0))]), np.array([self.ones_board, self.zero_board])), axis=0) if state.player_turn == Players.WHITE else np.concatenate((np.array([state.board[1], np.rot90(state.board[2], axes=(1, 0))]), np.array([self.zero_board, self.ones_board])), axis=0)
        return np.concatenate((np.array([state.board[0], state.board[1], np.rot90(state.board[0], axes=(1, 0)), np.rot90(state.board[2], axes=(1, 0))]), np.array([self.ones_board, self.zero_board])), axis=0) if state.player_turn == Players.WHITE else np.concatenate((np.array([state.board[0], state.board[1], np.rot90(state.board[0], axes=(1, 0)), np.rot90(state.board[2], axes=(1, 0))]), np.array([self.zero_board, self.ones_board])), axis=0)



    def switch_player_turn(self, player_turn: Players):
        return Players.WHITE if player_turn == Players.BLACK else Players.BLACK

    def in_bounds(self, coords: tuple):
        # Assuming board equal dimensions
        return 0 <= coords[0] < self.board_size and 0 <= coords[1] < self.board_size

