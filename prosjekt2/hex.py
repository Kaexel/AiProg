from disjoint_set import DisjointSetForest
from sim_world import SimWorld
from sim_world import Players
import numpy as np


neighbors = [(1, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 0)]
class Hex(SimWorld):
    def __init__(self, board_x: int, board_y: int):
        super().__init__((board_x, board_y))
        self.board = np.array([[(0, 0) for i in range(board_y)] for j in range(board_y)], dtype="i,i")
        self.board = np.array([[0 for i in range(board_y)] for j in range(board_y)])
        self.forest = DisjointSetForest()


    def get_legal_actions(self):
        actions = []
        for row, tile_y in enumerate(self.board):

            for col, tile_x in enumerate(tile_y):

                if tile_x.item() == (0, 0):
                    actions.append((row, col))

        return actions

    def is_current_state_final(self):
        return self.game_state

    def play_action(self, action):
        placement = (1, 0) if self.player_turn == Players.WHITE else (0, 1)  # Init placement indicator

        self.forest.make_set(action)
        for tile in neighbors:
            tile_coords = (action[0] + tile[0], action[1] + tile[1])
            if self.in_bounds(tile_coords):
                if tile_coords in self.forest.forest.keys():
                    print(f"UNION! {tile_coords}, {tile}")
                    self.forest.union(self.forest.forest[action], self.forest.forest[tile_coords])

        self.board[action[0]][action[1]] = placement
        self.switch_player_turn()
        # TODO: game over

    def get_sim_world_name(self):
        return "Hex"

    def channels(self):
        channels = []

    def print_board(self):
        for line in self.board:
            print(line)
        print("\n")

    def in_bounds(self, coords):
        size = self.board.shape
        if 0 <= coords[0] < size[0] and 0 <= coords[1] < size[0]:
            return True
        return False
