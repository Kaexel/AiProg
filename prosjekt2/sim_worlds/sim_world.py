from enum import Enum


class GameState(Enum):
    GAME_OVER_WHITE_WINNER = 0
    GAME_OVER_BLACK_WINNER = 1
    GAME_OVER_TIE = 2
    PLAYING = 3


class Players(Enum):
    WHITE = 1
    BLACK = -1


class SimWorld:
    """
    Abstract class to represent simple 2-player board games as SimWorlds
    """
    def __init__(self, board_size: tuple):
        self.game_state = GameState.PLAYING
        self.player_turn = Players.WHITE
        self.board_size = board_size


    def get_legal_actions(self):
        raise NotImplementedError

    def is_current_state_final(self):
        raise NotImplementedError

    def play_action(self, action, inplace=True):
        raise NotImplementedError

    def get_sim_world_name(self):
        raise NotImplementedError

    def get_final_result(self):
        assert self.is_current_state_final()
        return self.player_turn.value

    def to_play(self):
        return self.player_turn

    def switch_player_turn(self):
        if self.player_turn == Players.WHITE:
            self.player_turn = Players.BLACK
        else:
            self.player_turn = Players.WHITE

    def get_game_state(self):
        return self.game_state


