import glob
import os
import tkinter
from enum import Enum
from tkinter import *
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow import keras

import nn
import plotting
from game_managers.hex_manager import HexManager

# TODO: make fancy GUI
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class GameGUI:
    def __init__(self):
        self.root = Tk()

        self.board_figure = plt.figure(figsize=(4, 4))

        self.board_sub = self.board_figure.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.board_figure, self.root)

        self.text_box = Label(self.root, text="Cool gui 😎")
        self.text_box.pack(side=TOP, fill='both', expand=False)
        #frame = Frame(root)
        #frame.pack()
        self.canvas.get_tk_widget().pack()
        button = Button(self.root,
                   text="Show current game",
                   fg="green",
                   command=self.toggle_board)
        button.pack(side=LEFT)
        button2 = Button(self.root,
                   text="Quit",
                   fg="red",
                   command=quit)
        button2.pack(side=RIGHT)



    def get_board_state(self):
        return self.board_figure.get_visible()

    def toggle_board(self):
        self.board_figure.set_visible(not self.board_figure.get_visible())
        self.canvas.draw_idle()

    def update_title(self, text):
        self.text_box.config(text=text)
        self.update_gui()

    def update_plot(self, state):
        plotting.set_plot_45(state, self.board_sub)
        self.canvas.draw()
        self.update_gui()

    def update_gui(self):
        self.root.update_idletasks()
        self.root.update()


class TourneyState(Enum):
    pass


class TournamentGUI:
    def __init__(self, num_games, model_paths: list):
        self.root = Tk()
        self.root.geometry('600x400')

        self.text_box = Label(self.root, text="Cool gui 😎", background="green")
        self.text_box.pack(side=TOP, fill='both', expand=False)
        self.button_frame = Frame(self.root, background="pink")

        self.button_frame.pack(side=BOTTOM, fill='both', expand=False)

        self.board_figure = plt.figure(figsize=(3, 4))

        self.board_sub = self.board_figure.add_subplot(111)

        self.board_label_frame = LabelFrame(self.root, text="Board state", background="bisque")
        self.canvas = FigureCanvasTkAgg(self.board_figure, self.board_label_frame)
        self.canvas.get_tk_widget().pack()
        self.board_label_frame.pack(side=RIGHT, fill='both', expand=True)


        self.info_label_frame = LabelFrame(self.root, text="info", width=100, height=90, background="blue")
        self.info_text = Text(self.info_label_frame)
        self.info_text.insert(1.0, "skadooshasjdkljasopkejdqwoije\nqwpej\nqwopiejqw\nqweqw")
        self.info_text.pack(fill=None, expand=False)
        self.info_label_frame.pack(side=LEFT, fill='both', expand=True)

        self.button_state = tkinter.IntVar()


        # TODO: make this less of a mess
        self.num_games = num_games
        self.manager = HexManager(5)
        self.state = self.manager.generate_initial_state()
        self.game_in_series = 0

        self.models = self.load_models(model_paths)
        self.model_l = self.models[0]
        self.model_r = self.models[1]
        self.model_l_idx = 0
        self.model_r_idx = 1

        self.model_l_played_series = 0

        self.cur_mod = self.model_l
        self.waiting_mod = self.model_r

        self.switch_sides_next = True

        self.win_count_1 = 0
        self.win_count_2 = 0
        self.scores = {}



        #frame = Frame(root)
        #frame.pack()


        button = Button(self.button_frame,
                   text="Show current game",
                   fg="green",
                   command=self.toggle_board)
        button.pack(side=LEFT)
        button2 = Button(self.button_frame,
                   text="Play through current series",
                   fg="red",
                   command=self.play_through_series)
        button2.pack(side=RIGHT)

        button3 = Button(self.button_frame,
                   text="Play through current game",
                   fg='deep sky blue',
                   command=self.play_through_game)
        button3.pack(side=RIGHT)

        button4 = Button(self.button_frame,
                   text="Next move",
                   fg="azure4",
                   command=self.next_move)
        button4.pack(side=RIGHT)


    def next_move(self):
        self.update_info()
        nn_move = self.get_move(self.state)
        #self.state.set(1)
        self.manager.play_action(nn_move, self.state, True)
        self.update_plot(self.state)
        if self.manager.is_state_final(self.state):
            self.reset_game()
        self.cur_mod, self.waiting_mod = self.waiting_mod, self.cur_mod


    def play_through_series(self):
        self.finish_series()

    def play_through_game(self):
        result = self.finish_game()
        self.state = self.manager.generate_initial_state()

    def toggle_board(self):
        self.board_figure.set_visible(not self.board_figure.get_visible())
        self.canvas.draw_idle()

    def update_title(self, text):
        self.text_box.config(text=text)
        self.update_gui()

    def update_info(self):
        q = f"Current game:\n" \
            f"{self.model_l[0]} vs. {self.model_r[0]}\n" \
            f"Score: {self.win_count_1}-{self.win_count_2}\n" \
            f"Log:\n"
        self.info_text.delete(1.0, 5.0)
        self.info_text.insert(1.0, q)
        self.update_gui()

    def update_plot(self, state):
        plotting.set_plot_45(state, self.board_sub)
        self.canvas.draw()
        self.update_gui()

    def update_gui(self):
        self.root.update_idletasks()
        self.root.update()

    def load_models(self, models):
        loaded_models = [nn.LiteModel.from_keras_model(keras.models.load_model(model)) for model in models]
        models = [model.split("\\")[-1] for model in models]
        models_list = list(zip(models, loaded_models))
        return models_list


    def reset_game(self):
        self.game_in_series += 1
        # Update winner of previous game
        if self.cur_mod == self.model_l:
            self.win_count_1 += 1
        else:
            self.win_count_2 += 1

        self.update_info()

        if self.game_in_series >= self.num_games:
            self.update_players()
            self.game_in_series = 0
            self.win_count_1 = 0
            self.win_count_2 = 0


        self.cur_mod, self.waiting_mod = self.model_l, self.model_r
        self.state = self.manager.generate_initial_state()
        self.update_plot(self.state)

    # Updates current player and makes sure everyone plays each other
    def update_players(self):

        self.model_l_played_series += 1
        text = f"{self.model_l[0]} vs. {self.model_r[0]}\n" \
               f"Score: {self.win_count_1}-{self.win_count_2}\n"
        self.info_text.insert(5.0, text)

        if self.model_l_idx == len(self.models) - 1 and self.model_l_played_series == len(self.models) - 1:
            text = f"Tourney finished!\n"
            self.info_text.insert(1.0, text)
            for w in self.button_frame.winfo_children():
                w.configure(state="disabled")
            return

        if self.model_l_played_series >= len(self.models) - 1:
            self.model_l_idx += 1
            self.model_l = self.models[self.model_l_idx]
            self.model_r_idx = 0
            self.model_r = self.models[self.model_r_idx]
            self.model_l_played_series = 0
        else:

            self.model_r_idx = (self.model_r_idx + 1) % len(self.models)
            if self.model_r_idx == self.model_l_idx:
                self.model_r_idx += 1

            self.model_r = self.models[self.model_r_idx]


    def finish_game(self):
        while not self.manager.is_state_final(self.state):
            nn_move = self.cur_mod[1].get_action(self.state, self.manager)

            self.manager.play_action(nn_move, self.state, True)
            if self.manager.is_state_final(self.state):
                self.reset_game()
                return
            self.cur_mod, self.waiting_mod = self.waiting_mod, self.cur_mod


    def finish_series(self):
        for _ in range(self.num_games - self.game_in_series):
            self.finish_game()

    def get_move(self, state):
        return self.cur_mod[1].get_action(state, self.manager)


#g = GameGUI()
#g.update_plot(state)
#manager.play_action((0,0), state, True)
#g.update_plot(state)
#g.root.mainloop()
