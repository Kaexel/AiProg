from tkinter import *
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import plotting
from game_managers.hex_manager import HexManager

# TODO: make fancy GUI

class GameGUI:
    def __init__(self):
        self.root = Tk()
        self.board_figure = plt.figure(figsize=(4, 4))

        self.board_sub = self.board_figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.board_figure, self.root)

        self.text_box = Label(self.root, text="Cool gui 😎")
        self.text_box.pack(side=TOP)
        #frame = Frame(root)
        #frame.pack()
        self.canvas.get_tk_widget().pack()
        button = Button(self.root,
                   text="Show current game",
                   fg="green",
                   command=self.toggle_board)
        button.pack(side=LEFT)
        button2 = Button(self.root,
                   text="quit",
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


#g = GameGUI()
#g.update_plot(state)
#manager.play_action((0,0), state, True)
#g.update_plot(state)
#g.root.mainloop()
