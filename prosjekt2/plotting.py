import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import RegularPolygon


"""
Various plotting functions. Might need a refactor.
"""

# Plots the board as lying hex
def plot_board(state):
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    colors = {0: 'cyan', -1: 'red', 1: 'green', 3: 'pink'}
    board = np.zeros_like(state.board[0])
    board[np.nonzero(state.board[1])] = 1
    board[np.nonzero(state.board[2])] = -1
    offset_x = 0
    offset_y = 0
    for row, tile_y in enumerate(board):
        for col, tile_x in enumerate(tile_y):
            hex = RegularPolygon((col + offset_x, -row + offset_y), numVertices=6, radius=math.sqrt(3) / 3., alpha=0.2, edgecolor='k', facecolor=colors[tile_x])
            ax.add_patch(hex)
        offset_x += 0.5
        offset_y += math.sqrt(3) / 12

    plt.autoscale(enable=True)
    plt.show()


# Plots the board as diamond hex
def plot_board_45(state):
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    colors = {0: 'cyan', -1: 'red', 1: 'green', 3: 'pink'}
    board = np.zeros_like(state.board[0])
    board[np.nonzero(state.board[1])] = 1
    board[np.nonzero(state.board[2])] = -1
    offset_x = 0
    offset_y = 0
    for row, tile_y in enumerate(board):
        row_offset_x = offset_x
        row_offset_y = offset_y
        for col, tile_x in enumerate(tile_y):
            hex = RegularPolygon((row_offset_x, row_offset_y), numVertices=6, radius=math.sqrt(3) / 3., alpha=0.2, edgecolor='k', facecolor=colors[tile_x])
            ax.add_patch(hex)
            row_offset_x += 0.5
            row_offset_y -= math.sqrt(3) / 2
        offset_x -= 0.5
        offset_y -= math.sqrt(3) / 2
    plt.autoscale(enable=True)
    plt.show()


# Returns the plot
def get_plot(state):
    fig, ax = plt.subplots(1)
    set_plot(state, ax)
    return fig, ax


# Used to fill GUI with lying hex
def set_plot(state, ax):
    ax.clear()
    ax.set_aspect('equal')
    colors = {0: 'cyan', -1: 'red', 1: 'green', 3: 'pink'}
    board = np.zeros_like(state.board[0])
    board[np.nonzero(state.board[1])] = 1
    board[np.nonzero(state.board[2])] = -1
    offset_x = 0
    offset_y = 0
    for row, tile_y in enumerate(board):
        for col, tile_x in enumerate(tile_y):
            hex = RegularPolygon((col + offset_x, -row + offset_y), numVertices=6, radius=math.sqrt(3) / 3., alpha=0.2, edgecolor='k', facecolor=colors[tile_x])
            ax.add_patch(hex)
        offset_x += 0.5
        offset_y += math.sqrt(3) / 12
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.autoscale(enable=True)

# Used to fill GUI with diamond hex
def set_plot_45(state, ax):
    ax.clear()
    ax.set_aspect('equal')
    colors = {0: 'cyan', -1: 'red', 1: 'green', 3: 'pink'}
    board = np.zeros_like(state.board[0])
    board[np.nonzero(state.board[1])] = 1
    board[np.nonzero(state.board[2])] = -1
    offset_x = 0
    offset_y = 0
    for row, tile_y in enumerate(board):
        row_offset_x = offset_x
        row_offset_y = offset_y
        for col, tile_x in enumerate(tile_y):
            hex = RegularPolygon((row_offset_x, row_offset_y), numVertices=6, radius=math.sqrt(3) / 3., alpha=0.2, edgecolor='k', facecolor=colors[tile_x])
            ax.add_patch(hex)
            row_offset_x += 0.5
            row_offset_y -= math.sqrt(3) / 2
        offset_x -= 0.5
        offset_y -= math.sqrt(3) / 2
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.autoscale(enable=True)

