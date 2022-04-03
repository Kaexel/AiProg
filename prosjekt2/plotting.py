import math

import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon

from sim_worlds.hex import Hex


def plot_board(state: Hex):
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    colors = {0: 'cyan', -1: 'red', 1: 'green', 3: 'pink'}
    board = state.board
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
