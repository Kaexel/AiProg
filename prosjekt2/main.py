import configparser
import timeit
import time
import torch
import numpy as np
import math

import nn
from hex import Hex
from mcts import MCTS
from nim import Nim
from old_gold import OldGold
from sim_world import SimWorld
import random

random.seed(42)


def to_cuda(elements):
    """
    Transfers all parameters/tensors to GPU memory (cuda) if there is a GPU available
    """
    if not torch.cuda.is_available():
        return elements
    if isinstance(elements, tuple) or isinstance(elements, list):
        return [x.cuda() for x in elements]
    return elements.cuda()


"""
Setting up some primary variables
"""
available_sim_worlds = {0: Nim, 1: OldGold, 2: Hex}
config = configparser.ConfigParser()
config.read('config.ini')

sim_type = config["PRIMARY"].getint('SIM_WORLD_TYPE')
params = {}
for k, v in config[str(available_sim_worlds[sim_type].__name__).upper()].items():
    params[k] = int(v)

assert (sim_type in range(0, len(available_sim_worlds)))
sim_world = available_sim_worlds[sim_type](**params)

board_size = (5, 6)
nim = Nim(10, 5)
#nim = OldGold(8)


mcts = MCTS(sim_world)
t = time.time()
mcts.search(10000)
print(f"{(time.time() - t):.3}")




print(torch.version.cuda)
print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
