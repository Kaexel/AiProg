import random

import CANN
import CATT

from SimWorlds.Hanoi import Hanoi
from SimWorlds.PoleBalancing import PoleBalancing
from SimWorlds.Gambler import Gambler

import configparser
import matplotlib.pyplot as plt


"""
Setting up some primary variables
"""
available_sim_worlds = {0: PoleBalancing, 1: Hanoi, 2: Gambler}
config = configparser.ConfigParser()
config.read('config.ini')
random.seed(config["PRIMARY"].getint('RANDOM_SEED'))

sim_type = config["PRIMARY"].getint('SIM_WORLD_TYPE')

frame_delay = config["PRIMARY"].getint('FRAME_DELAY')
assert (sim_type in range(0, len(available_sim_worlds)))
sim_world = available_sim_worlds[sim_type]()

critic_type = config["PRIMARY"].getint("CRITIC_TYPE")


"""
Instantiating critic with or w/o NN
"""
if critic_type == 1:
    system = CANN.CANN(sim_world)
else:
    system = CATT.CATT(sim_world)


"""
Plotter for gambler
"""
def policy_plotter(p):
    plt.xlabel("State (amount of money)")
    plt.ylabel("Wager (optimal)")
    plt.title("Gambler")
    xy = {k: 0 for k in range(0, 100)}
    for k, v in p.items():

        if k[0] not in range(0, 100):
            continue
        if v > xy[k[0]]:
            xy[k[0]] = k[1]
    plt.plot(xy.keys(), xy.values())
    plt.show()


policy, success_step = system.rl_loop()


"""
Ensuring correct plot shown
"""
if sim_type == 2:
    policy_plotter(policy)
else:
    sim_world.visualize_best()
    plt.xlabel("Episode #")
    plt.ylabel("Number of steps in episode")
    plt.title(f"{sim_world.get_sim_world_name()}")
    x = success_step.keys()
    y = success_step.values()
    print(f"Max: {max(y)}\n"
          f"Min {min(y)}\n"
          f"Last: {list(y)[-1]}")
    plt.plot(x, y)
    plt.show()

