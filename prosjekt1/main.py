import math
import sys

import numpy as np
import random
from ActorCritic import Actor
from Critic import Critic, CriticNN
from Hanoi import Hanoi
from PoleBalancing import PoleBalancing
from Gambler import Gambler
from enum import Enum
from SimWorld import SimWorld

import configparser

import matplotlib.pyplot as plt

available_sim_worlds = {0: PoleBalancing, 1: Hanoi, 2: Gambler}
plt.interactive(False)
config = configparser.ConfigParser()
config.read('config.ini')
random.seed(config["PRIMARY"].getint('RANDOM_SEED'))

sim_type = config["PRIMARY"].getint('SIM_WORLD_TYPE')
max_episode_steps = config["PRIMARY"].getint('MAX_EPISODE_STEPS')
max_episodes = config["PRIMARY"].getint('MAX_EPISODES')
frame_delay = config["PRIMARY"].getint('FRAME_DELAY')
assert (sim_type in range(0, len(available_sim_worlds)))
sim_world = available_sim_worlds[sim_type]()

critic_type = config["PRIMARY"].getint("CRITIC_TYPE")

a = Actor(sim_world)
if critic_type == 0:
    c = Critic()
else:
    c = CriticNN()


def result_plotter(p):
    if sim_type == 0:
        return
    elif sim_type == 1:
        return
    elif sim_type == 2:
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


def rl_loop(actor: Actor, critic: Critic, sim_world: SimWorld):
    episode_success = {}
    for j in range(max_episodes):
        actor.reset_e()
        critic.reset_e()
        state = sim_world.produce_initial_state()
        print(f"Episode # {j}")
        possible_actions = sim_world.get_legal_actions()
        action = actor.get_action(state, possible_actions)
        t = 0  # TODO FJERN DENNE AXEL
        current_episode_sa = {(state, action)}
        if j == max_episodes - 1:
            # Get best actions for final episode
            actor.max_greed()
        for i in range(max_episode_steps):

            if j == max_episodes - 1:
                #sim_world.plot_world_state()
                pass
            if j % frame_delay == 0:
                #sim_world.plot_world_state()
                pass

            state_next, reward = sim_world.apply_action(action)
            possible_actions = sim_world.get_legal_actions()
            action_next = actor.get_action(state_next, possible_actions)
            actor.set_eligibility(state, action, 1)
            critic.update_td_error(reward, state, state_next)
            critic.set_eligibility(state, 1)

            critic.update_state_action(current_episode_sa)
            critic.update_e(current_episode_sa)
            actor.update_state_action(current_episode_sa, critic.get_td_error())
            actor.update_e(current_episode_sa)

            state = state_next
            action = action_next
            current_episode_sa.add((state, action))
            """ Break out of episode if current state is final"""
            t += 1
            if sim_world.is_current_state_final():
                break

            #print(actor.eligibilities)

        if j % frame_delay == 0:
            print(f"Episode # {j}")
            #sim_world.plot_world_state()

        if j == max_episodes - 1:
            print(f"Episode # {j}")
            sim_world.plot_world_state()

        episode_success[j] = t
        sim_world.reset_world()

        if j % max_episodes/10 == 0:  # This ensures a nice epsilon degradation. If it begins at 0.5, it ends at 0.001
            actor.update_epsilon()

    #print(a.policy)
    return a.policy, episode_success


policy, success_step = rl_loop(a, c, sim_world)
print(success_step)
result_plotter(policy)



plt.xlabel("Episode #")
plt.ylabel("Number of steps in episode")
plt.title(f"Pole balance")

# plt.xlim([0, 15])
#plt.ylim([0, 25])


x = success_step.keys()
y = success_step.values()
print(x)
print(y)
print(min(y))
print(max(y))
plt.plot(x, y)
#plt.legend(["Crowding entropy", "SGA entropy"])
plt.show()

