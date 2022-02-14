import sys

import numpy as np
import random
from ActorCritic import Actor
from ActorCritic import Critic
from Hanoi import Hanoi
from PoleBalancing import PoleBalancing
from Gambler import Gambler
from enum import Enum
from SimWorld import SimWorld

import configparser
import matplotlib.pyplot as plt

available_sim_worlds = {0: PoleBalancing, 1: Hanoi, 2: Gambler}

config = configparser.ConfigParser()
config.read('config')
random.seed(config["PRIMARY"].getint('RANDOM_SEED'))

sim_type = config["PRIMARY"].getint('SIM_WORLD_TYPE')
max_episode_steps = config["PRIMARY"].getint('MAX_EPISODE_STEPS')
max_episodes = config["PRIMARY"].getint('MAX_EPISODES')
frame_delay = config["PRIMARY"].getint('FRAME_DELAY')
assert (sim_type in range(0, len(available_sim_worlds)))
sim_world = available_sim_worlds[sim_type]()


a = Actor(sim_world)
c = Critic(sim_world)


def in_episode_loop(actor, critic, sim_world):
    # 1. Do action, move system
    # ACTOR: action from policy
    actor.set_eligibility()
    pass


def rl_loop(actor: Actor, critic: Critic, sim_world: SimWorld):
    episode_success = {}
    for j in range(max_episodes):
        actor.reset_e()
        critic.reset_e()
        state = sim_world.produce_initial_state()
        possible_actions = sim_world.get_legal_actions()
        print(possible_actions)
        action = actor.get_action(state, possible_actions)
        t = 0  # TODO FJERN DENNE AXEL
        for i in range(max_episode_steps):
            state_next, reward = sim_world.apply_action(action)
            action_next = actor.get_action(state_next, possible_actions)
            actor.set_eligibility(state, action, 1)
            critic.update_td_error(reward, state, state_next)
            critic.set_eligibility(state, 1)
            print(reward)

            critic.update_state_action()
            critic.update_e()
            actor.update_state_action(critic.get_td_error())
            actor.update_e()

            state = state_next
            action = action_next
            """ Break out of episode if current state is final"""
            t = i
            if sim_world.is_current_state_final():
                break
        episode_success[j] = t
        sim_world.reset_world()

        if j % frame_delay == 0:
            print(reward)
            actor.update_epsilon()
            print(actor.epsilon)
    return episode_success


good = rl_loop(a, c, sim_world)

plt.xlabel("Episode #")
plt.ylabel("Number of steps in episode")
plt.title(f"Pole balance")

# plt.xlim([0, 15])
# plt.ylim([-2, 2])


x = good.keys()
y = good.values()
plt.plot(x, y)
#plt.legend(["Crowding entropy", "SGA entropy"])
plt.show()
