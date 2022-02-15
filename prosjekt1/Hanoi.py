from matplotlib import animation, collections
from matplotlib.collections import PatchCollection

from SimWorld import SimWorld
import configparser
import matplotlib.pyplot as plt
import random
import numpy as np

class Hanoi(SimWorld):

    def __init__(self):
        super().__init__()
        config = configparser.ConfigParser()
        config.read('config.ini')
        self.num_peg = config['HANOI'].getint('NUM_PEGS')
        self.num_discs = config['HANOI'].getint('NUM_DISCS')
        self.max_episode_steps = config['PRIMARY'].getint('MAX_EPISODE_STEPS')
        self.optimal_solution = 2 ** self.num_discs - 1
        self.state = []
        self.move_count = 0

        self.last_action = None
        self.state_history = []
        self.stupid_multiplier = 1
        """ plotting vars """
        self.colors = {k: (random.random(), random.random(), random.random()) for k in range(self.num_discs + 1)}
        self.animation = None

    def get_reward(self, action):
        reward = 0
        if self.last_action == action[::-1]:
            return 0 # * self.stupid_multiplier
            #self.stupid_multiplier += 10
        if self.is_current_state_final():
            #reward += 5 * (self.max_episode_steps / self.move_count)
            reward += max(200, 500 * (self.max_episode_steps / (self.move_count/2) ** 2))
            return reward

            if self.move_count <= 2**self.num_discs - 1:
                reward += 1500

        if self.produce_state_definition() in self.state_history:
            reward -= 25

        # Reward scales on number of discs on peg distance
        #reward += sum(i * len(v) for i, v in enumerate(self.state))
        # Reward scales on size of disc on peg distance
        reward += sum((i + 1) * disc for i, v in enumerate(self.state) for j, disc in enumerate(v))
        #print(reward)
        #print(reward)
        #reward += sum((i + 1) ** 2 * len(v) for i, v in enumerate(self.state))
        return reward

    """ SimWorld functions"""
    def is_current_state_final(self):
        return self.state[-1] == list(range(self.num_discs, 0, -1))

    def produce_initial_state(self):
        state = [[] for i in range(self.num_peg)]
        state[0] = list(range(self.num_discs, 0, -1))
        self.state = state
        self.state_history.append(self.produce_state_definition())
        return self.produce_state_definition()

    def get_legal_actions(self):
        l_a = []
        for idx_1, select_peg in enumerate(self.state):
            if not select_peg:  # Checking for empty peg
                continue
            top = select_peg[-1]
            for idx_2, place_peg in enumerate(self.state):
                if not place_peg or top < place_peg[-1]:  # If peg to place is empty or disc to place is smaller, this is a legal action
                    l_a.append((idx_1, idx_2))  # Move top disc from peg with idx_1 to peg with idx_2
        return tuple(l_a)

    def apply_action(self, action):
        self.state[action[1]].append(self.state[action[0]].pop())
        self.move_count += 1
        reward = self.get_reward(action)
        self.last_action = action
        self.state_history.append(self.produce_state_definition())
        return self.produce_state_definition(), reward

    def reset_world(self):
        self.stupid_multiplier = 1
        self.last_action = None
        self.state_history = []
        self.produce_initial_state()
        self.move_count = 0

    # Helper functions to ensure consistent state signature
    def produce_state_definition(self):
        return tuple(tuple(tuple(state) for state in self.state))

    def plot_world_state(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.xlim([0, self.num_peg + 1])
        plt.ylim([0, 1])
        plt.xticks([k + 1 for k in range(self.num_peg)], [chr(97 + i) for i in range(self.num_peg)])
        print(len(self.state_history))
        rects = self.init_animation()
        self.animation = matplotlib.animation.FuncAnimation(fig, self.animation_func, fargs=(rects,ax), frames=len(self.state_history), interval=350, blit=True, repeat=True)
        #self.animation.save("anim.gif")
        plt.show()

    def init_animation(self):
        rects = {}
        for i, peg in enumerate(self.state_history[0]):
            for j, disc in enumerate(peg):
                width = 1 / self.num_discs * disc
                x = i - width / 2
                rect1 = matplotlib.patches.Rectangle((1 + x, (j * 0.06)), width, 0.05, color=self.colors[disc - 1])
                rects[disc] = rect1
        print(rects)
        return rects

    def animation_func(self, i, rects, ax):

        for i, peg in enumerate(self.state_history[i]):
            for j, disc in enumerate(peg):
                width = 1 / self.num_discs * disc
                x = i - width / 2
                rects[disc].set(xy=(1 + x, (j * 0.06)))
        t = [rect for rect in rects.values()]
        print(t)
        ax.add_collection(PatchCollection(t, animated=True))
        return t







"""
h = Hanoi()
h.produce_initial_state()

h.plot_world_state()
actions = h.get_legal_actions()
h.apply_action(actions[0])

h.plot_world_state()

actions = h.get_legal_actions()
h.apply_action(actions[0])

h.plot_world_state()

h.apply_action((1, 2))
h.plot_world_state()

print(actions)
"""
