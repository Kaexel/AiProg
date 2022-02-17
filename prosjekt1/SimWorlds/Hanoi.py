from matplotlib import animation
from SimWorlds.SimWorld import SimWorld
import configparser
import matplotlib.pyplot as plt
import random
import matplotlib


class Hanoi(SimWorld):
    """
    Class implementing the Hanoi Tower SimWorld
    """
    def __init__(self):
        super().__init__()
        self.name = "Towers of Hanoi"
        config = configparser.ConfigParser()
        config.read('config.ini')
        self.num_peg = config['HANOI'].getint('NUM_PEGS')
        self.num_discs = config['HANOI'].getint('NUM_DISCS')
        self.max_episode_steps = config['PRIMARY'].getint('MAX_EPISODE_STEPS')
        self.optimal_solution = 2 ** self.num_discs - 1
        self.state = []
        self.move_count = 0

        self.state_history = []

        """ plotting vars """
        self.colors = {k: (random.random(), random.random(), random.random()) for k in range(self.num_discs + 1)}
        self.animation = None
        self.rects = None

    def get_reward(self, action):
        """
        I tried various reward schemes, but found that the best was just to reward it for finishing,
        scaling with inverse of move count. No negative rewards work best.

        """
        # Penalty for repeating last action
        #if self.last_action == action[::-1]:
        #    return -1

        if self.is_current_state_final():
            return 500 * (self.max_episode_steps / self.move_count)
        reward = 0
        # Different attempted reward methods
        # Reward scales on number of discs on peg distance
        #reward += sum(i * len(v) for i, v in enumerate(self.state))
        # Reward scales on size of disc on peg distance
        #reward += sum((i + 1) * disc for i, v in enumerate(self.state) for j, disc in enumerate(v))
        return reward

    """ SimWorld functions"""
    def get_sim_world_name(self):
        return self.name

    def is_current_state_final(self):
        """
        State is only final if solved
        :return: True if all discs in correct order on final peg. Else False
        """
        return self.state[-1] == list(range(self.num_discs, 0, -1))

    def produce_initial_state(self):
        """
        Produces a state with all discs on leftmost peg
        :return: a tuple of num_peg tuples
        """
        state = [[] for i in range(self.num_peg)]
        state[0] = list(range(self.num_discs, 0, -1))
        self.state = state
        self.state_history.append(self.produce_state_definition())
        self.rects = self.init_anim_rects()
        return self.produce_state_definition_nn()

    def get_legal_actions(self):
        l_a = []
        for idx_1, select_peg in enumerate(self.state):
            if not select_peg:  # Checking for empty peg
                continue
            top = select_peg[-1] # Top disc
            for idx_2, place_peg in enumerate(self.state):
                if not place_peg or top < place_peg[-1]:  # If peg to place is empty or disc to place is smaller, this is a legal action
                    l_a.append((idx_1, idx_2))  # Move top disc from peg with idx_1 to peg with idx_2
        return tuple(l_a)

    def apply_action(self, action):
        self.state[action[1]].append(self.state[action[0]].pop())
        self.move_count += 1
        reward = self.get_reward(action)
        self.state_history.append(self.produce_state_definition())
        return self.produce_state_definition_nn(), reward

    def reset_world(self):
        self.state_history = []
        self.produce_initial_state()
        self.move_count = 0

    def plot_world_state(self):
        """
        Creates a gif animation of current episode
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.xlim([0, self.num_peg + 1])
        plt.ylim([0, 1])
        plt.xticks([k + 1 for k in range(self.num_peg)], [chr(97 + i) for i in range(self.num_peg)])
        ax.add_patch(self.rects[1])
        for rect in self.rects.values():
            ax.add_patch(rect)
        self.animation = matplotlib.animation.FuncAnimation(fig, self.animation_func, init_func=self.init_anim, frames=len(self.state_history), interval=350, blit=True, repeat=True)
        self.animation.save("anim.gif")
        plt.clf()

    def visualize_best(self):
        pass

    # Helper functions to ensure consistent state signature
    def produce_state_definition(self):
        """
        Produces a hashable tuple state
        """
        return tuple(tuple(tuple(state) for state in self.state))

    def produce_state_definition_nn(self):
        """
        Produces a state more suitable for NN
        """
        nn_state = []
        for peg in self.state:
            nn_peg = [0] * self.num_discs
            for j, disc in enumerate(peg):
                nn_peg[disc - 1] = 1
            nn_state.append(nn_peg)
        return tuple(tuple(tuple(state) for state in nn_state))

    """
    Functions for matplotlib function animation
    """
    def init_anim(self):
        for rect in self.rects.values():
            rect.set_visible(False)
        return list(self.rects.values())

    def animation_func(self, i):
        for i, peg in enumerate(self.state_history[i]):
            for j, disc in enumerate(peg):
                width = 1 / self.num_discs * disc
                x = i - width / 2
                self.rects[disc].set_xy((1 + x, (j * 0.06)))
                self.rects[disc].set_visible(True)
        return self.rects.values()

    def init_anim_rects(self):
        rects = {}
        for i, peg in enumerate(self.state_history[0]):
            for j, disc in enumerate(peg):
                width = 1 / self.num_discs * disc
                x = i - width / 2
                rect1 = matplotlib.patches.Rectangle((1 + x, (j * 0.06)), width, 0.05, color=self.colors[disc - 1])
                rect1.set_visible(False)
                rects[disc] = rect1
        return rects
