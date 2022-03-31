import math
from copy import deepcopy
from random import choice

from sim_worlds.hex import Hex
from sim_worlds.sim_world import SimWorld


class Node:
    def __init__(self, state: SimWorld, parent=None, move=None):  # Can maybe init without whole simworld object?
        self.E = 0  # Value of node (sum of results)
        self.N = 0  # Number of times node visited
        self.parent: Node = parent
        self.children = {}  # Maps actions to nodes
        self.s_a = {}
        self.move = move
        self.state: SimWorld = state

        self.terminal_node = self.state.is_current_state_final()
        self.max_expansion = self.terminal_node

    def add_child(self, child, move):
        self.children[move] = child

    def get_q(self):
        c = 1
        if self.N == 0:
            return float("inf")
        else:
            return self.E / self.N

    def get_uct(self, exploration_c):
        if self.N == 0:
            return float("inf")
        return exploration_c * math.sqrt(math.log(self.parent.N) / (1 + self.N))

    def __str__(self):
        s = [f"Total reward: {self.E}\t", f"Number of visits: {self.N}\t",
             f"Expected reward:  {self.get_q():.2}\t",
             f"Terminal: {self.terminal_node}\t",
             f"Move that led here: {self.move}\t",
             f"Possible actions: {self.children.keys()}"]
        return f"{self.__class__.__name__}: {''.join(s)}"


class MCTS:
    def __init__(self, state=Hex(4, 4)):

        self.initial_state: SimWorld = state
        self.root = Node(deepcopy(state))
        self.node_count = 1
        self.rollout_count = 0

    def search(self, max_rollouts):
        for i in range(max_rollouts):
            self.single_pass()
        self.print_stats()
        print(self.root)
        for child in self.root.children.values():
            print(child)

    def single_pass(self):
        node = self.select_node(self.root)
        result = self.rollout(node.state)
        self.backpropagate(node, result)

    def select_node(self, node):
        while not node.terminal_node:           # Search for a (current) leaf node
            if node.max_expansion:              # If node is fully expanded, we go to its best child
                node = self.best_action(node)
            else:
                return self.expand(node)        # If node not fully expanded, expand it with a random unexplored action
        return node

    def expand(self, node):  # Expands one node at a time
        actions = node.state.get_legal_actions()
        for action in actions:
            if action not in node.children:
                new_state: SimWorld = node.state.play_action(action)
                child = Node(new_state, node, action)
                self.node_count += 1
                node.children[action] = child
                if len(actions) == len(node.children):
                    node.max_expansion = True
                return child

    def rollout(self, state: SimWorld):
        state = deepcopy(state)
        while not state.is_current_state_final():
            actions = state.get_legal_actions()
            action_selected = choice(actions)  # TODO: get action from NN
            state = state.play_action(action_selected, inplace=True)

        self.rollout_count += 1
        winner = state.get_final_result()  # Maybe return value?
        return winner

    def backpropagate(self, node: Node, result):
        while node is not None:
            node.N += 1
            node.E += result
            node = node.parent

    def best_action(self, node):
        player_turn = node.state.player_turn
        best_children = []
        best_value = float("-inf")
        for child in node.children.values():
            val = player_turn.value * child.get_q() + child.get_uct(1)
            if val > best_value:
                best_value = val
                best_children = [child]
            elif val == best_value:
                best_children.append(child)

        return choice(best_children)

    def update_root(self, action):
        self.root = self.root.children[action]

    def print_stats(self):
        print(f"Used nodes: {self.node_count}\n"
              f"Rollouts performed:  {self.rollout_count}")
