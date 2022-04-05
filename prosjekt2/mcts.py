import copy
import math
from random import choice

from game_managers.game_manager import GameManager
from game_managers.hex_manager import HexManager
import multiprocessing as mp

class RandomPolicy:
    def get_action(self, state, mgr: GameManager):
        actions = tuple(mgr.get_legal_actions(state))
        return choice(actions)


class Node:
    def __init__(self, state, parent=None, move=None, terminal_node=False):
        self.E = 0  # Value of node (sum of results)
        self.N = 0  # Number of times node visited
        self.parent: Node = parent
        self.children = {}  # Maps actions to nodes

        self.move = move
        self.state = state

        self.terminal_node = terminal_node
        self.max_expansion = terminal_node

    def add_child(self, child, move):
        self.children[move] = child

    def get_q(self):
        if self.N == 0:
            return 1000000
        else:
            return self.E / self.N

    def get_uct(self, exploration_c):
        if self.N == 0:
            return 1000000
        return exploration_c * math.sqrt(math.log(self.parent.N) / (self.N))

    def __str__(self):
        s = [f"Total reward: {self.E}\t", f"Number of visits: {self.N}\t",
             f"Expected reward:  {self.get_q():.2}\t",
             f"Terminal: {self.terminal_node}\t",
             f"Move that led here: {self.move}\t",
             f"Possible actions: {self.children.keys()}"]
        return f"{self.__class__.__name__}: {''.join(s)}"


class MCTS:
    def __init__(self, game_manager=HexManager(7), policy_object=RandomPolicy()):

        self.game_manager: GameManager = game_manager
        self.root = Node(game_manager.generate_initial_state())
        self.nodes = []
        self.node_count = 1
        self.rollout_count = 0
        self.policy_object = policy_object

    def parallel_search(self, max_rollouts):
        with mp.Pool(8) as pool:
            nodes = pool.apply_async(self.search, (100,))
            print(nodes.get(100))


    def search(self, max_rollouts):
        for i in range(max_rollouts):
            self.single_pass()
        #self.print_stats()

        #return self.gen_distribution()
        return {key: v.N for key, v in self.root.children.items()}

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
        actions = self.game_manager.get_legal_actions(node.state)
        for action in actions:
            if action not in node.children:
                state = self.game_manager.play_action(action, node.state)
                child = Node(copy.copy(state), node, action, terminal_node=self.game_manager.is_state_final(state))
                self.node_count += 1
                node.children[action] = child
                #self.nodes.append(child)
                if len(actions) == len(node.children):
                    node.max_expansion = True
                return child

    def rollout(self, state):
        playout_state = copy.copy(state)
        move_count = 0
        while not self.game_manager.is_state_final(playout_state):
            action_selected = self.policy_object.get_action(playout_state, self.game_manager)
            self.game_manager.play_action(action_selected, playout_state, inplace=True)
            move_count += 1

        self.rollout_count += 1
        winner = self.game_manager.get_final_result(playout_state)  # Maybe return value?
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
            val = player_turn.value * child.get_q() + child.get_uct(2)
            if val > best_value:
                best_value = val
                best_children = [child]
            elif val == best_value:
                best_children.append(child)

        return choice(best_children)

    def gen_distribution(self):
        dist = []
        root_n = self.root.N
        for child in self.root.children.values():
            dist.append(child.N/root_n)
        return dist

    def update_root(self, action):
        self.rollout_count = 0
        self.root = self.root.children[action]
        self.root.parent = None
        self.root.children = {}
        self.root.max_expansion = False
        #self.reset_nodes()

    def reset_nodes(self):
        for node in self.nodes:
            node.N = 0
            #node.E = 0

    def print_stats(self):
        print(f"Used nodes: {self.node_count}\n"
              f"Rollouts performed:  {self.rollout_count}")
