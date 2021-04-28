import random
import numpy as np
import copy
import math


class Node:
    def __init__(self, game, action, state, player):
        self.n = 0
        self.value = 0
        self.children = []
        self.state = state
        self.game = game
        self.player = player  # who makes the next move
        self.action = action  # last move that created this node

    def is_terminal(self):
        return self.game.has_won(self.state, player=-1) or self.game.is_draw(self.state)

    def get_winner(self):
        return self.game.winner(self.state) * self.player

    def is_leaf(self):
        return not self.children

    def create_child(self, action):
        self.children.append(Node(self.game, action,
                                  state=self.game.move(self.state, action, 1).copy() * -1,
                                  player=self.player * -1))

    def expand(self):
        for action in self.game.get_legal_moves(self.state):
            self.create_child(action)

    def rollout(self):
        sim_state = copy.deepcopy(self.state)
        player_swap = 1
        while True:
            if self.game.has_won(sim_state):
                return player_swap
            elif self.game.is_draw(sim_state):
                return 0
            else:
                column = random.choice(self.game.get_legal_moves(sim_state))
                sim_state = self.game.move(sim_state, column, 1) * -1
                player_swap *= -1

    def select_child_ran(self, c):
        ubcs = np.array([self.upper_confidence_bound(child.value, self.n, child.n, c) for child in self.children])
        ubc_min = np.min(ubcs) - 0.00001
        ubcs -= ubc_min
        ubc_sum = np.sum(ubcs)
        ran = random.random() * ubc_sum
        i = 0
        while ran > ubcs[i]:
            ran -= ubcs[i]
            i += 1
        return self.children[i]

    def select_child_max(self, c):
        ubcs = np.array([self.upper_confidence_bound(child.value, self.n, child.n, c) for child in self.children])
        ubc_max = np.max(ubcs)
        max_children = [child for i, child in enumerate(self.children) if (ubcs[i] == ubc_max)]
        return random.choice(max_children)

    @staticmethod
    def upper_confidence_bound(value, parent_n, n, c, epsilon=0.00000001):
        return value + c * math.sqrt(math.log(parent_n + epsilon) / (n + epsilon))
