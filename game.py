import numpy as np
import math
import scipy.stats
import copy
import random
import time

from gui import Board
from nnet import NNet

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Game:
    def __init__(self, rows=6, columns=7, win_con=4, gui=False):
        self.rows = rows
        self.columns = columns
        self.game_state = np.zeros((columns, rows))
        self.win_con = win_con
        self.player = 1
        self.gui = gui
        if self.gui:
            self.board = Board(self, rows, columns)
            self.board.root.update()

    def make_move(self, column):
        self.game_state = self.move(self.game_state, column, self.player)
        self.player *= -1
        if self.gui:
            self.board.redraw()
            self.board.root.update()

    def run(self, method1, method2, structure_1='structure1', structure_2='structure1', print_out=True,
            runs1=200, runs2=200, c1=3, c2=3, pause=1):
        while self.winner(self.game_state) == 'none':
            if self.player == 1:
                self.make_move(
                    self.decide_move(method=method1, runs=runs1, c=c1, structure=structure_1, print_out=print_out))
            else:
                self.make_move(
                    self.decide_move(method=method2, runs=runs2, c=c2, structure=structure_2, print_out=print_out))
            time.sleep(pause)
        if self.gui:
            self.board.root.mainloop()

    def decide_move(self, method, c_puct=3, c=10, runs=200, structure='structure1', print_out=True):
        if method == 'input':
            while True:
                try:
                    move = int(input('input column number between 1 and 7: ')) - 1
                    if move in self.get_legal_moves(self.game_state):
                        return move
                except ValueError:
                    pass
                print('input not valid')

        if method == 'mcts':
            pi = self.tree_search(runs=runs, c=c)
            return np.argmax(pi)

        if method == 'nnet':
            pi = self.tree_search_nnet(nnet=NNet(structure=structure), c_puct=c_puct, runs=runs, print_out=print_out)
            return np.argmax(pi)

        if method == 'simple_nnet':
            policy, val = NNet().prediction(self.game_state, player=self.player)
            return np.argmax(policy)

        else:
            print('method not valid')

    def move(self, state, column, player):
        state_copy = copy.deepcopy(state)
        for i in range(self.rows):
            if state_copy[column][i] == 0:
                state_copy[column][i] = player
                break
        return state_copy

    def get_legal_moves(self, state):
        moves = []
        for column in range(self.columns):
            if state[column][self.rows - 1] == 0:
                moves.append(column)
        return moves

    def on_board(self, column, row):
        return self.columns > column >= 0 and self.rows > row >= 0

    def winner(self, state):
        if not self.get_legal_moves(state):
            return 0
        for col in range(self.columns):
            for row in range(self.rows):
                if state[col][row] != 0:
                    for direction in np.array([[0, 1], [1, 0], [1, -1], [1, 1]]):
                        count = 0
                        field = np.array([col, row])
                        while self.on_board(field[0], field[1]) and state[field[0]][field[1]] == state[col][row]:
                            count += 1
                            field = field + direction

                        if count >= self.win_con:
                            return state[col][row]
        return 'none'

    def has_won(self, state, player=1):
        for col in range(self.columns):
            directions = np.array([[0, 1]])
            if col <= (self.columns - self.win_con):
                directions = np.array([[0, 1], [1, 0], [1, -1], [1, 1]])
            for row in range(self.rows):
                if state[col][row] == player:
                    for direction in directions:
                        count = 0
                        field = np.array([col, row])
                        while self.on_board(field[0], field[1]) and state[field[0]][field[1]] == player:
                            count += 1
                            field = field + direction
                        if count >= self.win_con:
                            return True
        return False

    def is_draw(self, state):
        return not self.get_legal_moves(state)

    def tree_search(self, runs, temp=1, c=10):
        start_node = Node(self, -1, copy.deepcopy(self.game_state) * self.player, self.player)
        start_node.n = 1
        for i in range(runs):
            current_node, path = self.get_to_leaf(start_node, c)
            if current_node.is_terminal():
                value = -1  # * 2
            elif current_node.n == 0:
                value = current_node.rollout()
            else:
                current_node.expand()
                current_node = random.choice(current_node.children)
                path.append(current_node)
                value = current_node.rollout()
            for node in reversed(path):
                node.value += -value
                node.n += 1
                value *= -1
        return self.calc_pi(start_node=start_node, temp=temp)

    def tree_search_nnet(self, runs, nnet, c_puct=4, temp=1,
                         randomness=False, x_noise=0., decision='policy', print_out=False):
        start_node = Node(self, -1, copy.deepcopy(self.game_state) * self.player, self.player)
        start_node.n = 1
        for i in range(runs):
            current_node, path = self.get_to_leaf_nnet(start_node, c_puct, randomness)
            if current_node.is_terminal():
                value = 0
                current_node.n += 1
                current_node.nnet_value = 0
            else:
                if current_node == start_node:
                    current_node.fetch_prediction(nnet, x_noise=x_noise)
                else:
                    current_node.fetch_prediction(nnet)
                current_node.expand()
                current_node.n = 1
                value = current_node.nnet_value
            for node in reversed(path):
                value = 1 - value
                node.update_value(value)
        if print_out:
            print(f'estimated chance to win: {1 - start_node.select_child_nnet(0).nnet_value}')
        if decision == 'policy':
            return self.calc_pi(start_node=start_node, temp=temp)
        else:
            values = [1 for _ in range(7)]
            for child in start_node.children:
                values[child.action] = child.nnet_value
            return (1 - np.array(values)) / np.sum(1 - np.array(values))

    @staticmethod
    def get_to_leaf_nnet(start_node, c_puct, randomness):
        current_node = start_node
        path = []
        while not current_node.is_leaf():
            path.append(current_node)
            current_node = current_node.select_child_nnet(c_puct, randomness)
        return current_node, path

    @staticmethod
    def get_to_leaf(start_node, c):
        current_node = start_node
        path = [current_node]
        if not current_node.is_leaf():
            current_node = current_node.select_child_ran(c)
            path.append(current_node)
        while not current_node.is_leaf():
            current_node = current_node.select_child_max(c)
            path.append(current_node)
        return current_node, path

    def calc_pi(self, start_node, temp):
        pi = np.zeros(self.columns)
        for child in start_node.children:
            pi[child.action] = child.n ** temp
        return pi / np.sum(pi)


class Node:
    def __init__(self, game, action, state, player):
        self.n = 0
        self.value = 0
        self.children = []
        self.state = state
        self.game = game
        self.player = player  # who makes the next move
        self.action = action  # last move that created this node

        self.policy = None
        self.nnet_value = 0.5

    def fetch_prediction(self, nnet, x_noise=0.):
        self.policy, self.nnet_value = nnet.prediction(self.state, player=1)
        if not x_noise:
            return

        d = scipy.stats.dirichlet.rvs([1.0 for _ in range(7)])[0]
        self.policy = (1 - x_noise) * self.policy + 1 * d

    def update_value(self, value):
        self.nnet_value = self.n / (self.n + 1) * self.nnet_value + 1 / (self.n + 1) * value
        self.n += 1

    def is_terminal(self):
        return self.game.has_won(self.state, player=-1) or self.game.is_draw(self.state)

    def get_winner(self):
        return self.game.winner(self.state) * self.player

    def is_leaf(self):
        return not self.children

    def create_child(self, action):
        self.children.append(Node(self.game, action,
                                  state=self.game.move(self.state, action, 1) * -1,
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

    def select_child_nnet(self, c_puct, randomness=False):
        puct_scores = np.array([self.puct(child, c_puct) for child in self.children])
        if randomness:
            return self.children[random.choices(range(len(puct_scores)), weights=puct_scores)[0]]
        else:
            return self.children[np.random.choice(np.flatnonzero(puct_scores == np.max(puct_scores)))]

    def puct(self, child, c_puct):
        q = 1 - child.nnet_value
        u = c_puct * self.policy[child.action] * math.sqrt(self.n) / (1 + child.n)
        return q + u

    @staticmethod
    def upper_confidence_bound(value, parent_n, n, c, epsilon=0.00000001):
        return value + c * math.sqrt(math.log(parent_n + epsilon) / (n + epsilon))


if __name__ == '__main__':
    Game(gui=True).run(method1='input', method2='nnet', pause=0, runs1=50, runs2=10)
