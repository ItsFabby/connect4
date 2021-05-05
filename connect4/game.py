import numpy as np
import math
import scipy.stats
import copy
import random
import time

from .nnet import NNet
from . import constants as c


class Game:
    def __init__(self):
        self.game_state = np.zeros((c.COLUMNS, c.ROWS))
        self.player = 1
        self.finished = False

    def restart(self):
        self.game_state = np.zeros((c.COLUMNS, c.ROWS))
        self.finished = False

    def make_move(self, column):
        if column not in self.get_legal_moves(self.game_state):
            print('illegal move')
            return
        self.game_state = self.move(self.game_state, column, self.player)
        if self.has_won(self.game_state, self.player):
            self.finished = True
        self.player *= -1

    def run(self, method1, method2, structure_1='structure1', structure_2='structure1', print_out=True,
            iterations1=c.DEFAULT_ITERATIONS, iterations2=c.DEFAULT_ITERATIONS, pause=1):
        while self.winner(self.game_state) == 'none':
            if self.player == 1:
                self.make_move(
                    self.decide_move(method=method1, iterations=iterations1,
                                     structure=structure_1, print_out=print_out)
                )
            else:
                self.make_move(
                    self.decide_move(method=method2, iterations=iterations2,
                                     structure=structure_2, print_out=print_out)
                )
            time.sleep(pause)

    def decide_move(self, method, iterations, structure='structure1', print_out=True):
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
            pi = self.tree_search(iterations=iterations)
            return np.argmax(pi)

        if method == 'nnet':
            pi = self.tree_search_nnet(nnet=NNet(structure=structure), iterations=iterations,
                                       print_out=print_out)
            return np.argmax(pi)

        if method == 'simple_nnet':
            policy, val = NNet().prediction(self.game_state, player=self.player)
            return np.argmax(policy)

        else:
            print('method not valid')

    @staticmethod
    def get_legal_moves(state):
        moves = []
        for column in range(c.COLUMNS):
            if state[column][c.ROWS - 1] == 0:
                moves.append(column)
        return moves

    @staticmethod
    def move(state, column, player):
        state_copy = copy.deepcopy(state)
        for i in range(c.ROWS):
            if state_copy[column][i] == 0:
                state_copy[column][i] = player
                break
        return state_copy

    @staticmethod
    def on_board(column, row):
        return c.COLUMNS > column >= 0 and c.ROWS > row >= 0

    def winner(self, state):
        if not self.get_legal_moves(state):
            return 0
        for col in range(c.COLUMNS):
            for row in range(c.ROWS):
                if state[col][row] != 0:
                    for direction in np.array([[0, 1], [1, 0], [1, -1], [1, 1]]):
                        count = 0
                        field = np.array([col, row])
                        while self.on_board(field[0], field[1]) and state[field[0]][field[1]] == state[col][row]:
                            count += 1
                            field = field + direction

                        if count >= c.WIN_CONDITION:
                            return state[col][row]
        return 'none'

    def has_won(self, state, player=1):
        for col in range(c.COLUMNS):
            directions = np.array([[0, 1]])
            if col <= (c.COLUMNS - c.WIN_CONDITION):
                directions = np.array([[0, 1], [1, 0], [1, -1], [1, 1]])
            for row in range(c.ROWS):
                if state[col][row] == player:
                    for direction in directions:
                        count = 0
                        field = np.array([col, row])
                        while self.on_board(field[0], field[1]) and state[field[0]][field[1]] == player:
                            count += 1
                            field = field + direction
                        if count >= c.WIN_CONDITION:
                            return True
        return False

    def is_draw(self, state):
        return not self.get_legal_moves(state)

    def tree_search(self, iterations, temp=c.DEFAULT_TEMP):
        start_node = Node(self, -1, copy.deepcopy(self.game_state) * self.player, self.player)
        start_node.n = 1
        for i in range(iterations):
            current_node, path = self.get_to_leaf(start_node)
            if current_node.is_terminal():
                value = -1
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

    def tree_search_nnet(self, iterations, nnet, temp=c.DEFAULT_TEMP, randomness=False, x_noise=0., print_out=False):
        start_node = Node(self, -1, copy.deepcopy(self.game_state) * self.player, self.player)
        start_node.n = 1
        for i in range(iterations):
            current_node, path = self.get_to_leaf_nnet(start_node, randomness)
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
            print(f'estimated chance to win: {1 - start_node.select_child_puct().nnet_value}')
        return self.calc_pi(start_node=start_node, temp=temp)

    @staticmethod
    def get_to_leaf_nnet(start_node, randomness):
        current_node = start_node
        path = []
        while not current_node.is_leaf():
            path.append(current_node)
            current_node = current_node.select_child_puct(randomness)
        return current_node, path

    @staticmethod
    def get_to_leaf(start_node):
        current_node = start_node
        path = [current_node]
        if not current_node.is_leaf():
            current_node = current_node.select_child_ubc(randomness=True)
            path.append(current_node)
        while not current_node.is_leaf():
            current_node = current_node.select_child_ubc(randomness=False)
            path.append(current_node)
        return current_node, path

    @staticmethod
    def calc_pi(start_node, temp):
        pi = np.zeros(c.COLUMNS)
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

        d = scipy.stats.dirichlet.rvs([1.0 for _ in range(c.COLUMNS)])[0]
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

    def select_child_puct(self, randomness=False):
        puct_scores = np.array([self.puct(child) for child in self.children])
        if randomness:
            return self.children[random.choices(range(len(puct_scores)), weights=puct_scores)[0]]
        else:
            return self.children[np.random.choice(np.flatnonzero(puct_scores == np.max(puct_scores)))]

    def select_child_ubc(self, randomness=False):
        ubc_scores = np.array([self.upper_confidence_bound(child) for child in self.children])
        if randomness:
            return self.children[random.choices(range(len(ubc_scores)), weights=ubc_scores)[0]]
        else:
            return self.children[np.random.choice(np.flatnonzero(ubc_scores == np.max(ubc_scores)))]

    def puct(self, child):
        q = 1 - child.nnet_value
        u = c.C_PUCT * self.policy[child.action] * math.sqrt(self.n) / (1 + child.n)
        return q + u

    def upper_confidence_bound(self, child, epsilon=0.00000001):
        return child.value + c.C_UBC * math.sqrt(math.log(self.n + epsilon) / (child.n + epsilon))


if __name__ == '__main__':
    pass
    # thread_run = threading.Thread(target=game1.run(method1='input', method2='nnet', pause=0, runs1=50, runs2=10))
    # thread_run.start()
    # game1.run(method1='input', method2='nnet', pause=0, runs1=50, runs2=10), game1.board.mainloop()
