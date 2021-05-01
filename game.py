import numpy as np
import tkinter as tk
from gui import Board
from nnet import NNet
import os
from node import Node
import copy
import random
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Game:
    def __init__(self, rows=6, columns=7, win_con=4, gui=False):
        self.rows = rows
        self.columns = columns
        self.game_state = np.zeros((columns, rows))
        self.win_con = win_con
        self.player = 1
        self.gui = gui
        if self.gui:
            self.root = tk.Tk()
            self.board = Board(self.root, self, rows, columns)
            self.board.pack(side="top", fill="both", expand="true", padx=4, pady=4)
            self.root.update()

    def make_move(self, column):
        self.game_state = self.move(self.game_state, column, self.player)
        self.player *= -1
        if self.gui:
            print('updating root')
            self.board.redraw()
            self.root.update()

    def run(self, method1, method2, structure_1='structure1', structure_2='structure1', print_out=True,
            runs1=200, runs2=200, c1=3, c2=3, pause=1):
        while self.winner(self.game_state) == 'none':
            if self.player == 1:
                self.make_move(self.decide_move(method=method1, runs=runs1, c=c1, structure=structure_1,
                                                print_out=print_out))
            else:
                self.make_move(self.decide_move(method=method2, runs=runs2, c=c2, structure=structure_2,
                                                print_out=print_out))
            time.sleep(pause)
        if self.gui:
            self.root.mainloop()

    def decide_move(self, method, c=3, runs=200, structure='structure1', print_out=True):
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
            if print_out:
                print(f'pi: {pi}')
            return np.argmax(pi)

        if method == 'nnet':
            pi = self.tree_search_nnet(nnet=NNet(structure=structure), c_puct=c, runs=runs, print_out=print_out)
            if print_out:
                print(f'pi: {pi}')
            return np.argmax(pi)

        if method == 'simple_nnet':
            pi, val = NNet().prediction(self.game_state, player=self.player)
            if print_out:
                print(f'pi: {pi}')
            return np.argmax(pi)

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

    def tree_search(self, runs, temp=1, ignore_rate=0, c=10):
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
        return self.calc_pi(start_node=start_node, runs=runs, temp=temp, ignore_rate=ignore_rate)

    def tree_search_nnet(self, runs, nnet, c_puct=4, temp=1,
                         ignore_rate=0, randomness=False, x_noise=0., decision='policy', print_out=False):
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
            # print(f'start pol: {start_node.policy}')
            # print(f'start val: {[child.nnet_value for child in start_node.children]}')
            print(f'estimated chance to win: {1 - start_node.select_child_nnet(0).nnet_value}')
        if decision == 'policy':
            return self.calc_pi(start_node=start_node, runs=runs, temp=temp, ignore_rate=ignore_rate)
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

    def calc_pi(self, start_node, runs, temp, ignore_rate):
        pi = [0 for _ in range(self.columns)]
        for child in start_node.children:
            if ignore_rate:
                print(ignore_rate)
                pi[child.action] = max(0, ((child.n - ignore_rate * runs / self.columns) / ignore_rate / runs)) ** temp
            else:
                pi[child.action] = child.n ** temp
        return pi / np.sum(pi)


if __name__ == '__main__':
    game = Game(gui=True)
    game.run(method1='input', method2='nnet', pause=0)
