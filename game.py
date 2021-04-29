import numpy as np
import tkinter as tk
from gui import Board
from nnet import NNet
import os
from node import Node
import copy
import time
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
root = tk.Tk()


class Game:
    def __init__(self, rows=6, columns=7, win_con=4, gui=False):
        self.rows = rows
        self.columns = columns
        self.game_state = np.zeros((columns, rows))
        self.win_con = win_con
        self.player = 1
        if gui:
            self.board = Board(root, self, rows, columns)
            self.board.pack(side="top", fill="both", expand="true", padx=4, pady=4)

    def make_move(self, column):
        self.game_state = self.move(self.game_state, column, self.player)
        self.player *= -1

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

    def tree_search(self, runs, temp=1, ignore_rate=0, c=10, nnet=None):
        start_node = Node(self, -1, copy.deepcopy(self.game_state) * self.player, self.player)
        start_node.n = 1
        for i in range(runs):
            current_node, path = self.get_to_leaf(start_node, c)
            if current_node.is_terminal():
                value = -1  # * 2
            elif current_node.n == 0:
                value = self.get_value(current_node, nnet)
            else:
                current_node.expand()
                current_node = current_node.children[0]
                path.append(current_node)
                value = self.get_value(current_node, nnet)
            for node in reversed(path):
                node.value += -value
                node.n += 1
                value *= -1
        return self.calc_pi(start_node=start_node, runs=runs, temp=temp, ignore_rate=ignore_rate)

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

    @staticmethod
    def get_value(current_node, nnet):
        if nnet:
            return nnet.prediction(current_node.state)[1][0][0] * 2 - 1
        else:
            return current_node.rollout()

    def calc_pi(self, start_node, runs, temp, ignore_rate):
        pi = [0 for _ in range(self.columns)]
        for child in start_node.children:
            if ignore_rate:
                print(ignore_rate)
                pi[child.action] = max(0, ((child.n - ignore_rate * runs / self.columns) / ignore_rate / runs)) ** temp
            else:
                pi[child.action] = child.n ** temp
        return pi / np.sum(pi)


def test_run(nnet1, nnet2, gui=True, randomness=False):
    game = Game(gui=gui)
    game.board.redraw()
    root.update()
    while game.winner(game.game_state) == 'none':
        if game.player == 1:
            policy, val = nnet1.prediction(state=game.game_state, player=game.player)
            print(f'value p1:{val}')
        else:
            policy, val = nnet2.prediction(state=game.game_state, player=game.player)
            print(f'value p2:{val}')
        if randomness:
            game.make_move((random.choices(range(len(policy)), weights=policy)[0]))
        else:
            game.make_move(np.argmax(policy))
        time.sleep(1)
        game.board.redraw()
        root.update()
    root.mainloop()


if __name__ == '__main__':
    test_run(NNet(load_data=True), NNet(load_data=True))
