import numpy as np
import math
import scipy.stats
import copy
import random
from typing import Union, List, Tuple, Optional

from nnet import NNet
import constants as c


class Game:
    """
    Implements Connect 4.
    
    Attrs:
        game_state: 2-axis numpy array that saves the current state of the game. 1 is player 1, -1 is player 2, 0 is an 
        empty square. Index ordering is (column, row).
        
        player: Player whose turn is next.
        
        finished: True if the game is won or drawn.
    """

    def __init__(self):
        self.game_state = np.zeros((c.COLUMNS, c.ROWS))
        self.player = 1
        self.finished = False

    def decide_move(self, method: str, iterations: int, model_name: str = c.DEFAULT_MODEL_NAME,
                    print_out: bool = True) -> Union[int, np.ndarray]:
        """
        Outputs a move for the current game state determined by a specified method.
        
        :param method: 'nnet': neural network with tree search, 'simple_nnet': neural network without tree search,
            'mcts': Monte Carlo tree search, 'input': input via terminal.
        :param iterations: Number of iterations in tree searches
        :param model_name: Structure name of the neural network
        :param print_out: Prints out extra information by the neural network
        :return: Determined best move
        """
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
            pi = self.tree_search_nnet(nnet=NNet(model_name=model_name), iterations=iterations,
                                       print_out=print_out)
            return np.argmax(pi)

        if method == 'simple_nnet':
            policy, val = NNet().prediction(self.game_state, player=self.player)
            return np.argmax(policy)

        else:
            print('method not valid')

    def make_move(self, column: int) -> None:
        """
        Applies a move to the current game state.
        
        :param column: Column where to make the move. 
        """
        if column not in self.get_legal_moves(self.game_state):
            print('illegal move')
            return
        self.game_state = self.move(self.game_state, column, self.player)
        if self.has_won(self.game_state, self.player):
            self.finished = True
        self.player *= -1

    @staticmethod
    def move(state: np.ndarray, column: int, player: int) -> np.ndarray:
        """
        Applies a move to a given state and returns the new state.
        
        :param state: State to apply the move to
        :param column: Column where to make the move
        :param player: Player who makes the move
        :return: New state
        """
        state_copy = copy.deepcopy(state)
        for i in range(c.ROWS):
            if state_copy[column][i] == 0:
                state_copy[column][i] = player
                break
        return state_copy

    @staticmethod
    def get_legal_moves(state: np.ndarray) -> List[int]:
        """
        Allowed moves for a given state.
        
        :param state: Given state
        :return: List of allowed columns
        """
        moves = []
        for column in range(c.COLUMNS):
            if state[column][c.ROWS - 1] == 0:
                moves.append(column)
        return moves

    @classmethod
    def winner(cls, state: np.ndarray) -> Union[int, str, None]:
        """
        Returns the winner of the current state if their is one, or if it's draw

        :param state: Given state
        :return: 1 for player 1, -1 for player 2, 0 for a draw, None for a not finished game
        """
        if not cls.get_legal_moves(state):
            return 0
        for col in range(c.COLUMNS):
            for row in range(c.ROWS):
                if state[col][row] != 0:
                    for direction in np.array([[0, 1], [1, 0], [1, -1], [1, 1]]):
                        count = 0
                        current_field = np.array([col, row])
                        while cls._on_board(current_field[0], current_field[1]) \
                                and state[current_field[0]][current_field[1]] == state[col][row]:
                            count += 1
                            current_field = current_field + direction

                        if count >= c.WIN_CONDITION:
                            return state[col][row]
        return None

    @staticmethod
    def _on_board(column: int, row: int) -> bool:
        return c.COLUMNS > column >= 0 and c.ROWS > row >= 0

    @classmethod
    def has_won(cls, state: np.ndarray, player: int = 1) -> bool:
        """
        Determines if a given player has won. Faster than winner().

        :param state: Given state
        :param player: Given player 1 or -1
        :return: True if player has won.
        """
        for col in range(c.COLUMNS):
            directions = np.array([[0, 1]])
            if col <= (c.COLUMNS - c.WIN_CONDITION):
                directions = np.array([[0, 1], [1, 0], [1, -1], [1, 1]])
            for row in range(c.ROWS):
                if state[col][row] == player:
                    for direction in directions:
                        count = 0
                        current_field = np.array([col, row])
                        while cls._on_board(current_field[0], current_field[1]) \
                                and state[current_field[0]][current_field[1]] == player:
                            count += 1
                            current_field = current_field + direction
                        if count >= c.WIN_CONDITION:
                            return True
        return False

    @classmethod
    def is_draw(cls, state: np.ndarray) -> bool:
        """
        Determines if given state is a draw

        :param state:
        :return: True if game is a draw (no more possible moves).
        """
        return not cls.get_legal_moves(state)

    def tree_search(self, iterations: int, temp: float = c.DEFAULT_TEMP) -> np.array:
        """
        Standard Monte Carlo tree search without a neural network.

        :param iterations: Iterations of the tree search.
        :param temp: Element wise exponent on the policy vector.
        :return: Policy vector giving probabilities for each move.
        """
        start_node = Node(None, copy.deepcopy(self.game_state) * self.player, self.player)
        start_node.n = 1
        for i in range(iterations):
            current_node, path = self._get_to_leaf_mcts(start_node)
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
        return self._calc_pi(start_node=start_node, temp=temp)

    @staticmethod
    def _get_to_leaf_mcts(start_node: 'Node') -> Tuple['Node', list]:
        current_node = start_node
        path = [current_node]
        if not current_node.is_leaf():
            current_node = current_node.select_child_ubc(randomness=True)
            path.append(current_node)
        while not current_node.is_leaf():
            current_node = current_node.select_child_ubc(randomness=False)
            path.append(current_node)
        return current_node, path

    def tree_search_nnet(self, iterations: int, nnet: 'NNet', temp: float = c.DEFAULT_TEMP,
                         randomness: bool = False, x_noise: float = 0., print_out: bool = False) -> np.array:
        """
        Modified Monte Carlo tree search, using the neural network.

        :param iterations: Number of iterations of the tree search.
        :param nnet: neural network
        :param temp: Element wise exponent on the policy vector.
        :param randomness: If True, child nodes are randomly chosen with weighted probability
        :param x_noise: Between 0 and 1. Noise applied to the policy vector from the neural network.
        :param print_out: Prints out extra information from the neural network.
        :return: Policy vector giving probabilities for each move.
        """
        start_node = Node(None, copy.deepcopy(self.game_state) * self.player, self.player)
        start_node.n = 1
        for i in range(iterations):
            current_node, path = self._get_to_leaf_nnet(start_node, randomness)
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
        return self._calc_pi(start_node=start_node, temp=temp)

    @staticmethod
    def _get_to_leaf_nnet(start_node: 'Node', randomness: bool) -> Tuple['Node', list]:
        current_node = start_node
        path = []
        while not current_node.is_leaf():
            path.append(current_node)
            current_node = current_node.select_child_puct(randomness)
        return current_node, path

    @staticmethod
    def _calc_pi(start_node: 'Node', temp: float) -> np.array:
        pi = np.zeros(c.COLUMNS)
        for child in start_node.children:
            pi[child.action] = child.n ** temp
        return pi / np.sum(pi)


class Node:
    def __init__(self, action: Optional[int], state: np.ndarray, player: int):
        self.n = 0
        self.value = 0
        self.children = []
        self.state = state
        self.player = player  # who makes the next move
        self.action = action  # last move that created this node

        self.policy = None
        self.nnet_value = 0.5

    def is_terminal(self) -> bool:
        return Game.has_won(self.state, player=-1) or Game.is_draw(self.state)

    def get_winner(self) -> Union[int, str]:
        return Game.winner(self.state) * self.player

    def is_leaf(self) -> bool:
        return not self.children

    def create_child(self, action: int) -> None:
        self.children.append(Node(action, Game.move(self.state, action, 1) * -1, self.player * -1))

    def expand(self) -> None:
        for action in Game.get_legal_moves(self.state):
            self.create_child(action)

    def rollout(self) -> int:
        sim_state = copy.deepcopy(self.state)
        player_swap = 1
        while True:
            if Game.has_won(sim_state):
                return player_swap
            elif Game.is_draw(sim_state):
                return 0
            else:
                column = random.choice(Game.get_legal_moves(sim_state))
                sim_state = Game.move(sim_state, column, 1) * -1
                player_swap *= -1

    def select_child_puct(self, randomness: bool = False) -> 'Node':
        puct_scores = np.array([self._puct(child) for child in self.children])
        if randomness:
            return self.children[random.choices(range(len(puct_scores)), weights=puct_scores)[0]]
        else:
            return self.children[np.random.choice(np.flatnonzero(puct_scores == np.max(puct_scores)))]

    def _puct(self, child: 'Node') -> float:
        q = 1 - child.nnet_value
        u = c.C_PUCT * self.policy[child.action] * math.sqrt(self.n) / (1 + child.n)
        return q + u

    def select_child_ubc(self, randomness: bool = False) -> 'Node':
        ubc_scores = np.array([self._upper_confidence_bound(child) for child in self.children])
        if randomness:
            return self.children[random.choices(range(len(ubc_scores)), weights=ubc_scores)[0]]
        else:
            return self.children[np.random.choice(np.flatnonzero(ubc_scores == np.max(ubc_scores)))]

    def _upper_confidence_bound(self, child: 'Node', epsilon: float = 0.00000001) -> float:
        return child.value + c.C_UBC * math.sqrt(math.log(self.n + epsilon) / (child.n + epsilon))

    def fetch_prediction(self, nnet: NNet, x_noise: float = 0.) -> None:
        self.policy, self.nnet_value = nnet.prediction(self.state, player=1)
        if not x_noise:
            return

        d = scipy.stats.dirichlet.rvs([1.0 for _ in range(c.COLUMNS)])[0]
        self.policy = (1 - x_noise) * self.policy + 1 * d

    def update_value(self, value: float) -> None:
        self.nnet_value = self.n / (self.n + 1) * self.nnet_value + 1 / (self.n + 1) * value
        self.n += 1


if __name__ == '__main__':
    pass
