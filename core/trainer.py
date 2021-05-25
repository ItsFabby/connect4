import numpy as np
import random
import sys
import os
import copy
import time

import constants as c
from db_connector import Connector
from game import Game
from nnet import NNet

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train(table: str = c.DEFAULT_TRAINING_TABLE, model_name: str = c.DEFAULT_MODEL_NAME, matches: int = 10,
          threshold: int = c.DEFAULT_THRESHOLD, learning_rate: float = c.DEFAULT_LEARNING_RATE,
          epochs: int = c.DEFAULT_EPOCHS, batch_size: int = c.DEFAULT_BATCH_SIZE, data_limit: int = 600000) -> None:
    new_net = NNet(learning_rate=learning_rate, epochs=epochs, batch_size=batch_size, model_name=model_name)
    old_net = NNet(model_name=model_name)
    db = Connector()
    examples = db.df_to_examples(db.retrieve_data(
        query=f"SELECT * FROM {table} ORDER BY counter DESC LIMIT {data_limit};"
    ))
    new_net.train(examples)
    score = _match_series(nnet1=new_net, nnet2=old_net, matches=matches)
    _evaluate_score(new_net, score, model_name, threshold)


def gen_examples(iterations: int = c.DEFAULT_ITERATIONS, nnet: NNet = None, table: str = c.DEFAULT_TRAINING_TABLE,
                 count_factor: float = 1.0) -> None:
    start = time.time()
    examples = _run_episode(nnet=nnet, iterations=iterations)
    for ex in copy.copy(examples):
        examples.append(_mirror_example(ex))
    print(f'generating data took {time.time() - start}s')

    start = time.time()
    db = Connector()
    db.insert_examples(examples, count_factor=count_factor, table=table)
    print(f'inserting data took {time.time() - start}s')


def train_stream(episodes: int = 50, model_name: str = c.DEFAULT_MODEL_NAME, rollout: bool = True,
                 iterations: int = c.DEFAULT_ITERATIONS, matches: int = 10,
                 threshold: int = c.DEFAULT_THRESHOLD) -> None:
    learning_rate, epochs, batch_size = _sample_parameters()
    new_net = NNet(learning_rate=learning_rate, epochs=epochs, batch_size=batch_size)
    old_net = NNet()

    _train_new_net(episodes=episodes, new_net=new_net, rollout=rollout, iterations=iterations)
    score = _match_series(nnet1=new_net, nnet2=old_net, matches=matches)

    print(f'parameters: learning rate ={learning_rate}, epochs={epochs}, batch size={batch_size}')
    _evaluate_score(new_net, score, model_name, threshold)


def _sample_parameters() -> tuple:
    learning_rate = 10 ** random.uniform(-3, -2)
    epochs = random.randint(1, 2)
    batch_size = int(10 ** random.uniform(0.5, 1.5))
    return learning_rate, epochs, batch_size


def _train_new_net(episodes: int, new_net: NNet, rollout: bool, iterations: int) -> None:
    print('-' * 20)
    examples = []
    for i in range(episodes):
        if rollout:
            new_examples = _run_episode(iterations=iterations)
        else:
            new_examples = _run_episode(nnet=new_net, iterations=iterations)
        for ex in new_examples:
            examples.append(ex)
            examples.append(_mirror_example(ex))
        sys.stdout.write(f'\repisode: {i + 1}/{episodes}')
        sys.stdout.flush()
    print('')
    new_net.train(examples)


def _run_episode(iterations: int, nnet: NNet = None, x_noise: float = c.DEFAULT_TRAINING_NOISE) -> list:
    examples = []
    game = Game()
    while True:
        if nnet:
            pi = game.tree_search_nnet(iterations=iterations, nnet=nnet, randomness=False, x_noise=x_noise)
        else:
            pi = game.tree_search(iterations=iterations)
        state = copy.deepcopy(game.game_state) * game.player
        examples.append([state, [pi]])
        game.make_move(np.argmax(pi))

        if game.winner(game.game_state) is not None:
            winner = game.winner(game.game_state)
            for i, example in enumerate(examples):
                example[1].append((winner * ((-1) ** i) + 1) / 2)
            return examples


def _mirror_example(example: list) -> list:
    ex = copy.deepcopy(example)
    ex[0] = np.flip(ex[0], axis=0)
    ex[1][0] = np.flip(ex[1][0])
    return ex


def _match_series(nnet1: NNet, nnet2: NNet, matches: int = 20, iterations: int = c.DEFAULT_ITERATIONS,
                  x_noise: float = c.DEFAULT_TRAINING_NOISE) -> int:
    score = 0
    for i in range(int(matches / 2)):
        score += _match(nnet1, nnet2, iterations=iterations, x_noise=x_noise)
        score += _match(nnet2, nnet1, iterations=iterations, x_noise=x_noise) * -1
        sys.stdout.write(f'\rmatch: {(i + 1) * 2}/{matches}, score: {score}')
        sys.stdout.flush()
    print('')
    return score


def _match(nnet1: NNet, nnet2: NNet, iterations: int = c.DEFAULT_ITERATIONS,
           x_noise: float = c.DEFAULT_TRAINING_NOISE) -> int:
    game = Game()
    while True:
        if game.player == 1:
            pi = game.tree_search_nnet(iterations=iterations, nnet=nnet1, x_noise=x_noise)
        else:
            pi = game.tree_search_nnet(iterations=iterations, nnet=nnet2, x_noise=x_noise)
        game.make_move(random.choices(range(len(pi)), weights=pi)[0])
        if game.has_won(game.game_state, player=game.player * -1):
            return game.player * -1
        elif game.is_draw(game.game_state):
            return 0


def _fast_match(nnet1: NNet, nnet2: NNet) -> int:
    game = Game()
    while True:
        if game.player == 1:
            policy = nnet1.prediction(state=game.game_state, player=game.player)[0]
        else:
            policy = nnet2.prediction(state=game.game_state, player=game.player)[0]
        game.make_move(random.choices(range(len(policy)), weights=policy)[0])
        if game.has_won(game.game_state, player=game.player * -1):
            return game.player * -1
        elif game.is_draw(game.game_state):
            return 0


def _evaluate_score(nnet: NNet, score: int, model_name: str, threshold: int) -> None:
    if score > threshold:
        nnet.model.save_weights(f'{parent_dir}\\weights\\{model_name}\\')
        print(f'new model accepted with score: {score}')
    else:
        print(f'new model rejected with score: {score}')


if __name__ == '__main__':
    pass
    # example operations:

    # while True:
    #     train_stream()

    # while True:
    #     gen_examples()

    # while True:
    #     train()
