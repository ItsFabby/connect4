import numpy as np
import random
import sys
import copy
import time

from . import constants as c
from .db_connector import Connector
from .game import Game
from .nnet import NNet


def train(table=c.DEFAULT_TRAINING_TABLE, structure=c.DEFAULT_STRUCTURE, matches=10,
          threshold=c.DEFAULT_THRESHOLD, learning_rate=c.DEFAULT_LEARNING_RATE, epochs=c.DEFAULT_EPOCHS,
          batch_size=c.DEFAULT_BATCH_SIZE, data_limit=600000):
    new_net = NNet(learning_rate=learning_rate, epochs=epochs, batch_size=batch_size, structure=structure)
    old_net = NNet(structure=structure)
    db = Connector()
    examples = db.df_to_examples(db.retrieve_data(
        query=f"SELECT * FROM {table} ORDER BY counter DESC LIMIT {data_limit};"
    ))
    new_net.train(examples)
    score = match_series(nnet1=new_net, nnet2=old_net, matches=matches)
    evaluate_score(new_net, score, structure, threshold)


def gen_data(iterations=c.DEFAULT_ITERATIONS, nnet=None, count_factor=1.0):
    start = time.time()
    examples = run_episode(nnet=nnet, iterations=iterations)
    for ex in copy.copy(examples):
        examples.append(mirror_example(ex))
    print(f'generating data took {time.time() - start}s')

    start = time.time()
    db = Connector()
    db.insert_examples(examples, count_factor=count_factor, table='training_data')
    db.insert_examples(examples, count_factor=count_factor, table='training_data1')
    print(f'inserting data took {time.time() - start}s')


def train_stream(episodes=50, structure=c.DEFAULT_STRUCTURE, rollout=True, iterations=c.DEFAULT_ITERATIONS,
                 matches=10, insert=False, threshold=c.DEFAULT_THRESHOLD):
    learning_rate, epochs, batch_size = sample_parameters()
    new_net = NNet(learning_rate=learning_rate, epochs=epochs, batch_size=batch_size)
    old_net = NNet()

    train_new_net(episodes=episodes, new_net=new_net, rollout=rollout, iterations=iterations)
    score = match_series(nnet1=new_net, nnet2=old_net, matches=matches)

    print(f'parameters: learning rate ={learning_rate}, epochs={epochs}, batch size={batch_size}')
    evaluate_score(new_net, score, structure, threshold)

    if insert:
        database = Connector()
        database.send_query(
            f'INSERT INTO parameters (score, matches, learning_rate, epochs, batch_size)'
            f'VALUES ({score}, {matches}, {learning_rate}, {epochs}, {batch_size});',
            print_out=False)


def sample_parameters():
    learning_rate = 10 ** random.uniform(-3, -2)
    epochs = random.randint(1, 2)
    batch_size = int(10 ** random.uniform(0.5, 1.5))
    return learning_rate, epochs, batch_size


def train_new_net(episodes, new_net, rollout, iterations):
    print('-' * 20)
    examples = []
    for i in range(episodes):
        if rollout:
            new_examples = run_episode(iterations=iterations)
        else:
            new_examples = run_episode(nnet=new_net, iterations=iterations)
        for ex in new_examples:
            examples.append(ex)
            examples.append(mirror_example(ex))
        sys.stdout.write(f'\repisode: {i + 1}/{episodes}')
        sys.stdout.flush()
    print('')
    new_net.train(examples)


def run_episode(iterations, nnet=None):
    examples = []
    game = Game()
    while True:
        if nnet:
            pi = game.tree_search_nnet(iterations=iterations, nnet=nnet, randomness=False, x_noise=0.5)
        else:
            pi = game.tree_search(iterations=iterations)
        state = copy.deepcopy(game.game_state) * game.player
        examples.append([np.array(state), [pi]])
        action = np.argmax(pi)
        game.make_move(action)
        if game.winner(game.game_state) != 'none':
            winner = game.winner(game.game_state)
            for i, example in enumerate(examples):
                example[1].append((winner * ((-1) ** i) + 1) / 2)
            return examples


def mirror_example(example):
    ex = copy.deepcopy(example)
    ex[0] = np.flip(ex[0], axis=0)
    ex[1][0] = np.flip(ex[1][0])
    return ex


def fast_match(nnet1, nnet2):
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


def match(nnet1, nnet2, iterations=c.DEFAULT_ITERATIONS, x_noise=0.5):
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


def match_series(nnet1, nnet2, matches=20, iterations=c.DEFAULT_ITERATIONS, x_noise=0.5):
    score = 0
    for i in range(int(matches / 2)):
        score += match(nnet1, nnet2, iterations=iterations, x_noise=x_noise)
        score += match(nnet2, nnet1, iterations=iterations, x_noise=x_noise) * -1
        sys.stdout.write(f'\rmatch: {(i + 1) * 2}/{matches}, score: {score}')
        sys.stdout.flush()
    print('')
    return score


def evaluate_score(nnet, score, structure, threshold):
    if score > threshold:
        nnet.model.save_weights(f'connect4/weights/{structure}/')
        print(f'new model accepted with score: {score}')
    else:
        print(f'new model rejected with score: {score}')


if __name__ == '__main__':
    # while True:
    #     train_stream(episodes=1, runs=50, matches=18, insert=False, threshold=10)
    while True:
        # gen_data(c=2, runs=50, nnet=NNet(), count_factor=0.5)
        train(learning_rate=0.0000001, epochs=1, batch_size=256)
        train(learning_rate=0.0000001, epochs=1, batch_size=512)
