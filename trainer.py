from game import Game
from nnet import NNet
import numpy as np
import random
import sys
import copy
from db_connector import Connector
import time


def train(matches=50, threshold=0, learning_rate=0.00001, epochs=1, batch_size=128, data_limit=600000):
    new_net = NNet(learning_rate=learning_rate, epochs=epochs, batch_size=batch_size)
    old_net = NNet()
    db = Connector()
    examples = db.df_to_examples(db.retrieve_data(
        query=f"SELECT * FROM training_data ORDER BY counter DESC LIMIT {data_limit};"
    ))
    new_net.train(examples)
    score = duel(nnet1=new_net, nnet2=old_net, matches=matches)
    evaluate_score(new_net, score, threshold)


def gen_data(c, runs, nnet=None, count_factor=1.0):
    start = time.time()
    examples = run_episode(nnet=nnet, c=c, runs=runs)
    for ex in copy.copy(examples):
        examples.append(mirror_example(ex))
    db = Connector()
    print(f'generating data took {time.time() - start}s')
    start = time.time()
    db.insert_examples(examples, count_factor=count_factor, table='training_data')
    print(f'inserting data took {time.time() - start}s')
    start = time.time()
    db.insert_examples(examples, count_factor=count_factor, table='training_data1')
    print(f'inserting data took {time.time() - start}s')


def train_stream(episodes, rollout=True, runs=100, matches=100, insert=False, threshold=0):
    learning_rate, c, epochs, batch_size = sample_parameters()
    new_net = NNet(learning_rate=learning_rate, epochs=epochs, batch_size=batch_size)
    old_net = NNet()

    train_new_net(episodes=episodes, c=c, new_net=new_net, rollout=rollout, runs=runs)
    score = duel(nnet1=new_net, nnet2=old_net, matches=matches)

    print(f'parameters: c={c}, learning rate ={learning_rate}, epochs={epochs}, batch size={batch_size}')
    evaluate_score(new_net, score, threshold)

    if insert:
        database = Connector()
        database.send_query(
            f'INSERT INTO parameters (score, matches, c, learning_rate, epochs, batch_size)'
            f'VALUES ({score}, {matches}, {c}, {learning_rate}, {epochs}, {batch_size});',
            print_out=False)


def sample_parameters():
    learning_rate = 10 ** random.uniform(-3, -2)
    c = random.randint(2, 10)
    epochs = random.randint(1, 2)
    batch_size = int(10 ** random.uniform(0.5, 1.5))
    return learning_rate, c, epochs, batch_size


def train_new_net(episodes, c, new_net, rollout, runs):
    print('-' * 20)
    examples = []
    for i in range(episodes):
        if rollout:
            new_examples = run_episode(c=c, runs=runs)
        else:
            new_examples = run_episode(nnet=new_net, c=c, runs=runs)
        for ex in new_examples:
            examples.append(ex)
            examples.append(mirror_example(ex))
        sys.stdout.write(f'\repisode: {i + 1}/{episodes}')
        sys.stdout.flush()
    print('')
    new_net.train(examples)


def run_episode(c, runs, nnet=None):
    examples = []
    game = Game()
    while True:
        if nnet:
            pi = game.tree_search_nnet(runs=runs, nnet=nnet, c_puct=c,
                                       randomness=False, x_noise=0.5)
        else:
            pi = game.tree_search(runs=runs, c=c)
        state = copy.deepcopy(game.game_state) * game.player
        examples.append([np.array(state), [pi]])
        # action = random.choices(range(len(pi)), weights=pi)[0]
        action = np.argmax(pi)
        game.make_move(action)
        if game.winner(game.game_state) != 'none':
            winner = game.winner(game.game_state)
            for i, example in enumerate(examples):
                example[1].append((winner * ((-1) ** i) + 1) / 2)  # transform value from interval [-1,1] to [0,1]
            return examples


def mirror_example(example):
    ex = copy.deepcopy(example)
    ex[0] = np.flip(ex[0], axis=0)
    ex[1][0] = np.flip(ex[1][0])
    return ex


def match(nnet1, nnet2):
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


def duel(nnet1, nnet2, matches=100):
    score = 0
    for i in range(int(matches / 2)):
        score += match(nnet1, nnet2)
        score += match(nnet2, nnet1) * -1
        sys.stdout.write(f'\rmatch: {(i + 1) * 2}/{matches}, score: {score}')
        sys.stdout.flush()
    print('')
    return score


def evaluate_score(nnet, score, threshold):
    if score > threshold:
        nnet.model.save_weights('data/')
        print(f'new model accepted with score: {score}')
    else:
        print(f'new model rejected with score: {score}')


if __name__ == '__main__':
    # duel(NNet(load_data=False), NNet(load_data=True), matches=100)
    # while True:
    #     train_stream(episodes=1, runs=50, matches=18, insert=False, threshold=10)
    while True:
        gen_data(c=4, runs=50, nnet=NNet(), count_factor=0.5)
        # train(learning_rate=0.0003, epochs=1, batch_size=128)
