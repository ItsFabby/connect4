# Connect Four AI based on a Neural Network

## About the Project

This project allows to train a neural network to play Connect Four using reinforced learning. This is achieved by using
a Monte Carlo tree search build on the predictions fetched by the neural network. Although initially only given the
rules of the game, with repeated cycle of self-play the network eventually learns to play the game strategically. The
concept is inspired by
DeepMind's [AlphaZero](https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go).

## Build With

- [Tensorflow](https://www.tensorflow.org/)
- [MySQL](https://www.mysql.com/)

## Getting Started

- Install the prerequisites listed in `requirement.txt`

Playing against the pretrained network:

- Run `__main__.py` with a Python interpreter.

Training a network:

- Set up a [MySQL](https://www.mysql.com/) database and adjust the login credentials in `core/constants.py`.
- Generate training data using `gen_examples()` in `core/trainer.py`.
- Train the network using `train()` in `core/trainer.py`.

Make sure to change `model_name` when training a new network (either by giving a keyword argument or by changing the
default in `core/constants.py`), otherwise the old weights will be overwritten! \
Feel free to change the structure of the network in `core/nnet.py`.

## Network Architecture

The network consists of an initial convolutional layer with the shape of the board (6, 7), followed by three residual
blocks, each consisting of two convolutional layers. Following this, the network splits into a policy and a value head.

The policy head consists of a convolutional layer followed by a dense layer leading to an output layer with one output
for each column and softmax activation, covering all possible moves. The value head consists of a convolutional layer
leading to a single output with sigmoid activation. All convolutional layers have a kernel size of (4, 4) with 64
filters and batch normalisation is applied between each one.

## License

Distributed under the MIT License. See `LICENSE` for more information.
