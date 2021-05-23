import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, BatchNormalization, Activation, Add
import copy
import os
from typing import Any, Tuple

import constants as c

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
parent_dict = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class NNet:
    def __init__(self, epochs: int = c.DEFAULT_EPOCHS, learning_rate: float = c.DEFAULT_LEARNING_RATE,
                 batch_size: int = c.DEFAULT_BATCH_SIZE, structure: str = c.DEFAULT_STRUCTURE, load_data: bool = True):

        self.model = self._get_model(learning_rate, load_data, structure)
        self.structure = structure

        self.epochs = epochs
        self.batch_size = batch_size

    @classmethod
    def _get_model(cls, learning_rate: float, load_data: bool, structure: str) -> keras.Model:

        inputs = Input(shape=(c.COLUMNS, c.ROWS, 1))
        x = Conv2D(filters=64, kernel_size=(4, 4), padding='same')(inputs)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)

        x = cls._res_net(inputs=x, filters=64, kernel_size=(4, 4))
        x = cls._res_net(inputs=x, filters=64, kernel_size=(4, 4))
        x = cls._res_net(inputs=x, filters=64, kernel_size=(4, 4))

        policy = Conv2D(filters=64, kernel_size=(4, 4), padding='valid')(x)
        policy = BatchNormalization(axis=3)(policy)
        policy = Activation('relu')(policy)
        policy = Flatten()(policy)
        policy = Dense(256, activation='relu')(policy)
        policy = Dense(c.COLUMNS, activation='softmax', name='policy')(policy)

        value = Conv2D(filters=64, kernel_size=(4, 4), padding='valid')(x)
        value = BatchNormalization(axis=3)(value)
        value = Activation('relu')(value)
        value = Flatten()(value)
        value = Dense(1, activation='sigmoid', name='value')(value)

        model = keras.Model(inputs=inputs, outputs=[policy, value])

        model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
            loss={'value': 'mean_squared_error',
                  'policy': 'categorical_crossentropy'}
        )
        if load_data:
            try:
                model.load_weights(f'{parent_dict}\\weights\\{structure}\\').expect_partial()
            except ValueError:
                print('No saved weights found')
        return model

    @staticmethod
    def _res_net(inputs: Any, filters: int, kernel_size: Tuple[int, int]) -> Any:
        x_shortcut = inputs

        x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(inputs)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(x)
        x = BatchNormalization(axis=3)(x)

        x = Add()([x, x_shortcut])
        x = Activation('relu')(x)
        return x

    def train(self, examples: list, save_data: bool = False) -> None:
        x_train = np.array([example[0] for example in examples])
        y_train = (np.array([example[1][0] for example in examples]), np.array([example[1][1] for example in examples]))

        self.model.fit(x=x_train, y={'policy': y_train[0], 'value': y_train[1]},
                       epochs=self.epochs, batch_size=self.batch_size, shuffle=True)
        if save_data:
            self.model.save_weights(f'{parent_dict}\\weights\\{self.structure}\\')

    def prediction(self, state: np.ndarray, player: int = 1) -> np.array:
        state_copy = copy.deepcopy(state) * player
        prediction = self.model.predict(np.array([np.array(state_copy)]))
        policy = prediction[0][0]
        value = prediction[1][0][0]
        for move in range(len(policy)):
            if not self._is_legal(state=state_copy, move=move):
                policy[move] = 0
            else:
                policy[move] = policy[move] + 0.00001
        return policy / np.sum(policy), value

    @staticmethod
    def _is_legal(state: np.ndarray, move: int) -> bool:
        return state[move][c.ROWS - 1] == 0
