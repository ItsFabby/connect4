import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, BatchNormalization, Activation, Add
import copy
import os
# from sklearn.model_selection import KFold

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class NNet:
    def __init__(self, epochs=1, learning_rate=0.001, batch_size=1024, structure='structure0', load_data=True):
        self.structure = structure
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        # self.tb_callback = keras.callbacks.TensorBoard(log_dir='logs/', histogram_freq=1)

        inputs = Input(shape=(7, 6, 1))
        x = Conv2D(filters=64, kernel_size=(4, 4), padding='same')(inputs)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)

        x = self.res_net(inputs=x, filters=64, kernel_size=(4, 4))
        x = self.res_net(inputs=x, filters=64, kernel_size=(4, 4))
        x = self.res_net(inputs=x, filters=64, kernel_size=(4, 4))

        policy = Conv2D(filters=64, kernel_size=(4, 4), padding='valid')(x)
        policy = BatchNormalization(axis=3)(policy)
        policy = Activation('relu')(policy)
        policy = Flatten()(policy)
        if structure == 'structure1':
            policy = Dense(256, activation='relu')(policy)
        policy = Dense(7, activation='softmax', name='policy')(policy)

        value = Conv2D(filters=64, kernel_size=(4, 4), padding='valid')(x)
        value = BatchNormalization(axis=3)(value)
        value = Activation('relu')(value)
        value = Flatten()(value)
        value = Dense(1, activation='sigmoid', name='value')(value)

        self.model = keras.Model(inputs=inputs, outputs=[policy, value])

        self.model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=self.learning_rate),
            loss={'value': 'mean_squared_error',
                  'policy': 'categorical_crossentropy'}
        )
        if load_data:
            try:
                self.model.load_weights(f'{structure}/').expect_partial()
            except ValueError:
                print('could not load data')

    @staticmethod
    def res_net(inputs, filters, kernel_size):
        x_shortcut = inputs

        x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(inputs)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(x)
        x = BatchNormalization(axis=3)(x)

        x = Add()([x, x_shortcut])
        x = Activation('relu')(x)
        return x

    def train(self, examples, save_data=False):
        x_train = np.array([example[0] for example in examples])
        y_train = (np.array([example[1][0] for example in examples]), np.array([example[1][1] for example in examples]))
        # kf = KFold(n_splits=10, shuffle=True)
        # train_index, test_index = [i for i in kf.split(x)][0]
        # print(train_index)
        # x_train = x[train_index]
        # y_train = y[train_index]
        # x_test = x[test_index]
        # y_test = y[test_index]

        self.model.fit(x_train, {'policy': y_train[0], 'value': y_train[1]},
                       epochs=self.epochs, batch_size=self.batch_size, shuffle=True)  # , callbacks=[self.tb_callback]
        # evaluation = self.model.evaluate(x_test, y_test)
        # print(evaluation)
        if save_data:
            self.model.save_weights('data/')

    @staticmethod
    def is_legal(state, move, rows=6):
        return state[move][rows - 1] == 0

    def prediction(self, state, player=1):
        state_copy = copy.deepcopy(state) * player
        prediction = self.model.predict(np.array([np.array(state_copy)]))
        policy = prediction[0][0]
        value = prediction[1][0][0]
        for move in range(len(policy)):
            if not self.is_legal(state=state_copy, move=move):
                policy[move] = 0
            else:
                policy[move] = policy[move] + 0.001  # **2
        return policy / np.sum(policy), value
