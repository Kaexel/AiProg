import random
from abc import abstractmethod, ABC
import tensorflow as tf

import numpy as np

from game_managers.game_manager import GameManager


def make_keras_model(filters: tuple, dense: tuple, rows, cols, activation_function, optimizer):
    """
    :param filters: tuple of # filters used for each conv layer
    :param dense: tuple of # neurons used for each dense layer
    :param rows: rows on board
    :param cols: cols on board
    :param activation_function: activation function to be used for layers
    :param optimizer: optimizer to be used in model
    :return: compiled model
    """
    model = tf.keras.Sequential()
    # Setting up conv layers
    model.add(tf.keras.layers.Conv2D(filters=filters[0], kernel_size=(3, 3), strides=1, padding='same', data_format='channels_first', activation=activation_function, input_shape=(6, rows, cols)))
    model.add(tf.keras.layers.BatchNormalization())
    if len(filters) > 1:
        for w in filters[1:]:
            model.add(tf.keras.layers.Conv2D(filters=w, kernel_size=(3, 3), strides=1, padding='same', data_format='channels_first', activation=activation_function))
            model.add(tf.keras.layers.BatchNormalization())

    # TODO: maybe investigate pure conv further?
    #model.add(tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), padding="same", data_format='channels_first', activation='softmax'))
    #model.add(tf.keras.layers.Flatten())
    # Setting up final dense layers
    model.add(tf.keras.layers.Flatten())
    for neurons in dense:
        model.add(tf.keras.layers.Dense(neurons, activation=activation_function))
    model.add(tf.keras.layers.Dense(rows * cols, activation="softmax"))
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


class PolicyObject(ABC):
    """
    Abstract class defining methods for policy objects
    """
    @abstractmethod
    def get_action(self, state, mgr: GameManager):
        raise NotImplementedError


class PolicyModel(PolicyObject):
    """
    Policy object using keras predict API.
    """
    def __init__(self, model):
        self.model = model
        self.epsilon = 0.015

    def get_action(self, state, mgr: GameManager):

        nn_state_representation = mgr.nn_state_representation(state)
        move_distribution = self.model.predict(nn_state_representation.reshape((1, 5, 5, 5))).reshape((25,))
        board_size = mgr.get_size()

        # TODO: keep working on rotation
        # We've rotated player two so that the model generalizes better.
        # We therefore need to rotate the distribution back
        #if state.player_turn == Players.BLACK:
        #    move_distribution = np.rot90(np.reshape(move_distribution, (board_size, board_size))).flatten()
        legal_actions = mgr.get_legal_actions(state)
        legal_action_indices = [board_size * y + x for y, x in legal_actions]
        move_distribution_weights = move_distribution[legal_action_indices]

        #choice = np.random.choice(np.array(legal_actions, dtype=('i','i')), move_distribution_weights)

        # Balancing probability of picking best move and picking random choice weighted by distribution
        if random.random() >= self.epsilon:
            return legal_actions[np.ndarray.argmax(move_distribution_weights)]
        move_distribution_weights = move_distribution_weights / sum(move_distribution_weights)
        return random.choices(legal_actions, weights=move_distribution_weights)[0]


class LiteModel(PolicyObject):
    """
    Policy object using Tflite. MUCH faster.
    """
    @classmethod
    def from_file(cls, model_path):
        return LiteModel(tf.lite.Interpreter(model_path=model_path))

    @classmethod
    def from_keras_model(cls, kmodel):
        converter = tf.lite.TFLiteConverter.from_keras_model(kmodel)
        tflite_model = converter.convert()
        return LiteModel(tf.lite.Interpreter(model_content=tflite_model))

    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.interpreter.allocate_tensors()
        input_det = self.interpreter.get_input_details()[0]
        output_det = self.interpreter.get_output_details()[0]
        self.input_index = input_det["index"]
        self.output_index = output_det["index"]
        self.input_shape = input_det["shape"]
        self.output_shape = output_det["shape"]
        self.input_dtype = input_det["dtype"]
        self.output_dtype = output_det["dtype"]
        self.epsilon = 0.1

    def get_action(self, state, manager: GameManager):
        nn_state_representation = manager.nn_state_representation(state)
        move_distribution = self.predict_single(nn_state_representation)
        board_size = manager.get_size()


        # TODO: keep working on rotation
        # We've rotated player two so that the model generalizes better.
        # We therefore need to rotate the distribution back
        #if state.player_turn == Players.BLACK:
        #    move_distribution = np.rot90(np.reshape(move_distribution, (board_size, board_size))).flatten()
        legal_actions = manager.get_legal_actions(state)
        legal_action_indices = [board_size * y + x for y, x in legal_actions]
        move_distribution_weights = move_distribution[legal_action_indices]

        #choice = np.random.choice(np.array(legal_actions, dtype=('i','i')), move_distribution_weights)

        # Balancing probability of picking best move and picking random choice weighted by distribution
        if random.random() >= self.epsilon:
            return legal_actions[np.ndarray.argmax(move_distribution_weights)]
        move_distribution_weights = move_distribution_weights / sum(move_distribution_weights)
        return random.choices(legal_actions, weights=move_distribution_weights)[0]

    def predict(self, inp):
        inp = inp.astype(self.input_dtype)
        count = inp.shape[0]
        out = np.zeros((count, self.output_shape[1]), dtype=self.output_dtype)
        for i in range(count):
            self.interpreter.set_tensor(self.input_index, inp[i:i + 1])
            self.interpreter.invoke()
            out[i] = self.interpreter.get_tensor(self.output_index)[0]
        return out

    def predict_single(self, inp):
        """ Like predict(), but only for a single record. The input data can be a Python list. """
        inp = np.array([inp], dtype=self.input_dtype)
        self.interpreter.set_tensor(self.input_index, inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_index)
        return out[0]
