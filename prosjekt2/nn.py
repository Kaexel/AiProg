import random
from abc import abstractmethod, ABC

import keras
import tensorflow
from torch import nn
# from torchvision import datasets, transforms
import tensorflow as tf

import numpy as np

from game_managers.game_manager import GameManager
from sim_worlds.sim_world import Players


class NeuralNetworkTorch(nn.Module):
    def __init__(self, x, y, w):
        super(NeuralNetworkTorch, self).__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Conv2d(9, w, (5, 5), stride=(1, 1)),  # TODO: remember to pad image with two rows/columns of black/white
            nn.ReLU(),
            nn.Conv2d(w, w, (3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(w, w, (3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(w, w, (3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(w, w, (3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(w, 1, (1, 1), stride=(1, 1), padding=1),  # TODO: output must be x*y, and scaled to x*y - q
            nn.Softmax(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def make_keras_model(filters: tuple, dense: tuple, rows, cols, activation_function, optimizer):
    model = tf.keras.Sequential()
    # Setting up conv layers
    model.add(tf.keras.layers.Conv2D(filters=filters[0], kernel_size=(3, 3), strides=1, padding='same', data_format='channels_first', activation=activation_function, input_shape=(5, rows, cols)))
    model.add(tf.keras.layers.BatchNormalization())
    if len(filters) > 1:
        for w in filters[1:]:
            model.add(tf.keras.layers.Conv2D(filters=w, kernel_size=(3, 3), strides=1, padding='same', data_format='channels_first', activation=activation_function))
            model.add(tf.keras.layers.BatchNormalization())

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


class LiteModel(PolicyObject):

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
        self.epsilon = 0.05


    def get_action(self, state, manager: GameManager):

        nn_state_representation = manager.nn_state_representation(state)
        move_distribution = self.predict_single(nn_state_representation)
        board_size = manager.get_size()


        # TODO: keep working on rotation
        # We've rotated player two so that the model generalizes better.
        # We therefore need to rotate the distribution back
        #if state.player_turn == Players.BLACK:
        #    move_distribution = np.rot90(move_distribution)
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
