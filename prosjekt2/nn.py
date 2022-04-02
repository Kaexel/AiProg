import os
import random

import keras
import torch
from torch import nn
from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
import tensorflow as tf
from tensorflow import keras as ker
from keras.models import *
from keras.layers import *

import numpy as np

from sim_worlds.sim_world import SimWorld


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


t = NeuralNetworkTorch(1, 2, 3)


def make_keras_model(w, rows, cols):
    model = keras.models.Sequential()
    model.add(Conv2D(filters=w, kernel_size=(3, 3), strides=1, padding='valid', data_format='channels_first',
                     activation='relu', input_shape=(5, rows + 2 , cols + 2)))
                     #activation='relu', input_shape=(5, rows + 2, cols + 2)))
    model.add(BatchNormalizationV2())
    # model.add(Dropout(rate=0.2))
    model.add(Conv2D(filters=w, kernel_size=(5, 5), strides=1, padding='same', data_format='channels_first', activation='relu'))
    model.add(BatchNormalizationV2())
    model.add(Conv2D(filters=w, kernel_size=(3, 3), strides=1, padding='same', data_format='channels_first', activation='relu'))
    model.add(BatchNormalizationV2())
    #model.add(Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding='valid', activation="softmax"))
    # model.add(Conv2D(filters=w, kernel_size=(3, 3), strides=1, padding='same', data_format='channels_last', activation='relu'))
    # model.add(Conv2D(filters=rows*cols, kernel_size=(1, 1), strides=1, padding='valid', data_format='channels_last', activation='softmax'))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(rows * cols, activation="softmax"))
    model.summary()
    #optimizer = tf.optimizers.Adam(learning_rate=0.0001)
    optimizer = tf.optimizers.Adam()
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


class PolicyNetwork:
    def __init__(self, w, rows, cols):
        self.model = make_keras_model(w, rows, cols)

    def get_action(self, state):
        nn_state_representation = state.nn_state_representation()
        move_distribution = self.model.predict(nn_state_representation)

        best_move = np.argmax(move_distribution)

        return best_move // (state.board.shape[0] - 2), best_move % (state.board.shape[1] - 2)


class LiteModel:

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

    def get_action(self, state: SimWorld):
        if random.random() > self.epsilon:
            nn_state_representation = state.nn_state_representation()
            move_distribution = self.predict_single(nn_state_representation)
            shape = state.get_board_shape()
            #move_distribution = move_distribution.reshape(state.get_board_shape())
            legal_actions = state.get_legal_actions()
            #action_indices = [shape[1] * action[0] + action[1] for action in legal_actions]
            move_distribution[move_distribution]
            mask = np.ones()


                move_distribution[legal_actions] == True
            #move_distribution = np.where(mask == True, move_distribution, 0)
            move_distribution[mask] = 0
            best_move = np.ndarray.argmax(move_distribution)
            shape_b = state.get_board_shape()
            return best_move // (shape_b[1]), best_move % (shape_b[0])
        else:
            return random.choice(state.get_legal_actions())



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
