import ast
import configparser
from collections import defaultdict
import random

import keras.losses
import tensorflow as tf
from tensorflow import keras as ker


class Critic:
    """
    Class implementing Critic portion of Actor-Critic
    """
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('config.ini')
        self.discount_factor = config["PRIMARY"].getfloat('DISCOUNT_CRITIC')
        self.lr = config["PRIMARY"].getfloat('CRITIC_LR')
        self.edr = config["PRIMARY"].getfloat('EDR_CRITIC')
        self.nn_dims = ast.literal_eval(config["PRIMARY"]['CRITIC_NN_DIMS'])
        self.td_error = 1

        """ Values initialized with small random values, eligibilites with 0"""
        # TODO investigate initialization of values
        self.values = defaultdict(lambda: random.random() * 1)
        self.eligibilities = defaultdict(int)

    def reset_e(self):
        self.eligibilities = defaultdict(int)

    def set_eligibility(self, state, value):
        self.eligibilities[state] = value

    def get_td_error(self):
        return self.td_error

    def update_td_error(self, reward, state, state_next):
        self.td_error = reward + self.discount_factor * self.values[state_next] - self.values[state]

    def update_state_action(self, current_sa):
        self.values.update({k[0]: self.values[k[0]] + self.lr*self.td_error*self.eligibilities[k[0]] for k in current_sa})

    def update_e(self, current_sa):
        self.eligibilities.update({k[0]: self.discount_factor * self.edr * self.eligibilities[k[0]] for k in current_sa})


class CriticNN:
    """
    Class implementing Critic portion of Actor-Critic
    With NN
    """
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('config.ini')
        self.nn_dims = ast.literal_eval(config["PRIMARY"]['CRITIC_NN_DIMS'])
        self.discount_factor = config["PRIMARY"].getfloat('DISCOUNT_CRITIC')
        self.lr = config["PRIMARY"].getfloat('CRITIC_LR')
        self.edr = config["PRIMARY"].getfloat('EDR_CRITIC')
        self.nn_dims = ast.literal_eval(config["PRIMARY"]['CRITIC_NN_DIMS'])
        self.td_error = tf.Variable(1.)

        self.n_outputs = 1
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr)

        self.model = self.__gen_model()
        self.history = []

        """ Values initialized with small random values, eligibilites with 0"""
        # TODO investigate initialization of values
        self.values = defaultdict(lambda: random.random() * 1)

    # TODO implement eligibility traces for NN critic
    def reset_e(self):
        pass

    def set_eligibility(self, state, value):
        pass

    def batch_update_td(self, v_h, r):
        #t_d = # TODO: Fiks denne Axel
        pass

    def get_td_error(self):
        t = tf.keras.backend.eval(self.td_error[0, 0])
        return t

    """ Basic Temporal Differencing without eligibility traces """
    def update_td_error(self, reward, state, state_next):
        state = tf.expand_dims(tf.reshape(tf.convert_to_tensor(state), -1), 0)
        state_next = tf.expand_dims(tf.reshape(tf.convert_to_tensor(state_next), -1), 0)
        with tf.GradientTape() as tape:
            self.td_error = reward + self.discount_factor * self.model(state_next)
            loss = keras.losses.mean_squared_error(state, self.td_error)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # TODO: du holdt på her axel. prøv å bare oppdatere hver k steg.
        return

    def predict(self, state):
        state = tf.expand_dims(tf.reshape(tf.convert_to_tensor(state), -1), 0)
        return self.model(state)

    def update_state_action(self, current_sa):
        pass

    def update_e(self, current_sa):
        pass

    def __gen_model(self):
        opt = 'SGD'
        loss = tf.keras.metrics.mean_squared_error
        opt = eval('ker.optimizers.' + opt)
        model = ker.models.Sequential()  # The model can now be built sequentially from input to output
        model.add(ker.layers.Dense(self.nn_dims[0], activation='relu'))
        for i in self.nn_dims[1:]:
            model.add(ker.layers.Dense(i, activation='relu'))
        model.compile(optimizer=opt(learning_rate=self.lr), loss=loss, metrics=[ker.metrics.categorical_accuracy])
        return model


