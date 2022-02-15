import ast
import configparser
from collections import defaultdict
import random
import tensorflow as tf
from tensorflow import keras as ker

class Critic:

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
        #TODO: usikker på om eligibilites riktig satt

    def update_e(self, current_sa):
        self.eligibilities.update({k[0]: self.discount_factor * self.edr * self.eligibilities[k[0]] for k in current_sa})


class CriticNN:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('config.ini')
        self.nn_dims = ast.literal_eval(config["PRIMARY"]['CRITIC_NN_DIMS'])
        self.discount_factor = config["PRIMARY"].getfloat('DISCOUNT_CRITIC')
        self.lr = config["PRIMARY"].getfloat('CRITIC_LR')
        self.edr = config["PRIMARY"].getfloat('EDR_CRITIC')
        self.nn_dims = ast.literal_eval(config["PRIMARY"]['CRITIC_NN_DIMS'])
        self.td_error = 1

        self.input_shape = [4]
        self.n_outputs = 1

        self.model = self.__gen_model()

        """ Values initialized with small random values, eligibilites with 0"""
        # TODO investigate initialization of values
        self.values = defaultdict(lambda: random.random() * 1)


    # TODO implement eligibility traces for NN critic
    def reset_e(self):
        pass

    def set_eligibility(self, state, value):
        pass

    def get_td_error(self):
        return self.td_error

    """ Basic Temporal Differencing without eligibility traces """
    def update_td_error(self, reward, state, state_next):
        self.td_error = reward + self.discount_factor * self.model(tf.convert_to_tensor([state_next]))
        self.model.fit(tf.convert_to_tensor([state]), self.td_error, verbose=0)
        # TODO: du holdt på her axel. prøv å bare oppdatere hver k steg.

    def update_state_action(self, current_sa):
        pass

    def update_e(self, current_sa):
        pass

    def __gen_model(self):
        opt = 'Adam'
        loss = tf.keras.metrics.mean_squared_error
        opt = eval('ker.optimizers.' + opt)
        model = ker.models.Sequential()  # The model can now be built sequentially from input to output
        model.add(ker.layers.Dense(self.nn_dims[0], activation='relu', input_shape=self.input_shape))
        for i in self.nn_dims[1:]:
            model.add(ker.layers.Dense(i, activation='relu'))

        model.compile(optimizer=opt(learning_rate=self.lr), loss=loss, metrics=[ker.metrics.categorical_accuracy])

        print(model.summary())
        return model


