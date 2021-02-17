import numpy as np
from random import randint
import random
from math import exp
import operator
from tensorflow import keras
import tensorflow as tf
from Modules.SplitGD import SplitGD


class Actor:
    def __init__(self, learning_rate, eligibility_decay, discount_factor, epsilon_decay_degree, strategy, episodes):
        self.episodes = episodes
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.eligibility_decay = eligibility_decay
        self.epsilon_decay_degree = epsilon_decay_degree
        self.strategy = strategy
        self.epsilon = 1
        self.policy = {}
        self.eligibility = {}

    def reset_eligibility(self):
        self.eligibility = {}

    def update_policy(self, state, action, td_error):
        if not (state, action) in self.policy.keys():
            self.policy[state, action] = 0
        self.policy[(state, action)] += self.learning_rate * \
            td_error*self.eligibility[(state, action)]

    def boltzman_scale(self, state):
        actions = {k: v for k, v in self.policy.items() if k[0] == state}
        exp_summed = sum([exp(val) for val in actions.values()])
        for state, action in actions.keys():
            self.policy[(state, action)] = exp(
                self.policy[(state, action)])/exp_summed

    def update_eligibility(self, state, action):
        self.eligibility[(state, action)] *= self.discount_factor * \
            self.eligibility_decay

    def get_action(self, state, episode):
        possible_actions = {k: v for k,
                            v in self.policy.items() if k[0] == state}
        if self.strategy == "GREEDY":

            return max(possible_actions.items(), key=operator.itemgetter(1))[0][1]
        if self.strategy == "EPSILON-GREEDY":
            r = random.random()
            if r < self.epsilon:
                # Choose random action with probability epsilon
                action = list(possible_actions.keys())[
                    randint(0, len(possible_actions.keys())-1)][1]
            else:
                # Choose optimal policy action by probability 1-epsilon
                action = max(possible_actions.items(),
                             key=operator.itemgetter(1))[0][1]
            self.epsilon = 1 - \
                (episode/self.episodes)**self.epsilon_decay_degree
            return action

    def add_new_SAP(self, state, action_list):
        for action in action_list:
            if (state, action) not in self.policy.keys():
                self.policy[(state, action)] = 0


class Critic:
    def __init__(self, learning_rate, eligibility_decay, discount_factor):
        self.learning_rate = learning_rate
        self.eligibility_decay = eligibility_decay
        self.discount_factor = discount_factor
        self.value_function = {}
        self.eligibility = {}

    def update_value_function(self, state, td_error):
        self.value_function[state] += self.learning_rate * \
            td_error*self.eligibility[state]

    def update_eligibility(self, state):
        self.eligibility[state] *= self.discount_factor*self.eligibility_decay

    def calculate_td_error(self, reward, next_state, state):
        if next_state not in self.value_function.keys():
            self.value_function[next_state] = random.uniform(
                -0.01, 0.01)
        if state not in self.value_function.keys():
            self.value_function[state] = random.uniform(
                -0.01, 0.01)
        return reward + self.discount_factor*self.value_function[next_state] - self.value_function[state]

    def reset_eligibility(self):
        self.eligibility = {}


class Critic_NN:
    def __init__(self, learning_rate, eligibility_decay, discount_factor, environment, hiddenlayers):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.eligibility_decay = eligibility_decay

        #####Constructing neural network########
        input_rep = np.hstack(environment.get_state())
        model = keras.Sequential()
        # Input layer
        model.add(keras.layers.Input(shape=input_rep.shape))
        # Hiddenlayers
        for nodes in hiddenlayers:
            model.add(keras.layers.Dense(nodes, activation='tanh'))
        # Outputlayer
        model.add(keras.layers.Dense(1, activation='tanh'))
        loss = keras.losses.mse
        optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        model.compile(loss=loss, optimizer=optimizer,
                      metrics=['categorical_accuracy'])
        # Convert to a Value_Function class, which is a subclass of SplitGD
        self.value_function = Value_Function(
            model, learning_rate, discount_factor, eligibility_decay)

    def calculate_td_error(self, reward, next_state, state):
        next_state = np.hstack(next_state).reshape(1, -1)
        state = np.hstack(state).reshape(1, -1)
        state_value = self.value_function.model.predict(state)[0][0]
        next_state_value = self.value_function.model.predict(next_state)[0][0]
        return reward + self.discount_factor*next_state_value-state_value

    def update_value_function(self, state, td_error):
        state = np.hstack(state)
        self.value_function.estimate = self.value_function.model.predict(state.reshape(1, -1))[
            0][0]
        self.value_function.td_error = td_error
        self.value_function.fit([state], [td_error], verbosity=0)

    def reset_eligibility(self):
        self.value_function.weight_eligibility = None


class Value_Function(SplitGD):
    def __init__(self, kerasmodel, learning_rate, discount_factor, eligibility_decay):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.eligibility_decay = eligibility_decay
        self.weight_eligibility = None
        super().__init__(kerasmodel)

    def modify_gradients(self, gradients):
        gradients = np.array([gradient.numpy() for gradient in gradients])
        if self.weight_eligibility == None:
            self.weight_eligibility = self.discount_factor * \
                self.eligibility_decay + gradients
        else:
            self.weight_eligibility = self.discount_factor * \
                self.eligibility_decay*self.weight_eligibility + gradients
        # Retrieve current weights
        weights = np.array(self.model.get_weights())
        # Modifying weights
        weights += self.learning_rate*self.td_error*self.weight_eligibility
        # Set modified weights
        self.model.set_weights(weights)
        return gradients
