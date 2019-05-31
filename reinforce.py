import numpy as np
import random

from keras import layers
from keras.models import Model
from keras import backend as K
from keras import utils as np_utils
from keras import optimizers

import tensorflow as tf


class Reinforce():
    def __init__(self, 
                 controller, 
                 optimizer, 
                 input_dim,
                 state_dim,
                 action_dim,
                 init_exp=0.5,         # initial exploration prob
                 final_exp=0.0,        # final exploration prob
                 anneal_steps=10000,   # N steps for annealing exploration
                 discount_factor=0.99, # discount future rewards
                 reg_param=0.001):      # regularization constants
        self.optimizer = optimizer
        
        # controller network
        self.controller = controller 
        
        # training parameters
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount_factor = discount_factor
        self.reg_param = reg_param

        # exploration parameters
        self.exploration  = init_exp
        self.init_exp     = init_exp
        self.final_exp    = final_exp
        self.anneal_steps = anneal_steps

        self.action_buffer = []
        self.reward_buffer = []
        self.state_buffer = []

        # record reward history for normalization
        self.all_rewards = []
        self.max_reward_length = 10000

        self.train_iteration = 0
        self.__build_train_fn()

    def __build_train_fn(self):
        pi_placeholder = self.controller.model.output
        actions_onehot_placeholder = K.placeholder(shape=(None, self.action_dim),
                                                   name="actions_onehot")
        reward_placeholder = K.placeholder(shape=(None),
                                           name="discount_reward")

        logpi = K.log(K.squeeze(pi_placeholder, axis=0))
        log_action_prob = K.sum(logpi * actions_onehot_placeholder, axis=1)

        pg_loss = K.mean(-log_action_prob * reward_placeholder)
        reg_loss = K.sum(logpi * logpi)
        loss = pg_loss + self.reg_param * reg_loss

        updates = self.optimizer.get_updates(params=self.controller.model.trainable_weights,
                                             loss=loss)

        self.train_fn = K.function(inputs=self.controller.model.inputs + \
                                       [actions_onehot_placeholder, reward_placeholder],
                                   outputs=[],
                                   updates=updates)

    def setInitialHidden(self, init_hidden):
        self.hidden = init_hidden

    def sampleAction(self, state, actions=None):
        if not actions:
            actions = range(self.action_dim)

        if random.random() < self.exploration:
            return np.random.choice(actions)
        else:
            pi, next_hidden = self.controller.step(state[np.newaxis, :], 
                                                   self.hidden[np.newaxis, :])
            self.hidden = np.squeeze(next_hidden, axis=0)
            pi = np.squeeze(pi, axis=0)

            # Get mask of legal actions
            mask = np.zeros(self.action_dim)
            for action in actions:
                mask[action] = 1
            # Get new normalized policy
            pi = pi * mask 
            pi /= np.sum(pi)
            return np.random.choice(np.arange(self.action_dim), p=pi)

    def storeRollout(self, state, action, reward):
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.state_buffer.append(state)

    def annealExploration(self, stategy='linear'):
        ratio = max((self.anneal_steps - self.train_iteration)/float(self.anneal_steps), 0)
        self.exploration = (self.init_exp - self.final_exp) * ratio + self.final_exp

    def cleanUp(self):
        self.action_buffer = []
        self.reward_buffer = []
        self.state_buffer  = []

    def updateModel(self, input):
        assert input.shape == self.input_dim 
        
        N = len(self.reward_buffer)
        r = 0 # use discounted reward to approximate Q value

        # compute discounted future rewards
        discounted_rewards = np.zeros(N)
        for t in reversed(range(N)):
            # future discounted reward from now on
            r = self.reward_buffer[t] + self.discount_factor * r
            discounted_rewards[t] = r

        # reduce gradient variance by normalization
        self.all_rewards += discounted_rewards.tolist()
        self.all_rewards = self.all_rewards[:self.max_reward_length]
        if len(self.all_rewards) >= 2:
            discounted_rewards -= np.mean(self.all_rewards)
            discounted_rewards /= np.std(self.all_rewards)

        actions_onehot = np_utils.to_categorical(self.action_buffer, num_classes=self.action_dim)

        states = np.array(self.state_buffer)
        assert states.shape[1] == self.state_dim, "{} != {}".format(states.shape[1], self.state_dim)
        assert actions_onehot.shape[0] == states.shape[0], "{} != {}".format(actions_onehot.shape[0], states.shape[0])
        assert actions_onehot.shape[1] == self.action_dim, "{} != {}".format(actions_onehot.shape[1], self.action_dim)
        assert len(discounted_rewards.shape) == 1, "{} != 1".format(len(discounted_rewards.shape))

        input = input[np.newaxis, :]
        states = states[np.newaxis, :]

        self.train_fn([input, states, actions_onehot, discounted_rewards])

        self.train_iteration += 1
        self.annealExploration()

        # clean up
        self.cleanUp()
