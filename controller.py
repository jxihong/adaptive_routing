import numpy as np

from keras import layers
from keras.models import Model
from keras import backend as K
from keras import utils as np_utils
from keras import optimizers

import tensorflow as tf


class Controller():
    def __init__(self, 
                 input_dim, # image sample size
                 state_dim,
                 action_dim, # action size
                 filters=[64, 32, 32], # 3-layer encoder network
                 hidden_dim=64): # embedding size for image sample
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.__build_model(filters, hidden_dim)

    def __build_model(self, 
                      filters, 
                      hidden_dim):
        filters1, filters2, filters3 = filters

        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1

        # Encoder network
        img_input = layers.Input(shape=self.input_dim)

        x = layers.Conv2D(filters1,
                          kernel_size=(1, 1),
                          padding='same',
                          kernel_initializer='he_normal',
                          strides=(2, 2)) (img_input)
        x = layers.BatchNormalization(axis=bn_axis)(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters2,
                          kernel_size=(3, 3),
                          padding='same',
                          kernel_initializer='he_normal',
                          strides=(1, 1)) (x)
        x = layers.BatchNormalization(axis=bn_axis)(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters3,
                          kernel_size=(1, 1),
                          padding='same',
                          kernel_initializer='he_normal',
                          strides=(2, 2)) (x)
        x = layers.BatchNormalization(axis=bn_axis)(x)
        x = layers.Activation('relu')(x)
        x = layers.Flatten()(x)
        z = layers.Dense(hidden_dim, activation='relu')(x)
        
        # Policy network.
        policy_inputs = layers.Input(shape=(None, self.state_dim))
        policy_gru = layers.GRU(hidden_dim, return_sequences=True)
        policy_dense = layers.Dense(self.action_dim, activation='softmax')

        outs = policy_gru(policy_inputs, initial_state = z)
        pis = policy_dense(outs)
        
        self.model = Model(inputs=[img_input, policy_inputs], outputs=pis)    
        
        # Just handles the policy.
        z_input = layers.Input(shape=(hidden_dim,))
        
        outs = policy_gru(policy_inputs, initial_state = z_input)
        pis = policy_dense(outs)

        self.policy = Model(inputs=[z_input, policy_inputs], outputs=pis)

    def get_pi(self, state):
        shape = state.shape

        if len(shape) == 2:
            assert shape[1] == self.state_dim, "{} != {}".format(shape[1], self.state_dim)
            state = np.expand_dims(state, axis=1)

        z = np.zeros((1, 64)) # TODO: fix
        pi = np.squeeze(self.policy.predict([z, state]))
        assert len(pi) == self.action_dim, "{} != {}".format(len(pi), self.action_dim)
        return pi
