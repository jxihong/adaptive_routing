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
        hidden = layers.Dense(hidden_dim, activation='relu')(x)
        
        # Get image encoding.
        self.encoder = Model(inputs=img_input, outputs=hidden)

        # Policy network.
        pi_input = layers.Input(shape=(None, self.state_dim))
        policy_gru = layers.GRU(hidden_dim, return_state=True, return_sequences=True)
        policy_dense = layers.Dense(self.action_dim, activation='softmax')

        outs, _ = policy_gru(pi_input, initial_state = hidden)
        pis = policy_dense(outs)
        
        self.model = Model(inputs=[img_input, pi_input], outputs=pis)    
        
        # Just handles the policy.
        hidden_input = layers.Input(shape=(hidden_dim,))
        
        outs, hiddens = policy_gru(pi_input, initial_state = hidden_input)
        pis = policy_dense(outs)

        self.policy = Model(inputs=[pi_input, hidden_input], outputs=[pis, hiddens])

    def step(self, state, hidden):
        shape = state.shape

        if len(shape) == 2:
            assert shape[1] == self.state_dim, "{} != {}".format(shape[1], self.state_dim)
            state = np.expand_dims(state, axis=1)

        pi, next_hidden = self.policy.predict([state, hidden])
        # Remove timestep dimension.
        pi = np.squeeze(pi, axis=1)

        assert pi.shape[1] == self.action_dim, "{} != {}".format(pi.shape[1], self.action_dim)
        return pi, next_hidden
