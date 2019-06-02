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
                 encoder): # embedding size for image sample
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        assert encoder.input_shape[1:] == input_dim
        self.encoder = encoder
        self.__build_model()

    def __build_model(self):
        hidden_dim = self.encoder.output_shape[1]
        hidden = self.encoder.output

        # Policy network.
        pi_input = layers.Input(shape=(None, self.state_dim))
        policy_gru = layers.GRU(hidden_dim, return_state=True, return_sequences=True)
        policy_dense = layers.Dense(self.action_dim, activation='softmax')

        outs, _ = policy_gru(pi_input, initial_state = hidden)
        pis = policy_dense(outs)

        self.model = Model(inputs=[self.encoder.input, pi_input], outputs=pis)    

        # Just handles the policy.
        hidden_input = layers.Input(shape=(hidden_dim,))
        
        outs, next_hiddens = policy_gru(pi_input, initial_state = hidden_input)
        pis = policy_dense(outs)

        self.policy = Model(inputs=[pi_input, hidden_input], outputs=[pis, next_hiddens])

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

    def set_weights(self, filepath):
        self.model.load_weights(filepath)
        
    def save_weights(self, filepath):
        self.model.save_weights(filepath)
