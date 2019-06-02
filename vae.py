""" Categorical VAE implementation in Keras.
# Adapted from continuous version at https://github.com/AppliedDataSciencePartners/WorldModels
"""

import numpy as np

import tensorflow as tf

from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.activations import softmax
from keras.losses import mean_squared_error

CONV_FILTERS = [32,64,64,128]
CONV_KERNEL_SIZES = [2,2,2,2]
CONV_STRIDES = [2,2,2,2]
CONV_ACTIVATIONS = ['relu','relu','relu','relu']

DENSE_SIZE = 1024

CONV_T_FILTERS = [64,64,32,3]
CONV_T_KERNEL_SIZES = [3,3,3,4]
CONV_T_STRIDES = [2,2,2,2]
CONV_T_ACTIVATIONS = ['relu','relu','relu','sigmoid']


class VAE():
    def __init__(self, 
                 optimizer=Adam(), 
                 input_shape=(32, 32, 3),
                 M=10,
                 N=2,
                 init_tau=1.0):
        self.optimizer = optimizer
        self.input_dim = input_shape

        self.tau = K.variable(init_tau, name="temperature")
        self.hard = K.variable(False, name="hard")
        
        self.__build(M, N)
        
    def __build(self, M, N):
        vae_x = Input(shape=self.input_dim, name='observation_input')
        # Encoder
        vae_c1 = Conv2D(filters = CONV_FILTERS[0], kernel_size = CONV_KERNEL_SIZES[0], strides = CONV_STRIDES[0], 
                        activation=CONV_ACTIVATIONS[0], 
                        name='conv_layer_1')(vae_x)
        vae_c2 = Conv2D(filters = CONV_FILTERS[1], kernel_size = CONV_KERNEL_SIZES[1], strides = CONV_STRIDES[1], 
                        activation=CONV_ACTIVATIONS[1], 
                        name='conv_layer_2')(vae_c1)
        vae_c3= Conv2D(filters = CONV_FILTERS[2], kernel_size = CONV_KERNEL_SIZES[2], strides = CONV_STRIDES[2],
                       activation=CONV_ACTIVATIONS[2], 
                       name='conv_layer_3')(vae_c2)
        vae_c4= Conv2D(filters = CONV_FILTERS[3], kernel_size = CONV_KERNEL_SIZES[3], strides = CONV_STRIDES[3], 
                       activation=CONV_ACTIVATIONS[3], 
                       name='conv_layer_4')(vae_c3)

        h = Flatten()(vae_c4)
        logits_y = Dense(M * N)(h)

        def gumbel_sampling(logits_y):
            U = K.random_uniform(K.shape(logits_y), 0, 1)
            y = logits_y - K.log(-K.log(U + 1e-20) + 1e-20) # logits + gumbel noise
            y = softmax(K.reshape(y, (-1, N, M)) / self.tau)
            
            # Return one-hot samples
            if self.hard:
                y_hard = K.cast(K.equal(y, K.max(y, -1, keepdims=True)), y.dtype)
                y = K.stop_gradient(y_hard - y) + y

            y = K.reshape(y, (-1, M*N))
            return y

        vae_z = Lambda(gumbel_sampling, name='z')(logits_y)
        vae_z_input = Input(shape=(M * N,), name='z_input')

        # Decoder layers
        vae_dense = Dense(DENSE_SIZE, name='dense_layer')
        vae_z_out = Reshape((1,1,DENSE_SIZE), name='unflatten')
        vae_d1 = Conv2DTranspose(filters = CONV_T_FILTERS[0], kernel_size = CONV_T_KERNEL_SIZES[0] , strides = CONV_T_STRIDES[0], 
                                 activation=CONV_T_ACTIVATIONS[0], 
                                 name='deconv_layer_1')
        vae_d2 = Conv2DTranspose(filters = CONV_T_FILTERS[1], kernel_size = CONV_T_KERNEL_SIZES[1] , strides = CONV_T_STRIDES[1], 
                                 activation=CONV_T_ACTIVATIONS[1], 
                                 name='deconv_layer_2')
        vae_d3 = Conv2DTranspose(filters = CONV_T_FILTERS[2], kernel_size = CONV_T_KERNEL_SIZES[2] , strides = CONV_T_STRIDES[2], 
                                 activation=CONV_T_ACTIVATIONS[2], 
                                 name='deconv_layer_3')
        vae_d4 = Conv2DTranspose(filters = CONV_T_FILTERS[3], kernel_size = CONV_T_KERNEL_SIZES[3] , strides = CONV_T_STRIDES[3], 
                                 activation=CONV_T_ACTIVATIONS[3], 
                                 name='deconv_layer_4')
        
        # Decoder in full model.
        vae_dense_model = vae_dense(vae_z)
        vae_z_out_model = vae_z_out(vae_dense_model)

        vae_d1_model = vae_d1(vae_z_out_model)
        vae_d2_model = vae_d2(vae_d1_model)
        vae_d3_model = vae_d3(vae_d2_model)
        vae_d4_model = vae_d4(vae_d3_model)

        # Decoder by itself
        vae_dense_decoder = vae_dense(vae_z_input)
        vae_z_out_decoder = vae_z_out(vae_dense_decoder)

        vae_d1_decoder = vae_d1(vae_z_out_decoder)
        vae_d2_decoder = vae_d2(vae_d1_decoder)
        vae_d3_decoder = vae_d3(vae_d2_decoder)
        vae_d4_decoder = vae_d4(vae_d3_decoder)

        # All models
        self.full_model = Model(vae_x, vae_d4_model)
        self.encoder = Model(vae_x, vae_z)
        self.decoder = Model(vae_z_input, vae_d4_decoder)

        self.full_model.compile(optimizer = self.optimizer, 
                                loss = mean_squared_error)

    def fit(self, x, x_test,
            num_epochs = 200, 
            batch_size = 128, 
            anneal_rate = 0.0003,
            min_temperature = 0.5,
            callbacks = []):
        for e in range(num_epochs):
            # Fit the model on the batches generated by datagen.flow().
            self.full_model.fit(x, x, 
                                batch_size=batch_size,
                                validation_data=(x_test, x_test),
                                epochs=1, 
                                verbose=1, 
                                shuffle=True,
                                callbacks=callbacks)
            K.set_value(self.tau, np.max([K.get_value(self.tau) * np.exp(-anneal_rate * e), min_temperature]))

    def fit_generator(self, datagen, x, x_test,
                      num_epochs = 200, 
                      batch_size = 128, 
                      anneal_rate = 0.0003,
                      min_temperature = 0.5,
                      callbacks = []):        
        for e in range(num_epochs):
            # Fit the model on the batches generated by datagen.flow().
            self.full_model.fit_generator(datagen.flow(x, x, batch_size=batch_size),
                                          validation_data=(x_test, x_test),
                                          steps_per_epoch = x.shape[0] // batch_size,
                                          epochs=1, 
                                          verbose=1, 
                                          workers=4,
                                          callbacks=callbacks)
            K.set_value(self.tau, np.max([K.get_value(self.tau) * np.exp(-anneal_rate * e), min_temperature]))

    def set_weights(self, filepath):
        self.full_model.load_weights(filepath)

    def save_weights(self, filepath):
        self.full_model.save_weights(filepath)
