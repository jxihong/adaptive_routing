import numpy as np
import tensorflow as tf

import resnet50

from keras import backend as K
import keras.layers as layers
from keras.models import Model

RESNET50_TOTAL_STAGES = 4
RESNET50_TOTAL_BLOCKS = 16

class RouteManager():
    def __init__(self, 
                 input_shape=(32, 32, 3), 
                 num_classes=10, 
                 weights=None):
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.model = resnet50.ResNet50(input_shape=(32, 32, 3), 
                                       classes=num_classes, 
                                       weights=weights)


    def __identity_block_route(self, input_tensor, stage, block):
        # Convert to ResNet50 layer names.
        block_str = ['a', 'b', 'c', 'd', 'e', 'f'][block]        
        conv_name_base = 'res' + str(stage + 2) + block_str  + '_branch'
        bn_name_base = 'bn' + str(stage + 2) + block_str  + '_branch'

        x = self.model.get_layer(conv_name_base + '2a')(input_tensor)
        x = self.model.get_layer(bn_name_base + '2a')(x)
        x = layers.Activation('relu')(x)
        
        x = self.model.get_layer(conv_name_base + '2b')(x)
        x = self.model.get_layer(bn_name_base + '2b')(x)
        x = layers.Activation('relu')(x)

        x = self.model.get_layer(conv_name_base + '2c')(x)
        x = self.model.get_layer(bn_name_base + '2c')(x)

        x = layers.add([x, input_tensor])
        x = layers.Activation('relu')(x)
        return x

    def __conv_block_route(self, input_tensor, stage, block):
        # Convert to ResNet50 layer names.
        block_str = ['a', 'b', 'c', 'd', 'e', 'f'][block]        
        conv_name_base = 'res' + str(stage + 2) + block_str  + '_branch'
        bn_name_base = 'bn' + str(stage + 2) + block_str  + '_branch'

        x = self.model.get_layer(conv_name_base + '2a')(input_tensor)
        x = self.model.get_layer(bn_name_base + '2a')(x)
        x = layers.Activation('relu')(x)

        x = self.model.get_layer(conv_name_base + '2b')(x)
        x = self.model.get_layer(bn_name_base + '2b')(x)
        x = layers.Activation('relu')(x)

        x = self.model.get_layer(conv_name_base + '2c')(x)
        x = self.model.get_layer(bn_name_base + '2c')(x)

        shortcut = self.model.get_layer(conv_name_base + '1')(input_tensor)
        shortcut = self.model.get_layer(bn_name_base + '1')(shortcut)

        x = layers.add([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

    def __build_route(self, states):
        # Shared beginning layers.
        x = self.model.input
        for layer in self.model.layers[:7]:
            x = layer(x)
        
        for s in states:
            stage_1h, block_1h = np.split(s, [RESNET50_TOTAL_STAGES])
            stage = np.argmax(stage_1h)
            block = np.argmax(block_1h)

            if block == 0:
                x = self.__conv_block_route(x, stage, block)
            else:
                x = self.__identity_block_route(x, stage, block)
                
        # Shared top of model.
        for layer in self.model.layers[175:]:
            x = layer(x)

        return K.function(inputs = [self.model.input], 
                          outputs = [x])

    def get_reward(self, input, label, states):
        input = input[np.newaxis, :]
        route_fn = self.__build_route(states)
        
        baseline = np.squeeze(self.model.predict(input), axis=0)
        value = np.squeeze(route_fn([input])[0], axis=0)        
        adv = value[label] - baseline[label]

        skip = (RESNET50_TOTAL_BLOCKS - len(states)) / RESNET50_TOTAL_BLOCKS
        return adv + 0.2 * skip
