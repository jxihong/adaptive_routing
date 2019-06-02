import numpy as np
import tensorflow as tf

from keras import backend as K
import keras.layers as layers
from keras.models import Model, clone_model

RESNET50_TOTAL_STAGES = 4
RESNET50_TOTAL_BLOCKS = 16

RESNET50_DICT = { '2a': 7,
                  '2b': 19,
                  '2c': 29,
                  '3a': 39,
                  '3b': 51,
                  '3c': 61,
                  '3d': 71,
                  '4a': 81,
                  '4b': 93,
                  '4c': 103,
                  '4d': 113,
                  '4e': 123,
                  '4f': 133,
                  '5a': 143,
                  '5b': 155,
                  '5c': 165 }

# Just used as a sanity check.
TRIVIAL_ROUTE = [np.array([1., 0., 0., 0., 1., 0., 0., 0., 0., 0.]), 
                 np.array([1., 0., 0., 0., 0., 1., 0., 0., 0., 0.]), 
                 np.array([1., 0., 0., 0., 0., 0., 1., 0., 0., 0.]), 
                 np.array([0., 1., 0., 0., 1., 0., 0., 0., 0., 0.]), 
                 np.array([0., 1., 0., 0., 0., 1., 0., 0., 0., 0.]), 
                 np.array([0., 1., 0., 0., 0., 0., 1., 0., 0., 0.]), 
                 np.array([0., 1., 0., 0., 0., 0., 0., 1., 0., 0.]), 
                 np.array([0., 0., 1., 0., 1., 0., 0., 0., 0., 0.]), 
                 np.array([0., 0., 1., 0., 0., 1., 0., 0., 0., 0.]), 
                 np.array([0., 0., 1., 0., 0., 0., 1., 0., 0., 0.]), 
                 np.array([0., 0., 1., 0., 0., 0., 0., 1., 0., 0.]), 
                 np.array([0., 0., 1., 0., 0., 0., 0., 0., 1., 0.]), 
                 np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 1.]), 
                 np.array([0., 0., 0., 1., 1., 0., 0., 0., 0., 0.]), 
                 np.array([0., 0., 0., 1., 0., 1., 0., 0., 0., 0.]), 
                 np.array([0., 0., 0., 1., 0., 0., 1., 0., 0., 0.])]

class RouteManager():
    def __init__(self, 
                 model,
                 input_shape=(32, 32, 3), 
                 num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.model = model


    def __identity_block_route(self, input_tensor, stage, block):
        # Convert to ResNet50 layer names.
        block_str = ['a', 'b', 'c', 'd', 'e', 'f'][block]        
        resnet50_key = str(stage + 2) + block_str

        assert resnet50_key in RESNET50_DICT.keys()
        start = RESNET50_DICT[resnet50_key]
        x = self.model.layers[start](input_tensor)
        x = self.model.layers[start + 1](x)
        x = self.model.layers[start + 2](x)
        
        x = self.model.layers[start + 3](x)
        x = self.model.layers[start + 4](x)
        x = self.model.layers[start + 5](x)

        x = self.model.layers[start + 6](x)
        x = self.model.layers[start + 7](x)

        x = self.model.layers[start + 8]([x, input_tensor])
        x = self.model.layers[start + 9](x)
        return x

    def __conv_block_route(self, input_tensor, stage, block):
        # Convert to ResNet50 layer names.
        block_str = ['a', 'b', 'c', 'd', 'e', 'f'][block]        
        resnet50_key = str(stage + 2) + block_str

        assert resnet50_key in RESNET50_DICT.keys()
        start = RESNET50_DICT[resnet50_key]
        x = self.model.layers[start](input_tensor)
        x = self.model.layers[start + 1](x)
        x = self.model.layers[start + 2](x)
        
        x = self.model.layers[start + 3](x)
        x = self.model.layers[start + 4](x)
        x = self.model.layers[start + 5](x)

        x = self.model.layers[start + 6](x)
        x = self.model.layers[start + 8](x)

        shortcut = self.model.layers[start + 7](input_tensor)
        shortcut = self.model.layers[start + 9](shortcut)

        x = self.model.layers[start + 10]([x, shortcut])
        x = self.model.layers[start + 11](x)
        return x

    def __build_route(self, states):
        # Shared beginning layers.
        x = self.model.layers[6].output
        
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
    
        reward = 0.5 if np.argmax(value) == label else -0.5
        baseline_reward = 0.5 if np.argmax(baseline) == label else -0.5

        advantage = reward + value[label] - (baseline_reward + baseline[label])
        return advantage
