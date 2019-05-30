import tensorflow as tf
import resnet50


RESNET50_DICT = { '2a': range(7, 19),
                  '2b': range(19, 29),
                  '2c': range(29, 39),
                  '3a': range(39, 51),
                  '3b': range(51, 61),
                  '3c': range(61, 71),
                  '3d': range(71, 81),
                  '4a': range(81, 93),
                  '4b': range(93, 103),
                  '4c': range(103, 113),
                  '4d': range(113, 123),
                  '4e': range(123, 133),
                  '4f': range(133, 143),
                  '5a': range(143, 155),
                  '5b': range(155, 165),
                  '5c': range(165, 175) }
    
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

    def build_route(self, states):
        # Shared beginning layers.
        x = self.model.get_layer('max_pooling2d_1').output
        
        for s in states: # TODO
            pass

        # Shared top of model.
        for layer in self.model.layers[175:]:
            x = layer(x)

        return Model(self.model.inputs, x, name='resnet50_route')
            
    def get_reward(self, states):
        route = build_route(states)
        
        # TODO
        return 0
