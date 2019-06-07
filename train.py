import tensorflow as tf
import numpy as np

import keras
from keras import optimizers
from keras.datasets import cifar10
from keras import backend as K 
from keras.preprocessing.image import ImageDataGenerator

from reinforce import Reinforce
from controller import Controller
from manager import RouteManager
from env import *
import vae
import resnet50


MAX_EPISODES = 200000
MAX_STEPS = 20

def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Input image dimensions.
    input_shape = x_train.shape[1:]
    num_classes = 10

    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    
    return (x_train, y_train), (x_test, y_test)


def main():
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    datagen = ImageDataGenerator(featurewise_center=False,
                                 samplewise_center=False,
                                 featurewise_std_normalization=False,
                                 samplewise_std_normalization=False,
                                 zca_whitening=False,
                                 zca_epsilon=1e-06,
                                 rotation_range=0,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 shear_range=0.,
                                 zoom_range=0.,
                                 channel_shift_range=0.,
                                 fill_mode='nearest',
                                 cval=0.,
                                 horizontal_flip=True,
                                 vertical_flip=False,
                                 rescale=None,
                                 preprocessing_function=None,
                                 data_format=None,
                                 validation_split=0.0)

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    input_dim = x_train.shape[1:]
    num_classes = 10

    state_dim = NUM_STAGES + MAX_NUM_BLOCKS
    action_dim = MAX_NUM_BLOCKS
    
    env = ResNet_Env()
    
    v = vae.VAE(input_shape=input_dim, M=4, N=4)
    v.set_weights('./saved_models/cifar10_discrete_vae_model.h5')
    for layer in v.full_model.layers: # Freeze weights in encoder
        layer.trainable = False
    K.set_value(v.hard, True) # Hard sampling

    controller = Controller(input_dim, state_dim, action_dim, v.encoder)
    controller.set_weights('./saved_models/controller.h5')

    adam = optimizers.Adam(lr=0.0006)
    reinforce = Reinforce(controller, adam, input_dim, state_dim, action_dim)

    weights = './saved_models/cifar10_resnet50_model.h5'
    model = resnet50.ResNet50(input_shape=input_dim,
                              classes=num_classes, 
                              weights=weights)
    manager = RouteManager(model, input_dim, num_classes)

    step = 0
    batch_size = 128
    all_rewards = []
    for x, y in datagen.flow(x_train, y_train, batch_size=1):        
        x = np.squeeze(x, axis=0)
        y = np.squeeze(y, axis=0)[0]
        hidden = controller.encoder.predict(x[np.newaxis, :])

        reinforce.setInitialHidden(np.squeeze(hidden, axis=0))

        state = env.reset()        
        for t in range(MAX_STEPS):
            mask = env.get_mask_actions()
            action = reinforce.sampleAction(state, mask)
            next_state, reward, done = env.step(action)
            reinforce.storeRollout(state, action, reward)
            state = next_state
            if done: break
            
        reward = manager.get_reward(x, y, reinforce.state_buffer)            
        reinforce.reward_buffer[-1] += reward # Replace last reward

        reinforce.updateModel(x)

        all_rewards.append(reward)

        step += 1        
        if step and step % batch_size == 0:
            print('Step: {}, Mean Reward: {}'.format(step, np.mean(all_rewards)))
            all_rewards = []
            controller.save_weights('./saved_models/controller.h5')

        if step == MAX_EPISODES: break
        
if __name__ == '__main__':
    main()
