import tensorflow as tf
import numpy as np
from keras import optimizers

from reinforce import Reinforce
from controller import Controller
from env import ResNet_Env

MAX_EPISODES = 10000
MAX_STEPS = 200

# TODO
if __name__ == '__main__':
    input_dim = (32, 32, 3)
    
    env = ResNet_Env()
    state_dim = env.state_dim
    action_dim = 6

    controller = Controller(input_dim, state_dim, action_dim)
    adam = optimizers.Adam()
    reinforce = Reinforce(controller, adam, input_dim, state_dim, action_dim)
    
    state = env.reset()
    for t in range(MAX_STEPS):
        actions= env.get_legal_actions()
        action = reinforce.sampleAction(state[np.newaxis, :], actions)
        next_state, reward, done = env.step(action)
        
        reinforce.storeRollout(state, action, reward)
        state = next_state
        if done: break

    print(reinforce.state_buffer)
    print(reinforce.action_buffer)
    print(reinforce.reward_buffer)

    reinforce.updateModel(np.zeros((32, 32, 3)))
