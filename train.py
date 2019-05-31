import tensorflow as tf
import numpy as np
from keras import optimizers

from reinforce import Reinforce
from controller import Controller
from manager import RouteManager
from env import *

MAX_EPISODES = 10000
MAX_STEPS = 200

# TODO
if __name__ == '__main__':
    input_dim = (32, 32, 3)
    
    env = ResNet_Env()
    state_dim = env.state_dim
    action_dim = MAX_NUM_BLOCKS
    hidden_dim = 64

    controller = Controller(input_dim, state_dim, action_dim, hidden_dim=hidden_dim)
    adam = optimizers.Adam()
    reinforce = Reinforce(controller, adam, input_dim, state_dim, action_dim)

    manager = RouteManager()

    for i in range(MAX_EPISODES):
        reinforce.setInitialHidden(np.zeros(hidden_dim))

        state = env.reset()        
        for t in range(MAX_STEPS):
            actions= env.get_legal_actions()
            action = reinforce.sampleAction(state, actions)
            next_state, reward, done = env.step(action)
        
            reinforce.storeRollout(state, action, reward)
            state = next_state
            if done: break

        reward = manager.get_reward(np.zeros((32, 32, 3)), 1, reinforce.state_buffer)

        print(reinforce.state_buffer)
        print(reinforce.action_buffer)
        print(reinforce.reward_buffer)

        reinforce.updateModel(np.zeros((32, 32, 3)))
