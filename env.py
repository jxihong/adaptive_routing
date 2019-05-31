import numpy as np

NUM_STAGES = 4
MAX_NUM_BLOCKS = 6

def one_hot(n_values, value):
    return np.eye(n_values)[value]

class ResNet_Env():
    def __init__(self):
        self.num_blocks = {0: 3, 
                           1: 4, 
                           2: 6, 
                           3: 3}
        self.state_dim = NUM_STAGES + MAX_NUM_BLOCKS
        
    def reset(self):
        self.stage = 0
        
        stage_1h = one_hot(NUM_STAGES, self.stage)
        prev_block_1h = one_hot(MAX_NUM_BLOCKS, 0)
        
        init_state = np.append(stage_1h, prev_block_1h, axis=-1)
        return init_state
        
    def step(self, action):
        illegal = (action >= self.num_blocks[self.stage])
        if action == self.num_blocks[self.stage] - 1:
            # Move to next block.
            self.stage += 1
            block = 0
        else:
            block = action + 1
        done = (self.stage >= NUM_STAGES) | illegal

        if done:
            state = np.zeros(self.state_dim)
        else:
            stage_1h = one_hot(NUM_STAGES, self.stage)
            prev_block_1h = one_hot(MAX_NUM_BLOCKS, block)
                
            state = np.append(stage_1h, prev_block_1h, axis=-1)

        reward = -1.0 if illegal else 0
        return state, reward, done

    def get_legal_actions(self):
        max_blocks = self.num_blocks[self.stage]
        return range(max_blocks)
