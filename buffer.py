# buffer.py
import numpy as np
import os
import pickle


class ReplayBuffer:
    def __init__(self, max_size, obs_dims, n_actions):
        self.mem_size = max_size  # 1000000
        self.mem_cntr = 0

        # 1000000 * 230
        self.state_memory = np.zeros((self.mem_size, obs_dims))
        self.new_state_memory = np.zeros((self.mem_size, obs_dims))

        # 1000000 * 54
        self.action_memory = np.zeros((self.mem_size, n_actions))

        self.reward_memory = np.zeros(self.mem_size)

        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    # 存经验
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    # 取经验
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)  # 随机获取batch_size个样本索引

        # 根据索引获取batch_size个(s,a,r,s)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

    def save_buffer(self, filename):
        """保存 ReplayBuffer 的状态到文件"""
        buffer_data = {
            'mem_cntr': self.mem_cntr,
            'state_memory': self.state_memory,
            'new_state_memory': self.new_state_memory,
            'action_memory': self.action_memory,
            'reward_memory': self.reward_memory,
            'terminal_memory': self.terminal_memory
        }
        with open(filename, 'wb') as f:
            pickle.dump(buffer_data, f)

    def load_buffer(self, filename):
        """从文件加载 ReplayBuffer 的状态"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                buffer_data = pickle.load(f)
            self.mem_cntr = buffer_data['mem_cntr']
            self.state_memory = buffer_data['state_memory']
            self.new_state_memory = buffer_data['new_state_memory']
            self.action_memory = buffer_data['action_memory']
            self.reward_memory = buffer_data['reward_memory']
            self.terminal_memory = buffer_data['terminal_memory']
            print(f"Loaded ReplayBuffer from {filename}")
        else:
            print(f"No ReplayBuffer file found at {filename}")
