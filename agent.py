# agent.py
import numpy as np
import torch
from networks import ActorNetwork, CriticNetwork
from buffer import ReplayBuffer
import torch.nn.functional as F
import os


class Agent:
    def __init__(self, name, alpha, beta, input_dims, n_actions, gamma=0.95, max_size=1000000, fc1_dims=64, fc2_dims=64,
                 batch_size=64):
        """
        :param name: 智能体名称
        :param alpha: actor网络学习率
        :param beta: critic网络学习率
        :param input_dims: 状态空间维度S
        :param n_actions: 动作空间维度A
        :param gamma: 折扣因子
        :param max_size: 缓冲区存储的最大样本容量
        :param fc1_dims: 第一层全连接层的神经元数量
        :param fc2_dims: 第二层全连接层的神经元数量
        :param batch_size: 每次训练时使用的样本数量
        """

        self.name = name
        self.gamma = gamma
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        # 每个智能体有自己的actor网络,它根据自己的局部观察输出动作
        self.actor = ActorNetwork(alpha, input_dims, n_actions, fc1_dims, fc2_dims, name=name + '_actor')

        # Critic网络是集中训练、分布执行,在训练时接收所有智能体的观察和动作
        self.critic = CriticNetwork(beta, input_dims * 6, n_actions * 6, fc1_dims, fc2_dims, name=name + '_critic')

        # 和actor网络维度一样
        self.target_actor = ActorNetwork(alpha, input_dims, n_actions, fc1_dims, fc2_dims, name=name + '_target_actor')

        # 和critic网络维度一样
        self.target_critic = CriticNetwork(beta, input_dims * 6, n_actions * 6, fc1_dims, fc2_dims,
                                           name=name + '_target_critic')

        # 每个智能体创建自己的ReplayBuffer
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        self.update_network_parameters(tau=0.01)

        # 每回合开始需要清空
        self.critic_loss_history = []  # 新增Critic损失记录列表
        self.actor_loss_history = []  # 可选：记录Actor损失

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = 0.01

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        actor_state_dict = dict(actor_params)
        critic_state_dict = dict(critic_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_state_dict = dict(target_critic_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                                      (1 - tau) * target_critic_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                     (1 - tau) * target_actor_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    # 该方法接收当前的观测observation,返回一个根据策略网络（Actor）生成的动作
    def choose_action(self, observation, evaluate=False):
        state = torch.from_numpy(observation).float().to(self.actor.device)
        actions = self.actor.forward(state)
        if not evaluate:
            actions = actions + torch.randn_like(actions) * 0.1  # 添加探索噪声(生成与动作张量相同形状的标准正态噪声,0.1是标准差)
            actions = torch.clamp(actions, 0, 1)  # 截断,限制在[0,1]范围内
        return actions.detach().cpu().numpy()

    def learn(self, agents):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, new_states, dones = self.memory.sample_buffer(self.batch_size)
        device = self.actor.device
        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.float).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        new_states = torch.tensor(new_states, dtype=torch.float).to(device)
        dones = torch.tensor(dones, dtype=torch.float).to(device)

        all_states = []  # 6 * batch_size * 230
        all_new_states = []  # 6 * batch_size * 230
        all_actions = []  # 6 * batch_size * 54
        for agent in agents:
            indices = np.random.randint(0, self.memory.mem_cntr, size=self.batch_size)

            # batch_size * 230,即batch_size条state经验
            all_states.append(
                agent.memory.state_memory[indices])  # agent.memory.mem_cntr - self.batch_size:agent.memory.mem_cntr

            # batch_size * 230,即batch_size条new_state经验
            all_new_states.append(agent.memory.new_state_memory[indices])

            # batch_size * 54,即batch_size条action经验
            all_actions.append(agent.memory.action_memory[indices])

        # 64 * 1380(230 * 6)
        all_states = torch.tensor(np.concatenate(all_states, axis=1), dtype=torch.float).to(device)
        # 所有智能体的下一个动作拼接
        all_new_states = torch.tensor(np.concatenate(all_new_states, axis=1), dtype=torch.float).to(device)
        all_actions = torch.tensor(np.concatenate(all_actions, axis=1), dtype=torch.float).to(device)

        all_new_actions = []  # 6*64*54,通过forward函数预测了六个智能体的动作,且对于批次中的每个样本,网络都预测了一个54维的动作向量
        for i, agent in enumerate(agents):
            start = i * agent.actor.input_dims
            end = start + agent.actor.input_dims
            agent_new_state = all_new_states[:, start:end]
            # 前面获取了每个智能体自己的batch_size个经验后,通过forward函数得出batch_size个预测值
            all_new_actions.append(agent.target_actor.forward(agent_new_state))
        all_new_actions = torch.cat(all_new_actions, dim=1)  # 64*324,即拼接起了六个智能体的预测动作用于之后的critic训练

        self.target_critic.eval()  # 将目标critic网络设置为评估模式
        self.critic.eval()  # 将主critic网络设置为评估模式,因为我们暂时只需要forward pass，不需要计算梯度
        target_value = self.target_critic.forward(all_new_states, all_new_actions).to(device)
        critic_value = self.critic.forward(all_states, all_actions).to(device)
        self.critic.train()  # 将主critic网络设置回训练模式,为后续的反向传播和参数更新做准备

        target = rewards.unsqueeze(1) + self.gamma * target_value * (1 - dones.unsqueeze(1))
        critic_loss = F.mse_loss(target, critic_value)
        self.critic_loss_history.append(critic_loss.item())  # 记录当前Critic损失

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        # self._log_gradient_norm(self.critic, 'Critic')  # 打印critic网络的梯度信息

        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(states)
        self_actions = mu
        all_mu_actions = all_actions.clone()
        start = 0
        for agent in agents:
            if agent.name == self.name:
                all_mu_actions[:, start:start + agent.actor.n_actions] = self_actions
            start += agent.actor.n_actions

        actor_loss = -self.critic.forward(all_states, all_mu_actions).mean()
        self.actor_loss_history.append(actor_loss.item())

        actor_loss.backward()
        # self._log_gradient_norm(self.actor, 'Actor')  # 打印actor网络的梯度信息

        self.actor.optimizer.step()

        self.update_network_parameters()

    def _log_gradient_norm(self, network, name=''):
        total_norm = 0
        for p in network.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        print(f"[{self.name}] {name} Gradient Norm: {total_norm:.6f}")

    def save_checkpoint(self):
        """保存 actor 和 critic 的模型参数以及 ReplayBuffer"""
        print(f"Saving checkpoints for {self.name}...")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()
        self.memory.save_buffer(f'tmp/maddpg/{self.name}_buffer.pkl')

    def load_checkpoint(self):
        """加载 actor 和 critic 的模型参数以及 ReplayBuffer"""
        if os.path.exists(self.actor.checkpoint_file) and os.path.exists(self.critic.checkpoint_file):
            print(f"Loading checkpoints for {self.name}...")
            self.actor.load_checkpoint()
            self.critic.load_checkpoint()
            self.target_actor.load_checkpoint()
            self.target_critic.load_checkpoint()
            self.memory.load_buffer(f'tmp/maddpg/{self.name}_buffer.pkl')
        else:
            print(f"No checkpoints found for {self.name}. Starting with a fresh model.")
