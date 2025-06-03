from pettingzoo.utils.env import AECEnv
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class SAGINTwoLayerWrapper(AECEnv):
    def __init__(self, env):
        super().__init__()
        self.env = env

        # 打印环境变量时会显示metadata里面的东西而不是技术细节
        self.metadata = {
            "name": "sagin_env",
            "render_modes": [],
            "is_parallel": False,
        }

        self.slice_types = ['H', 'L', 'C']
        self.layer_types = ['TL', 'NTL']

        # 如"TL_H"
        self.agents = [f"{layer}_{s}" for layer in self.layer_types for s in self.slice_types]

        # 智能体编号映射
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.agents)}

        # 定义每个智能体的状态与动作空间
        h_vector = 10 * self.env.M * self.env.N  # 任意切片有10个用户,每个用户与每个基站的每个子载波的信道增益
        AL_vector = 10  # L切片有10个用户,论文中提到的AL(t)队列,即每个用户在slot_t要传输多少bit,遵循参数为λk,L的泊松过程
        position_vector = 10 * 2  # 用户位置,每个切片分配10个用户,每个用户两个坐标
        o1_dim = h_vector + AL_vector + position_vector
        # 10个用户的(los,nlos)(分为实部和虚部) + 10个用户的(h_leo)都是一样的 + 切片数据压力 + 用户位置 + 未服务比例
        o2_dim = 10 * (2 * 2) + 1 + AL_vector + position_vector + 1  # 72

        # 切片10个用户的网络组件分配符(两个基站)、子载波分配符、功率分配
        a1_dim = 10 * 2 + 10 + 10
        # 两个无人机位置 + 10个用户的(UAV分配符(两个无人机)、UAV分配功率、LEO分配符、LEO分配功率、子载波分配符)
        a2_dim = 2 * 2 + 10 * 2 + 4 * 10

        self.obs_dims = max(o1_dim, o2_dim)  # o1_dim = 230,o2_dim = 72
        self.act_dims = max(a1_dim, a2_dim)  # a1_dim = 40,a2_dim = 64

        """
            {
                "TL_H": Box(low=0, high=1, shape=(10,), dtype=np.float32),
                "TL_L": Box(low=0, high=1, shape=(10,), dtype=np.float32),
                ...
            }
            那么Box(low=0, high=1, shape=(10,), dtype=np.float32)又是什么作用呢?
            这是确保它们能接收到的观测数据在[0, 1]范围内,形状由self._obs_dim()决定,
            最终生成如[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]的向量
        """
        self.observation_spaces = {
            agent: spaces.Box(low=0, high=1, shape=(self.obs_dims,), dtype=np.float32)
            for agent in self.agents
        }
        self.action_spaces = {
            agent: spaces.Box(low=0, high=1, shape=(self.act_dims,), dtype=np.float32)
            for agent in self.agents
        }

        self.TL_H_reward = []
        self.TL_L_reward = []
        self.TL_C_reward = []
        self.NTL_H_reward = []
        self.NTL_L_reward = []
        self.NTL_C_reward = []

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # 重置游戏环境
    def reset(self, seed=None, options=None, return_info=False):
        # 重置用户关联情况
        self.env._init_users()
        return {agent: self._get_obs(agent) for agent in self.agents}

    def _get_obs(self, agent):
        layer, slice_type = agent.split('_')

        # 获得属于这一切片的所有用户
        users = [u for u in self.env.users if u['slice_type'] == slice_type]

        # 获得AKL并归一化
        arrivals = [user['A_K_L'] for user in self.env.users if user['slice_type'] == "L"]
        arrival_max = 71.0  # lambda=50时 3σ约为71
        arrivals_normalized = np.array(arrivals, dtype=np.float32) / arrival_max

        # 获得用户位置并归一化
        positions_normalized = np.array([u['pos'] for u in users]).flatten() / self.env.area_size

        # 第一层状态: h向量 + 到达请求 + 位置
        if layer == 'TL':
            h_flat = []
            for user in users:
                h_flat.extend(self.env.h[user['id']].flatten())  # 获得此用户与2个基站的10个子载波的h,将2*10的数组展平为一维
            # h向量归一化
            h_flat = np.array(h_flat, dtype=np.float32)
            h_max = 4.6  # 指数分布(scale=1) 99%分位数
            h_flat_normalized = h_flat / h_max  # 指数分布归一化
            obs = np.concatenate([h_flat_normalized, arrivals_normalized, positions_normalized])
        # 第二层状态: LOS + NLOS + h_leo + 到达请求 + 位置 + 未服务比例(KNT是该切片在TL层未被服务的用户集合)
        else:
            los_nlos_max = 3.0  # 假设LOS/NLOS分量服从标准正态分布，覆盖99.7%范围

            los_real = np.array([u.get('los', 0).real for u in users])
            los_imag = np.array([u.get('los', 0).imag for u in users])
            nlos_real = np.array([u.get('nlos', 0).real for u in users])
            nlos_imag = np.array([u.get('nlos', 0).imag for u in users])

            los_real_normalized = los_real / np.sqrt(self.env.K_factor)
            los_imag_normalized = los_imag / np.sqrt(self.env.K_factor)
            nlos_real_normalized = nlos_real / (3.72 * np.sqrt(1 / 2 * (self.env.R + 1)))
            nlos_imag_normalized = nlos_imag / (3.72 * np.sqrt(1 / 2 * (self.env.R + 1)))

            # 归一化
            h_leo = np.array([1], dtype=np.float32)  # 包成一维(不能是一个数)
            unserved_ratio = np.array([sum(not u['served'] for u in users) / len(users)], dtype=np.float32)  # 包成一维

            obs = np.concatenate(
                [los_real_normalized, los_imag_normalized, nlos_real_normalized, nlos_imag_normalized, h_leo,
                 arrivals_normalized,
                 positions_normalized, unserved_ratio])

        # 补零使得一二层智能体的维度都相同
        padded_obs = np.zeros(self.obs_dims, dtype=np.float32)
        padded_obs[:len(obs)] = obs

        padded_obs = np.clip(padded_obs, 0.0, 1.0)
        return padded_obs

    def step(self, actions):
        self.env._init_users()
        for agent, action in actions.items():
            self._apply_action(agent, action)

        # 计算信道系数
        self.env._get_channels(self.env.users)
        self.env._calculate_interference(self.env.users)
        self.env._calculate_others(self.env.users)

        # 基站或卫星有时也会因为分配功率太小导致sinr低于阈值,此时就需要卫星了
        # for user in self.env.users:
        #     if (user['associated_type'] == 'UAV' or user['associated_type'] == 'BS') and user['sinr'] < 0.01:
        #         print(user)

        # slot迭代
        self.env.h = np.random.exponential(scale=1, size=(self.env.num_users, self.env.M, self.env.N))
        for user in self.env.users:
            # 更新nlos分量
            user['nlos'] = np.sqrt(1 / (self.env.R + 1)) * (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
            # 更新AKL
            if user['slice_type'] == 'L':
                user['A_K_L'] = np.random.poisson(lam=self.env.lambda_kL)
        # 用户移动
        self.env.update_location()

        obs = {agent: self._get_obs(agent) for agent in self.agents}
        rewards = {agent: self._get_reward(agent) for agent in self.agents}
        dones = {agent: False for agent in self.agents}
        truncs = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return obs, rewards, dones, truncs, infos

    def _get_reward(self, agent):
        layer, slice_type = agent.split('_')
        users = [u for u in self.env.users if u['slice_type'] == slice_type]
        reward = 0
        # TL 层：只看连接的是 BS 的用户
        if layer == 'TL':
            users = [u for u in users if u['associated_type'] == 'BS']
            if slice_type == 'H':
                reward = sum(u['r'] for u in users) / 100  # 吞吐量最大化：单位 100Mbps
                self.TL_H_reward.append(reward)
            elif slice_type == 'L':
                delays = []
                c = self.env.c  # 光速
                for u in users:
                    if u['r'] > 0:
                        # 传播延迟（使用与BS的直线距离）
                        x1, y1 = u['pos']
                        bs = self.env.bs_positions[u['associated_index']]
                        x2, y2 = bs[0], bs[1]
                        d = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                        propagation = d / c
                        # 传输延迟
                        miu = u['r'] * 1e3      # 数据吞吐量
                        transmission = u['A_K_L'] / miu
                        # 排队延迟（避免除0或负值）
                        if miu > self.env.lambda_kL:
                            queue = (self.env.lambda_kL / 2) * (1 / (miu ** 2 - miu * self.env.lambda_kL))
                        else:
                            # 连接地面层的用户如果延迟太高,惩罚会较重
                            queue = 5
                        delay = propagation + transmission + queue
                        delays.append(delay)
                # print("延迟：", delays)
                if len(users) != 0:
                    reward = self.env.beta1 - sum(delays) / len(users)
                else:
                    reward = 0
                self.TL_L_reward.append(reward)
            elif slice_type == 'C':
                # 覆盖数最大化：服务人数
                reward = sum(1 for u in users if u['sinr'] > 0.01)
                self.TL_C_reward.append(reward)
        # NTL 层：只看连接的是 UAV / LEO 的用户
        elif layer == 'NTL':
            users = [u for u in users if u['associated_type'] in ['UAV', 'LEO']]
            if slice_type == 'H':
                reward = sum(u['r'] for u in users) / 100  # 100Mbps
                self.NTL_H_reward.append(reward)
            elif slice_type == 'L':
                delays_NTL = []
                c = self.env.c
                for u in users:
                    if u['r'] > 0:
                        x1, y1 = u['pos']
                        d = 0
                        # 传播延迟：LEO 用高度，UAV 用 3D 距离
                        if u['associated_type'] == 'UAV':
                            uav = self.env.uav_positions[u['associated_index']]
                            d = np.linalg.norm([x1 - uav[0], y1 - uav[1], uav[2]])
                        elif u['associated_type'] == 'LEO':
                            d = self.env.leo_altitude  # 近似地认为是垂直距离
                        propagation = d / c
                        # 传输延迟和排队延迟
                        miu = u['r'] * 1e3
                        transmission = u['A_K_L'] / miu
                        if miu > self.env.lambda_kL:
                            queue = (self.env.lambda_kL / 2) * (1 / (miu ** 2 - miu * self.env.lambda_kL))
                        else:
                            # 连接NTL层的用户如果延迟,惩罚相较TL轻
                            queue = 0.5
                        delay = propagation + transmission + queue
                        delays_NTL.append(delay)
                # print("延迟：", delays_NTL)
                if len(users) != 0:
                    reward = self.env.beta2 - sum(delays_NTL) / len(users)
                else:
                    reward = 0
                self.NTL_L_reward.append(reward)
            elif slice_type == 'C':
                reward = sum(1 for u in users if u['sinr'] > 0.01) / 10
                self.NTL_C_reward.append(reward)
        # print(agent, "的奖励是", reward)
        return reward

    def _apply_action(self, agent, action):
        layer, slice_type = agent.split('_')
        users = [u for u in self.env.users if u['slice_type'] == slice_type]
        # TL 第一层：每个用户2维动作（子载波 + 功率）
        num_users = len(users)
        if layer == 'TL':
            for i, user in enumerate(users):
                # 获得子载波分配符
                associated_index = 0 if action[i] > action[i + num_users] else 1
                subcarrier_val = action[i + num_users * self.env.M]
                power_val = action[i + num_users * (self.env.M + 1)]
                user['allocated_power'] = power_val * self.env.bs_allocation_power[associated_index]
                self.env.bs_allocation_power[associated_index] -= user['allocated_power']
                if user['allocated_power'] > 0.1:
                    user['served'] = True
                    user['associated_type'] = 'BS'
                    user['associated_index'] = associated_index
                    user['subcarrier'] = int(subcarrier_val * self.env.N) % self.env.N
                else:
                    user['allocated_power'] = 0

        # NTL 第二层：UAV位置（前4维） + 每个用户5维动作
        else:
            # 触发卸载
            self.env._priority_based_offloading(self.env.users)
            # 获得未服务用户
            users_unserved = [u for u in self.env.users if u['served'] == False and u['slice_type'] == slice_type]
            # 更新 UAV 位置(满足最小距离限制则更新,否则不动)
            x_new_positions = action[0:2] * self.env.area_size
            y_new_positions = action[2:4] * self.env.area_size
            distance = np.linalg.norm(y_new_positions - x_new_positions)
            if distance >= self.env.d_min_uav:
                self.env.uav_positions[0][:2] = x_new_positions
                self.env.uav_positions[1][:2] = y_new_positions
            base = self.env.M * 2
            for i, user in enumerate(users_unserved):
                uav_0_value = action[base + i]  # UAV_0分配符
                uav_1_value = action[base + num_users + i]  # UAV_1分配符
                uav_power_val = action[base + 2 * num_users + i]  # UAV分配功率
                leo_value = action[base + 3 * num_users + i]  # LEO分配符
                if leo_value > max(uav_0_value, uav_1_value):
                    flag = 0
                    associated_type = 'LEO'
                else:
                    flag = 0 if uav_0_value > uav_1_value else 1
                    associated_type = 'UAV'
                leo_power_val = action[base + 4 * num_users + i]  # LEO分配功率
                subcarrier_val = action[base + 5 * num_users + i]  # 子载波分配符

                # 决策使用 UAV 或 LEO（flag值大的获胜）
                if associated_type == 'UAV':
                    user['allocated_power'] = uav_power_val * self.env.uav_allocation_power[i % self.env.V]
                    self.env.uav_allocation_power[flag] -= user['allocated_power']
                    if user['allocated_power'] > 0.1:
                        user['associated_type'] = 'UAV'
                        user['associated_index'] = flag
                        user['subcarrier'] = int(subcarrier_val * self.env.N) % self.env.N
                        user['served'] = True
                    else:
                        user['allocated_power'] = 0
                else:
                    user['allocated_power'] = leo_power_val * self.env.leo_allocation_power
                    self.env.leo_allocation_power -= user['allocated_power']
                    if user['allocated_power'] > 0.1:
                        user['associated_type'] = 'LEO'
                        user['associated_index'] = flag
                        user['subcarrier'] = int(subcarrier_val * self.env.N) % self.env.N
                        user['served'] = True
                    else:
                        user['allocated_power'] = 0

    def render(self):
        plt.figure(figsize=(10, 5))
        new_array = np.array(self.TL_H_reward)[::1000].tolist()
        plt.plot(new_array, label='Reward per Episode')
        plt.xlabel('Episode')  # 或 'Step'，根据你的记录单位
        plt.ylabel('Reward')
        plt.title('Training Reward Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

    # def render(self):
    #     if not hasattr(self, 'fig'):
    #         self.fig, self.ax = plt.subplots(figsize=(10, 5))
    #         self.line, = self.ax.plot([], [], label='Reward per Episode')
    #         self.ax.set_xlabel('Episode')
    #         self.ax.set_ylabel('Reward')
    #         self.ax.set_title('Training Reward Over Time')
    #         self.ax.legend()
    #         self.ax.grid(True)
    #
    #     self.line.set_xdata(range(len(self.TL_H_reward)))
    #     self.line.set_ydata(self.TL_H_reward)
    #     self.ax.relim()
    #     self.ax.autoscale_view()
    #     plt.pause(0.01)  # 非阻塞刷新
