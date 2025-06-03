import numpy as np
from scipy.spatial.distance import cdist
import math


class SAGINEnv():
    def __init__(self, num_users=30):
        # 系统参数初始化
        self.area_size = 3000  # 地面区域3000x3000平方米
        self.M = 2  # BS数量
        self.V = 2  # UAV数量
        self.O = 1  # LEO卫星数量
        self.N = 10  # 子载波数
        self.B = 30e6  # 总带宽30MHz
        self.B_sub = self.B / self.N  # 子载波带宽
        self.C_th = 10  # 基站最大容量
        self.num_users = num_users  # 用户数量
        self.users = []
        # 基站当前关联人数
        self.bs_0 = 0
        self.bs_1 = 0
        # 网络组件参数
        self.bs_power = 10  # BS发射功率10dBW,即10w
        self.uav_power = 20  # UAV发射功率20dBW,即100w
        self.leo_power = 30  # LEO发射功率30dBw,即1000w
        self.N0 = 1e-16  # 噪声功率谱密度-130dBm/Hz = 1e-13 mW/HZ = 1e-16 W/Hz
        self.noise = self.B_sub * self.N0  # 噪声也已经被换算成线性值 3e-10 w/hz了
        self.fc = 5e9  # 载波频率5GHz
        self.alpha = 1.5  # 路径损耗指数
        self.R = 6  # 莱斯因子
        self.h0 = 1e-3  # 参考距离信道增益（-30dB = 0.001,即缩小1000倍）
        self.d_min_uav = 100  # UAV最小间距
        # 初始化网络组件位置
        self.uav_altitude = 100  # UAV固定高度100米
        self.leo_altitude = 2e5  # LEO高度200,000米
        self.bs_positions = np.array([[1000, 1000, 0],  # BS1坐标
                                      [2000, 2000, 0]])  # BS2坐标
        # 初始化LEO位置（假设在区域中心正上方）
        self.leo_position = np.array([self.area_size / 2, self.area_size / 2, self.leo_altitude])
        """
            初始化用户位置
            np.random.rand(2):返回[a1,a2],值在[0,1)中取
        """
        self.users_position = [[round(x, 2) for x in np.random.rand(2) * self.area_size] for _ in range(self.num_users)]

        # 初始化无人机位置
        self.uav_positions = np.array([[1000, 1000, self.uav_altitude],  # UAV1坐标
                                       [2000, 2000, self.uav_altitude]])  # UAV2坐标
        # 信道模型参数
        self.c = 3e8  # 光速

        # 记录子载波使用情况
        self.subcarrier_usage = {}

        """
            USER-BS的信道参数设置,scale就是概率密度函数里的σ
            瑞利衰落指的是(信号的幅度r)服从瑞利分布,自然与用户、基站、子载波有关系
            瞬时接受功率P=r^2服从指数分布
        """
        self.h = np.random.exponential(scale=1, size=(self.num_users, self.M, self.N))

        # USER-UAV的信道参数设置
        self.K_factor = self.R / (self.R + 1)  # 莱斯信道参数
        # self.los_phase = [(np.random.uniform(0, 2 * np.pi)) for _ in range(self.num_users)]  # 直射路径的随机相位,均匀分布在[0,2𝜋)之间
        # self.los = [np.sqrt(self.K_factor) * np.exp(1j * self.los_phase[i]) for i in range(self.num_users)]  # 幅度为1的复数信号
        # self.nlos = [(np.sqrt(1 / (self.R + 1)) * (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)) for _ in
        #              range(self.num_users)]  # 复数信号

        # USER-LEO的信道参数设置
        self.h_leo = (((self.c / self.fc) / (4 * np.pi)) ** 2)

        # AL(t)
        self.lambda_kL = 50  # kbits/slot

        # 功率分配剩余余量
        self.bs_allocation_power = [10 for _ in range(self.M)]
        self.uav_allocation_power = [100 for _ in range(self.V)]
        self.leo_allocation_power = 1000

        self.beta1 = 0.1
        self.beta2 = 0.1

        self.slot = 0.1  # 每个step间隔0.1s

    def _init_users(self):
        self.users = []

        # 重置功率资源池
        self.bs_allocation_power = [10 for _ in range(self.M)]
        self.uav_allocation_power = [100 for _ in range(self.V)]
        self.leo_allocation_power = 1000

        slice_types = ['H', 'L', 'C']

        for uid in range(self.num_users):
            self.users.append({
                'id': uid,
                'pos': self.users_position[uid],  # (x, y) 坐标
                'slice_type': slice_types[int(uid / 10)],  # [0,9]是'H'用户,[10,19]是’L‘用户,[20,29]是'C'用户

                'associated_type': None,  # BS / UAV / LEO
                'associated_index': None,  # 哪一个网络组件编号
                'subcarrier': -1,  # -1 表示未分配子载波
                'allocated_power': 0,  # 分配功率（W）

                'g': 0,  # 信道增益
                'interference': 0,  # 干扰
                'sinr': 0,  # SINR
                'r': 0,  # 数据率 Mbps
                'los': np.sqrt(self.K_factor) * np.exp(1j * np.random.uniform(0, 2 * np.pi)),
                'nlos': np.sqrt(1 / (self.R + 1)) * (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2),
                'served': False,  # 是否已被服务

                'direction': round(np.random.uniform(0, 2 * np.pi), 3),  # 初始方向
                'speed': round(np.random.uniform(0, 5), 2)  # 初始速度，0~5m/s
            })
        for user in self.users:
            if user['slice_type'] == 'L':
                user['A_K_L'] = np.random.poisson(lam=self.lambda_kL)  # 服从泊松分布的数据到达速率(kbit)

    # 计算每个用户的信道增益g
    def _get_channels(self, users):
        for user in users:
            if user['associated_type'] == 'BS':
                dists = cdist([user['pos']], self.bs_positions[:, :2])[0]  # [与基站1的距离,与基站2的距离]
                path_loss = (dists ** 2 + 1e-9) ** (-self.alpha / 2)  # 路径损耗增益,距离越大增益越小,也就是倍数越小
                user['g'] = self.h[user['id'], user['associated_index'], user['subcarrier']] * path_loss[
                    user['associated_index']]
            elif user['associated_type'] == 'UAV':
                dists = np.linalg.norm(self.uav_positions - np.append(user['pos'], 0), axis=1)
                path_loss = (dists ** 2) ** (-self.alpha / 2)
                user['g'] = self.h0 * path_loss[user['associated_index']] * np.abs(user['los'] + user['nlos']) ** 2
            elif user['associated_type'] == 'LEO':
                dists = self.leo_altitude  # 用户与卫星的距离近似等于卫星的高度
                # 固定值:2.55*10^(-13)
                user['g'] = self.h_leo * (dists ** -1.5)  # 自由空间路径损耗的指数应为2,但这样所获得的信道增益g太小

    # 干扰项计算(p*h,单位是w)
    def _calculate_interference(self, users):
        for user in users:
            g_sub = user['subcarrier']
            g_associated = user['associated_type']
            # 获得与该用户使用同类网络组件同一子载波的其他用户
            other_users = [
                u for u in users
                if u['associated_type'] == g_associated
                   and u['subcarrier'] == g_sub
                   and u['id'] != user['id']
            ]
            user['interference'] = sum(u['allocated_power'] * user['g'] for u in other_users)

    # SINR(并没有换算成db,是线性值)与数据量(在香农公式中,当sinr换算成线性值时,计算出的就是bps)计算
    def _calculate_others(self, users):
        for user in users:
            if user['served']:
                user['sinr'] = (user['allocated_power'] * user['g']) / (user['interference'] + self.noise)
                # 将bps变成Mbps,常说的百兆网就是100Mbps/s
                user['r'] = (self.B_sub * math.log2(1 + user['sinr'])) / 1e6

    # TL用户卸载
    def _priority_based_offloading(self, users):
        """
        根据论文中定义的优先级（TL优先 > NTL，H > L > C），当BS超载时将低优先级用户卸载到UAV/LEO。
        """
        for bs_index in range(self.M):
            bs_users = [u for u in users if u['associated_type'] == 'BS' and u['associated_index'] == bs_index]
            if len(bs_users) > self.C_th:
                # 按照切片优先级排序
                priority_order = {'C': 0, 'L': 1, 'H': 2}
                bs_users.sort(key=lambda u: priority_order[u['slice_type']])
                # 卸载最底层优先级用户
                offload_count = len(bs_users) - self.C_th
                for u in bs_users[:offload_count]:
                    u['associated_type'] = None
                    u['associated_index'] = None
                    u['subcarrier'] = -1
                    u['allocated_power'] = 0
                    u['g'] = 0
                    u['interference'] = 0
                    u['sinr'] = 0
                    u['r'] = 0
                    u['served'] = False

    # 用户位置更新
    def update_location(self):
        """
        每个用户按照自己的速度和方向移动，
        有10%概率改变速度和方向，遇到边界自动反弹。
        """
        for user in self.users:
            # 10% 概率更新方向和速度
            if np.random.rand() < 0.1:
                user['direction'] = round(np.random.uniform(0, 2 * np.pi), 2)
                user['speed'] = round(np.random.uniform(0, 5), 2)

            # 计算原始偏移
            dx = user['speed'] * np.cos(user['direction']) * self.slot
            dy = user['speed'] * np.sin(user['direction']) * self.slot

            new_x = round((user['pos'][0] + dx), 2)
            new_y = round((user['pos'][1] + dy), 2)

            bounced = False

            # 判断 x 越界
            if new_x < 0 or new_x > self.area_size:
                user['direction'] = round(np.pi - user['direction'], 2)  # 水平反射
                bounced = True

            # 判断 y 越界
            if new_y < 0 or new_y > self.area_size:
                user['direction'] = -user['direction']  # 垂直反射
                bounced = True

            # 如果反弹了，重新计算 dx, dy
            if bounced:
                dx = user['speed'] * np.cos(user['direction']) * self.slot
                dy = user['speed'] * np.sin(user['direction']) * self.slot

            # 更新位置并 clip 保证合法
            user['pos'][0] = np.clip(round(user['pos'][0] + dx, 2), 0, self.area_size)
            user['pos'][1] = np.clip(round(user['pos'][1] + dy, 2), 0, self.area_size)
