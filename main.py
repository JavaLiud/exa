# main.py
import numpy as np
from sagin_env import SAGINEnv
from train_env import SAGINTwoLayerWrapper
from agent import Agent
import os
import matplotlib.pyplot as plt


def main():
    # 创建保存模型的目录
    if not os.path.exists('tmp/maddpg'):
        os.makedirs('tmp/maddpg')

    # 创建保存结果的目录
    if not os.path.exists('result'):
        os.makedirs('result')

    # 初始化环境
    env = SAGINEnv()
    env = SAGINTwoLayerWrapper(env)

    # 为每个智能体创建一个Agent实例
    agents = {}
    for agent_name in env.agents:
        agents[agent_name] = Agent(
            name=agent_name,
            alpha=0.0001,  # actor learning rate
            beta=0.001,  # critic learning rate
            input_dims=env.obs_dims,
            n_actions=env.act_dims,
            gamma=0.95,
            batch_size=64
        )
        # 加载之前保存的模型参数
        agents[agent_name].load_checkpoint()

    best_score = float('-inf')  # 负无穷
    score_history = []

    # 初始化记录数据结构
    actor_losses = {agent: [] for agent in env.agents}
    critic_losses = {agent: [] for agent in env.agents}
    scores = {agent: [] for agent in env.agents}

    n_games = 100
    interval = 20  # 每20个回合保存一次图像

    for i in range(n_games):
        obs = env.reset()
        done = {agent: False for agent in env.agents}

        # 清空loss信息
        for agent in agents.values():
            agent.actor_loss_history = []
            agent.critic_loss_history = []

        score_dict = {
            'TL_H': 0,
            'TL_L': 0,
            'TL_C': 0,
            'NTL_H': 0,
            'NTL_L': 0,
            'NTL_C': 0
        }
        episode_length = 10000
        while not all(done.values()) and episode_length > 0:
            actions = {}
            for agent_name in env.agents:
                action = agents[agent_name].choose_action(obs[agent_name])
                actions[agent_name] = action
            new_obs, rewards, dones, truncs, infos = env.step(actions)

            # 存储(s,a,r)
            for agent_name in env.agents:
                agents[agent_name].store_transition(
                    obs[agent_name],
                    actions[agent_name],
                    rewards[agent_name],
                    new_obs[agent_name],
                    dones[agent_name]
                )

            # 学习
            for agent in agents.values():
                agent.learn(list(agents.values()))

            obs = new_obs
            done = dones

            # 累加每个智能体的分数
            for agent_name in env.agents:
                score_dict[agent_name] += rewards[agent_name]

            episode_length -= 1

        # 记录每个智能体的平均loss
        for agent_name in env.agents:
            # 计算平均actor loss
            if len(agents[agent_name].actor_loss_history) > 0:
                avg_actor_loss = sum(agents[agent_name].actor_loss_history) / len(agents[agent_name].actor_loss_history)
            else:
                avg_actor_loss = 0
            actor_losses[agent_name].append(avg_actor_loss)

            # 计算平均critic loss
            if len(agents[agent_name].critic_loss_history) > 0:
                avg_critic_loss = sum(agents[agent_name].critic_loss_history) / len(
                    agents[agent_name].critic_loss_history)
            else:
                avg_critic_loss = 0
            critic_losses[agent_name].append(avg_critic_loss)

            score_dict[agent_name] /= 10000
            scores[agent_name].append(score_dict[agent_name])

        print(f'episode {i + 1}')
        for agent in agents.values():
            print(agent.name, "的平均actor_loss:",
                  sum(agent.actor_loss_history) / len(agent.actor_loss_history) if len(
                      agent.actor_loss_history) > 0 else 0)
            print(agent.name, "的平均critic_loss:",
                  sum(agent.critic_loss_history) / len(agent.critic_loss_history) if len(
                      agent.critic_loss_history) > 0 else 0)

        for agent_name in env.agents:
            print(f'{agent_name} score: {score_dict[agent_name]:.2f}')
        print("完成")

        # 每20个episode或最后一步生成图像
        if (i + 1) % interval == 0 or i == n_games - 1:
            # 计算当前周期序号
            cycle = (i // interval) + 1
            # 绘制并保存当前周期结果
            plot_results(actor_losses, critic_losses, scores, i, cycle)


def plot_results(actor_losses, critic_losses, scores, current_episode, cycle):
    """绘制结果并保存到result目录"""
    agent_names = list(actor_losses.keys())
    n_episodes = current_episode + 1  # 当前总episode数

    # 绘制actor loss
    plt.figure(figsize=(15, 10))
    for i, agent in enumerate(agent_names):
        plt.subplot(2, 3, i + 1)
        plt.plot(range(n_episodes), actor_losses[agent])
        plt.title(f'{agent} Actor Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(f'result/{cycle}_actor_losses.png')
    plt.close()

    # 绘制critic loss
    plt.figure(figsize=(15, 10))
    for i, agent in enumerate(agent_names):
        plt.subplot(2, 3, i + 1)
        plt.plot(range(n_episodes), critic_losses[agent])
        plt.title(f'{agent} Critic Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(f'result/{cycle}_critic_losses.png')
    plt.close()

    # 绘制score
    plt.figure(figsize=(15, 10))
    for i, agent in enumerate(agent_names):
        plt.subplot(2, 3, i + 1)
        plt.plot(range(n_episodes), scores[agent])
        plt.title(f'{agent} Score')
        plt.xlabel('Episode')
        plt.ylabel('Score')
    plt.tight_layout()
    plt.savefig(f'result/{cycle}_scores.png')
    plt.close()

    print(f"已保存第{cycle}周期的结果图像")


if __name__ == '__main__':
    main()
