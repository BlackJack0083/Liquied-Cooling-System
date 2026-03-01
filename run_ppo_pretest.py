#!/usr/bin/env python
"""
PPO 预实验脚本 - 4组6电池小规模测试

与 DSAC、SAC 对比测试:
- DSAC: 使用 Diffusion 模型生成动作
- SAC: 使用普通高斯策略直接输出动作
- PPO: 使用 Proximal Policy Optimization (on-policy)

修改内容:
1. 使用 DummyVectorEnv (更稳定)
2. 小规模: 4组 x 6电池
3. 使用 PPO 算法 (on-policy)
"""

import os
import torch
import numpy as np
import argparse
import itertools
from torch import nn
import torch.distributions as D
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net, ActorCritic
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from datetime import datetime
from BatteryEnv.multi_battery_env import make_env
from tianshou.utils import RunningMeanStd

module_name = "ppo_policy.pth"

# ============ 小规模预实验参数 ============
num_batteries_per_group = 12   # 与 DSAC/SAC 保持一致
num_groups = 4                # 与 DSAC/SAC 保持一致
episode_steps = 200           # 每轮步数

# 网络参数 - PPO 使用普通 MLP (降低学习率以避免数值爆炸)
actor_lr = 3e-4
critic_lr = 1e-3
actor_hidden_dims = [256, 256]
critic_hidden_dims = [256, 256]
seed = 42

# PPO 专用参数 (调整以避免数值不稳定)
max_grad_norm = 0.5     # 梯度裁剪
gamma = 0.95            # 奖励折扣因子
value_clip = True       # 启用 value clip 避免 value 估计爆炸

# 训练参数 (减少每次更新的数据量以增加稳定性)
epoch = 500          # 减少用于快速验证
step_per_epoch = 400
episode_per_collect = 2   # 减少每次收集的 episode 数
episode_per_test = 3
repeat_per_collect = 4    # 减少更新次数
batch_size = 128         # 减小 batch size
training_num = 4      # 4个并行环境
test_num = 1
wd = 0.005

# 日志
logdir = f"log/ppo_pretest_{num_batteries_per_group}_{num_groups}_ep{epoch}/"
time_now = datetime.now().strftime('%b%d-%H%M%S')
log_path = os.path.join(logdir, 'ppo', str(time_now))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============ 创建环境 ============
# 使用 DummyVectorEnv (更稳定)
env, train_envs, test_envs = make_env(
    num_batteries_per_group=num_batteries_per_group,
    num_groups=num_groups,
    episode_steps=episode_steps,
    log_path=log_path,
    con=True,
    num_train_envs=training_num,
    num_test_envs=test_num,
    use_subproc=False,  # 使用 DummyVectorEnv
    debug=False,
)

print(f"环境创建完成: {num_groups}组 x {num_batteries_per_group}电池")
print(f"状态维度: {env.observation_space.shape}")
print(f"动作维度: {env.action_space.shape}")


# ============ 网络定义 ============
def create_actor_critic(state_shape, action_shape):
    """PPO 使用 Actor-Critic 架构"""
    # Actor: 策略网络
    actor_net = Net(
        state_shape,
        hidden_sizes=actor_hidden_dims,
        activation=nn.Tanh,
        device=device
    )
    actor = ActorProb(
        actor_net,
        action_shape,
        unbounded=False,
        conditioned_sigma=True,
        device=device
    ).to(device)
    
    # Critic: 价值网络
    critic_net = Net(
        state_shape,
        hidden_sizes=critic_hidden_dims,
        activation=nn.Tanh,
        device=device
    )
    critic = Critic(critic_net, device=device).to(device)
    
    actor_critic = ActorCritic(actor=actor, critic=critic)
    optim = torch.optim.Adam(
        actor_critic.parameters(),
        lr=actor_lr,
        weight_decay=wd
    )
    
    return actor, critic, actor_critic, optim


def main():
    # 获取状态和动作维度
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    state_shape = state_shape[0]
    action_shape = action_shape[0]
    print(f"state_dims: {state_shape}")
    print(f"action_dims: {action_shape}")
    
    # 创建网络
    actor, critic, actor_critic, optim = create_actor_critic(state_shape, action_shape)
    
    # 定义动作分布函数 - PPO 需要同时传入 mean 和 log_std
    def dist_fn(mean, log_std):
        return D.Normal(mean, log_std.exp())

    # 创建策略 - 使用 PPO
    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist_fn,
        discount_factor=gamma,  
        action_space=env.action_space,
        max_grad_norm=max_grad_norm,
        value_clip=value_clip,
    )
        
    buffer = VectorReplayBuffer(
        total_size=episode_per_collect * episode_steps,
        buffer_num=training_num,
        ignore_obs_next=True,        
    )

    # 创建Collector 
    train_collector = Collector(
        policy, 
        train_envs, 
        buffer=buffer, 
        exploration_noise=True,         
    )
    
    test_collector = Collector(policy, test_envs)

    # 日志
    tb_log_path = os.path.join(log_path, 'tensorboard')
    writer = SummaryWriter(tb_log_path)
    logger = TensorboardLogger(writer, update_interval=1)

    # 训练函数
    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, module_name))

    # 开始训练
    print("\n" + "="*50)
    print("开始PPO训练! 查看日志: tensorboard --logdir=log/ppo_pretest_6_4/")
    print("="*50 + "\n")

    result = onpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=epoch,
        step_per_epoch=step_per_epoch,
        step_per_collect=None,
        episode_per_collect=episode_per_collect,
        repeat_per_collect=repeat_per_collect,
        episode_per_test=episode_per_test,
        batch_size=batch_size,
        save_best_fn=save_best_fn,
        logger=logger,
        test_in_train=False,
    )

    print(f"\n训练完成!")
    print(f"最佳测试 reward: {result['best_reward']}")

    # 保存最终模型
    torch.save(policy.state_dict(), os.path.join(log_path, 'final_policy.pth'))
    print(f"模型已保存到: {os.path.join(log_path, 'final_policy.pth')}")


if __name__ == '__main__':
    main()
