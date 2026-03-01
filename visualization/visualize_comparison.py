"""
DSAC、SAC、PPO 三种算法对比可视化脚本
生成三种算法的训练结果对比图表
"""
import os
import pandas as pd
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (16, 12)

# ==================== 路径配置 ====================
DSAC_DIR = 'log/dsac_pretest_6_4/dsac/Feb26-013005/'
SAC_DIR = 'log/sac_pretest_6_4/sac/Feb26-163116/'
PPO_DIR = 'log/ppo_pretest_6_4/ppo/Feb27-230953/'
OUTPUT_PATH = 'log/dsac_sac_ppo_comparison.png'


def load_algorithm_data(log_dir):
    """加载单个算法的 tensorboard 数据"""
    tensorboard_dir = os.path.join(log_dir, 'tensorboard')
    events_file = os.listdir(tensorboard_dir)[0]
    ea = event_accumulator.EventAccumulator(os.path.join(tensorboard_dir, events_file))
    ea.Reload()
    return ea


def plot_test_reward(ax, ea, name, color):
    """绘制测试奖励曲线"""
    test_rewards = ea.Scalars('test/reward')
    vals = np.array([r.value for r in test_rewards])

    ax.scatter(range(len(vals)), vals, alpha=0.3, s=10, color=color)
    smoothed = np.convolve(vals, np.ones(20)/20, mode='valid')
    ax.plot(range(19, len(vals)), smoothed, color, linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_title(f'{name}: Test Reward')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reward')
    ax.grid(True, alpha=0.3)


def plot_loss(ax, ea, name, color):
    """绘制损失曲线"""
    try:
        actor = ea.Scalars('update/loss/actor')
        actor_vals = [l.value for l in actor]
        smoothed = np.convolve(actor_vals, np.ones(10)/10, mode='valid')
        ax.plot(range(9, len(actor_vals)), smoothed, color, label='actor')
        ax.set_title(f'{name}: Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    except:
        ax.text(0.5, 0.5, 'No loss data', ha='center', va='center', transform=ax.transAxes)


def plot_temperature(ax, log_dir, name):
    """绘制温度对比"""
    csv_files = [f for f in os.listdir(log_dir) if f.endswith('.csv') and 'env_0' in f]
    if not csv_files:
        ax.text(0.5, 0.5, 'No CSV files', ha='center', va='center', transform=ax.transAxes)
        return

    csv_files.sort()
    early_csv = pd.read_csv(os.path.join(log_dir, csv_files[0]))
    late_csv = pd.read_csv(os.path.join(log_dir, csv_files[-1]))

    for g in range(4):
        ax.plot(early_csv['step'], early_csv[f'core_temp_g{g}'], alpha=0.4, linestyle='-', label=f'G{g} early')
        ax.plot(late_csv['step'], late_csv[f'core_temp_g{g}'], alpha=0.4, linestyle='--', label=f'G{g} late')

    ax.axhline(y=298, color='green', linestyle=':')
    ax.set_title(f'{name}: Temperature')
    ax.set_xlabel('Step')
    ax.set_ylabel('Temp (K)')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)


def main():
    print("=" * 60)
    print("DSAC vs SAC vs PPO 训练结果对比")
    print("=" * 60)

    # 创建 3x3 图表
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    # ==================== DSAC ====================
    print("加载 DSAC 数据...")
    dsac_ea = load_algorithm_data(DSAC_DIR)
    plot_test_reward(axes[0, 0], dsac_ea, 'DSAC', 'red')
    plot_loss(axes[0, 1], dsac_ea, 'DSAC', 'red')
    plot_temperature(axes[0, 2], DSAC_DIR, 'DSAC')

    # ==================== SAC ====================
    print("加载 SAC 数据...")
    sac_ea = load_algorithm_data(SAC_DIR)
    plot_test_reward(axes[1, 0], sac_ea, 'SAC', 'blue')
    plot_loss(axes[1, 1], sac_ea, 'SAC', 'blue')
    plot_temperature(axes[1, 2], SAC_DIR, 'SAC')

    # ==================== PPO ====================
    print("加载 PPO 数据...")
    ppo_ea = load_algorithm_data(PPO_DIR)
    ppo_test = ppo_ea.Scalars('test/reward')
    ppo_vals = np.array([r.value for r in ppo_test])
    # 过滤异常值
    valid_mask = (ppo_vals > -50000) & (ppo_vals < 10000)
    ppo_vals_valid = ppo_vals[valid_mask]
    epochs_valid = np.array(range(len(ppo_vals)))[valid_mask]

    ax = axes[2, 0]
    ax.scatter(epochs_valid, ppo_vals_valid, alpha=0.3, s=10, color='purple')
    smoothed = np.convolve(ppo_vals_valid, np.ones(20)/20, mode='valid')
    ax.plot(range(19, len(ppo_vals_valid)), smoothed, 'purple', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_title('PPO: Test Reward')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reward')
    ax.grid(True, alpha=0.3)

    # PPO Loss
    try:
        ppo_loss = ppo_ea.Scalars('update/loss')
        ppo_loss_vals = np.array([l.value for l in ppo_loss])
        ppo_loss_vals = ppo_loss_vals[np.isfinite(ppo_loss_vals)]
        if len(ppo_loss_vals) > 10:
            smoothed = np.convolve(ppo_loss_vals, np.ones(10)/10, mode='valid')
            axes[2, 1].plot(range(9, len(ppo_loss_vals)), smoothed, 'purple', label='total loss')
            axes[2, 1].legend()
    except:
        pass
    
    axes[2, 1].set_title('PPO: Loss')
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('Loss')
    axes[2, 1].grid(True, alpha=0.3)

    plot_temperature(axes[2, 2], PPO_DIR, 'PPO')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"\n对比图已保存到: {OUTPUT_PATH}")

    # ==================== 打印统计信息 ====================
    print("\n=== 三个算法对比 ===")

    dsac_test = dsac_ea.Scalars('test/reward')
    dsac_vals = np.array([r.value for r in dsac_test])
    print(f"DSAC: 最佳={max(dsac_vals):.0f}, 均值={np.mean(dsac_vals):.0f}")

    sac_test = sac_ea.Scalars('test/reward')
    sac_vals = np.array([r.value for r in sac_test])
    print(f"SAC:  最佳={max(sac_vals):.0f}, 均值={np.mean(sac_vals):.0f}")

    print(f"PPO:  最佳={max(ppo_vals_valid):.0f}, 均值={np.mean(ppo_vals_valid):.0f}")


if __name__ == '__main__':
    main()
