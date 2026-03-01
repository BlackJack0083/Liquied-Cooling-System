"""
DSAC 训练结果可视化脚本
生成 DSAC 训练的分析图表，包括奖励曲线、损失曲线、温度变化等
"""
import os
import pandas as pd
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (14, 10)

# ==================== DSAC 日志 ====================
LOG_DIR = 'log/dsac_pretest_12_4_500/dsac/Mar01-111818'
OUTPUT_DIR = 'log/dsac_pretest_12_4_500/dsac/Mar01-111818'


def load_tensorboard_data():
    """加载 tensorboard 数据"""
    tensorboard_dir = os.path.join(LOG_DIR, 'tensorboard')
    events_file = os.listdir(tensorboard_dir)[0]
    ea = event_accumulator.EventAccumulator(os.path.join(tensorboard_dir, events_file))
    ea.Reload()
    return ea


def plot_test_reward(ax, ea):
    """绘制测试奖励曲线 - 显示0-500 epoch"""
    test_rewards = ea.Scalars('test/reward')
    test_vals = np.array([r.value for r in test_rewards])

    # 显示所有数据 (0-500 epoch)
    test_epochs = np.arange(len(test_vals))

    # 画折线图
    ax.plot(test_epochs, test_vals, alpha=0.3, linewidth=1, label='raw')
    window = 20
    if len(test_vals) > window:
        smoothed = np.convolve(test_vals, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(test_vals)), smoothed, 'red', linewidth=2, label=f'ma({window})')
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Reward')
    ax.set_title('1. DSAC: Test Reward (0-500 epoch)')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_train_vs_test(ax, log_dir):
    """绘制电流对比 - 使用早期和晚期的CSV数据"""
    csv_files = get_csv_files_by_env(log_dir, 'env_0')

    if len(csv_files) < 2:
        ax.text(0.5, 0.5, 'Not enough CSV files', ha='center', va='center', transform=ax.transAxes)
        return

    early_csv = pd.read_csv(os.path.join(log_dir, csv_files[0]))
    late_csv = pd.read_csv(os.path.join(log_dir, csv_files[-1]))

    early_epoch = csv_files[0].split('_ep_')[1].replace('.csv', '')
    late_epoch = csv_files[-1].split('_ep_')[1].replace('.csv', '')

    # 绘制各组的电流
    for g in range(4):
        col = f'current_g{g}'
        ax.plot(early_csv['step'], early_csv[col], alpha=0.5, label=f'G{g} ep{early_epoch}', linestyle='-')
        ax.plot(late_csv['step'], late_csv[col], alpha=0.5, label=f'G{g} ep{late_epoch}', linestyle='--')

    ax.set_xlabel('Step')
    ax.set_ylabel('Current (A)')
    ax.set_title(f'2. Current: ep{early_epoch} vs ep{late_epoch} (env_0)')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_loss(ax, ea):
    """绘制损失曲线"""
    has_data = False
    # DSAC TensorBoard 使用 update/loss/actor 格式
    for tag, color in [('update/loss/actor', 'blue'), ('update/loss/critic1', 'red'), ('update/loss/critic2', 'orange')]:
        try:
            losses = ea.Scalars(tag)
            vals = [l.value for l in losses]
            if len(vals) > 0:
                has_data = True
                smoothed = np.convolve(vals, np.ones(10)/10, mode='valid')
                ax.plot(range(9, len(vals)), smoothed, label=tag.split('/')[-1], color=color)
        except:
            pass
    if not has_data:
        ax.text(0.5, 0.5, 'No loss data available', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('3. DSAC: Loss Curves (0-100 epoch)')
    ax.legend()
    ax.grid(True, alpha=0.3)


def get_csv_files_by_env(log_dir, env_name='env_0'):
    """获取特定环境的CSV文件列表，按epoch排序"""
    csv_files = [f for f in os.listdir(log_dir) if f.endswith('.csv') and f.startswith(env_name + '_ep_')]
    def get_epoch(f):
        parts = f.replace('.csv', '').split('_ep_')
        return int(parts[1]) if len(parts) == 2 else 0
    csv_files.sort(key=get_epoch)
    return csv_files


def plot_temperature_comparison(ax, log_dir):
    """绘制温度对比 - 使用同一环境的早期和晚期数据"""
    csv_files = get_csv_files_by_env(log_dir, 'env_0')

    if len(csv_files) < 2:
        ax.text(0.5, 0.5, 'Not enough CSV files', ha='center', va='center', transform=ax.transAxes)
        return

    early_csv = pd.read_csv(os.path.join(log_dir, csv_files[0]))
    late_csv = pd.read_csv(os.path.join(log_dir, csv_files[-1]))

    early_epoch = csv_files[0].split('_ep_')[1].replace('.csv', '')
    late_epoch = csv_files[-1].split('_ep_')[1].replace('.csv', '')

    for g in range(4):
        col = f'core_temp_g{g}'
        ax.plot(early_csv['step'], early_csv[col], alpha=0.5, label=f'G{g} ep{early_epoch}', linestyle='-')
        ax.plot(late_csv['step'], late_csv[col], alpha=0.5, label=f'G{g} ep{late_epoch}', linestyle='--')

    ax.axhline(y=298, color='green', linestyle=':', label='target')
    ax.axhspan(293, 303, alpha=0.1, color='green', label='±5K zone')
    ax.set_xlabel('Step')
    ax.set_ylabel('Temperature (K)')
    ax.set_title('4. Temperature Early vs Late (env_0)')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_action_distribution(ax, log_dir):
    """绘制动作分布 - 使用同一环境的早期和晚期数据"""
    csv_files = get_csv_files_by_env(log_dir, 'env_0')

    if len(csv_files) < 2:
        ax.text(0.5, 0.5, 'Not enough CSV files', ha='center', va='center', transform=ax.transAxes)
        return

    early_csv = pd.read_csv(os.path.join(log_dir, csv_files[0]))
    late_csv = pd.read_csv(os.path.join(log_dir, csv_files[-1]))

    early_epoch = csv_files[0].split('_ep_')[1].replace('.csv', '')
    late_epoch = csv_files[-1].split('_ep_')[1].replace('.csv', '')

    ax.scatter(early_csv['flow_rate'], early_csv['inlet_temp'], alpha=0.3, label=f'early ep{early_epoch}', s=10)
    ax.scatter(late_csv['flow_rate'], late_csv['inlet_temp'], alpha=0.3, label=f'late ep{late_epoch}', s=10)
    ax.set_xlabel('Flow Rate (m/s)')
    ax.set_ylabel('Inlet Temp (K)')
    ax.set_title('5. Action Distribution (env_0)')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_temperature_histogram(ax, log_dir):
    """绘制温度分布直方图 - 使用同一环境的早期和晚期数据"""
    csv_files = get_csv_files_by_env(log_dir, 'env_0')

    if len(csv_files) < 2:
        ax.text(0.5, 0.5, 'Not enough CSV files', ha='center', va='center', transform=ax.transAxes)
        return

    early_csv = pd.read_csv(os.path.join(log_dir, csv_files[0]))
    late_csv = pd.read_csv(os.path.join(log_dir, csv_files[-1]))

    early_epoch = csv_files[0].split('_ep_')[1].replace('.csv', '')
    late_epoch = csv_files[-1].split('_ep_')[1].replace('.csv', '')

    target = 298.0
    all_early = []
    all_late = []
    for g in range(4):
        all_early.extend(early_csv[f'core_temp_g{g}'].values)
        all_late.extend(late_csv[f'core_temp_g{g}'].values)

    ax.hist(all_early, bins=30, alpha=0.5, label=f'early ep{early_epoch}', density=True)
    ax.hist(all_late, bins=30, alpha=0.5, label=f'late ep{late_epoch}', density=True)
    ax.axvline(x=target, color='green', linestyle='-', label='target')
    ax.axvline(x=target-5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=target+5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Density')
    ax.set_title('6. Temperature Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)


def main():
    print("=" * 60)
    print("DSAC 训练结果可视化")
    print("=" * 60)
    print(f"数据目录: {LOG_DIR}")

    # 加载数据
    ea = load_tensorboard_data()

    # 创建图表
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # 绘制各子图
    plot_test_reward(axes[0, 0], ea)
    plot_train_vs_test(axes[0, 1], LOG_DIR)
    plot_loss(axes[1, 0], ea)
    plot_temperature_comparison(axes[1, 1], LOG_DIR)
    plot_action_distribution(axes[2, 0], LOG_DIR)
    plot_temperature_histogram(axes[2, 1], LOG_DIR)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'training_analysis_dsac_new.png')
    plt.savefig(output_path, dpi=150)
    print(f"\n图表已保存: {output_path}")

    # 打印统计信息
    test_rewards = ea.Scalars('test/reward')
    test_vals = [r.value for r in test_rewards]

    if test_vals:
        print(f"\n=== DSAC 训练统计 ===")
        print(f"数据点: {len(test_vals)}")
        print(f"最佳 reward: {max(test_vals):.2f}")
        print(f"平均 reward: {np.mean(test_vals):.2f}")


if __name__ == '__main__':
    main()
