"""
性能分析脚本 - 用于分析动作数据加载的性能瓶颈

使用方法:
1. 基本用法:
   python scripts/profile_motion_loading.py --task=Isaac-Velocity-Rough-Unitree-G1-v0

2. 自定义 profiling 参数:
   python scripts/profile_motion_loading.py --task=YOUR_TASK --warmup=10 --active=20 --num_steps=100

3. 查看结果:
   tensorboard --logdir=./profiler_logs

说明:
- warmup: 在真正开始 profiling 之前的预热步数（让 CUDA 内核编译完成）
- active: 实际进行 profiling 的步数
- num_steps: 总共运行的步数
- trace_path: Profiler 输出目录（可用 TensorBoard 查看）
"""

import argparse
import sys

# 导入 Isaac Lab 相关模块
from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="Profile motion data loading performance")
parser.add_argument("--task", type=str, default=None, help="Name of the task")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments")
parser.add_argument("--warmup", type=int, default=5, help="Number of warmup steps before profiling")
parser.add_argument("--active", type=int, default=10, help="Number of active profiling steps")
parser.add_argument("--num_steps", type=int, default=100, help="Total number of steps to run")
parser.add_argument("--trace_path", type=str, default="./temp/profiler_logs", help="Path to save profiler traces")

# 添加 AppLauncher 参数
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# 启动应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg
from legged_lab.envs import ManagerBasedAnimationEnv, ManagerBasedAmpEnv

import legged_lab.tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg):
    """Profile motion data loading performance."""
    
    # 设置环境数量
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    
    # 创建环境
    print(f"[INFO] Creating environment: {args_cli.task}")
    env: ManagerBasedAmpEnv = gym.make(
        args_cli.task, 
        cfg=env_cfg,
        render_mode=None
    ).unwrapped
    
    # 检查环境是否有 motion_data_manager
    if not hasattr(env, 'motion_data_manager'):
        print("[ERROR] Environment does not have motion_data_manager")
        env.close()
        return
    
    print(f"[INFO] Environment created with {env.num_envs} environments")
    print(f"[INFO] Motion data manager: {env.motion_data_manager}")
    
    # 重置环境
    print("[INFO] Resetting environment...")
    obs, _ = env.reset()
    
    # 开始 profiling
    print(f"\n{'='*80}")
    print(f"[INFO] Starting profiler...")
    print(f"[INFO] Warmup steps: {args_cli.warmup}")
    print(f"[INFO] Active profiling steps: {args_cli.active}")
    print(f"[INFO] Total steps: {args_cli.num_steps}")
    print(f"[INFO] Trace output: {args_cli.trace_path}")
    print(f"{'='*80}\n")
    
    env.motion_data_manager.start_profiling(
        trace_path=args_cli.trace_path,
        warmup_steps=args_cli.warmup,
        active_steps=args_cli.active
    )
    
    # 运行仿真
    print("[INFO] Running simulation...")
    for step in range(args_cli.num_steps):
        # 执行一步
        obs, reward, terminated, truncated, info = env.step(env.action_manager.action)
        
        # 更新 profiler
        env.motion_data_manager.step_profiler()
        
        if (step + 1) % 10 == 0:
            print(f"  Step {step + 1}/{args_cli.num_steps}")
        
        # 如果达到 active 步数，可以提前停止
        if step >= args_cli.warmup + args_cli.active:
            print(f"[INFO] Reached target profiling steps, stopping early...")
            break
    
    # 停止 profiling
    print("\n[INFO] Stopping profiler...")
    env.motion_data_manager.stop_profiling()
    
    print(f"\n{'='*80}")
    print(f"[INFO] Profiling complete!")
    print(f"[INFO] View results with: tensorboard --logdir={args_cli.trace_path}")
    print(f"{'='*80}\n")
    
    # 关闭环境
    env.close()


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭仿真应用
    simulation_app.close()
