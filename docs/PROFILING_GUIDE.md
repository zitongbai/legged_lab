# 性能分析指南 (Performance Profiling Guide)

## 概述

本项目已经集成了 PyTorch Profiler，用于分析动作数据加载的性能瓶颈。通过 profiling，你可以：

1. 识别最耗时的操作
2. 查看 CPU 和 GPU 时间分布
3. 分析内存使用情况
4. 找到优化机会

## 快速开始

### 1. 运行 Profiling

使用提供的脚本进行性能分析：

```bash
# 基本用法
python scripts/profile_motion_loading.py --task=YOUR_TASK_NAME

# 完整示例
python scripts/profile_motion_loading.py \
    --task=Isaac-Velocity-Rough-Unitree-G1-v0 \
    --num_envs=4096 \
    --warmup=5 \
    --active=10 \
    --num_steps=100 \
    --trace_path=./profiler_logs \
    --headless
```

参数说明：
- `--task`: 任务名称
- `--num_envs`: 环境数量（默认 4096）
- `--warmup`: 预热步数（默认 5）- CUDA 内核编译预热
- `--active`: 实际 profiling 步数（默认 10）
- `--num_steps`: 总运行步数（默认 100）
- `--trace_path`: Profiler 输出目录
- `--headless`: 无头模式运行

### 2. 查看结果

#### 方法 1: TensorBoard 可视化（推荐）

```bash
tensorboard --logdir=./profiler_logs
```

然后在浏览器中打开 `http://localhost:6006`，切换到 "PYTORCH_PROFILER" 标签页。

你可以看到：
- **Overview**: 性能总览，显示 CPU/GPU 利用率
- **Operator View**: 按操作类型排序的时间统计
- **Kernel View**: CUDA 内核级别的统计
- **Trace View**: 时间线视图，显示操作的执行顺序和重叠情况
- **Memory View**: 内存分配和释放的详细信息

#### 方法 2: 命令行输出

Profiling 完成后会自动在命令行打印摘要：

```
================================================================================
Top 10 operations by CUDA time:
================================================================================
---------------------------------  ------------  ------------  ------------  ...
                             Name    Self CPU %      Self CPU   CPU total %  ...
---------------------------------  ------------  ------------  ------------  ...
              get_motion_state          15.23%       2.456ms        45.67%  ...
         indexing_data                  8.91%       1.435ms        12.34%  ...
...
```

## 代码中的 Profiling 标记

### MotionDataTerm.get_motion_state()

该方法被分为几个关键部分：

```python
with torch.profiler.record_function("get_motion_state"):
    with torch.profiler.record_function("calc_frame_blend"):
        # 计算帧索引和混合权重
        ...
    
    with torch.profiler.record_function("indexing_data"):
        # 从数据缓冲区索引数据
        ...
    
    with torch.profiler.record_function("interpolation"):
        # 线性插值和四元数球面插值
        ...
    
    with torch.profiler.record_function("coordinate_transform"):
        # 坐标系转换
        ...
```

### AnimationTerm._fetch_motion_data()

```python
with torch.profiler.record_function("AnimationTerm::_fetch_motion_data"):
    with torch.profiler.record_function("prepare_query"):
        # 准备查询参数
        ...
    
    with torch.profiler.record_function("get_motion_state_call"):
        # 调用 get_motion_state
        ...
    
    with torch.profiler.record_function("store_buffers"):
        # 存储到缓冲区
        ...
```

## 常见性能瓶颈和优化建议

### 1. 索引操作 (indexing_data)

**问题**: 多次重复索引同一数据

**优化方案**:
- 减少索引操作次数
- 使用连续内存访问模式
- 考虑预先计算常用索引

### 2. 插值计算 (interpolation)

**问题**: 大量的 `torch.lerp` 和 `quat_slerp` 操作

**优化方案**:
- 使用 `torch.lerp` 的 in-place 版本（如果可能）
- 对四元数插值使用更快的近似方法
- 批量处理多个组件

### 3. 坐标转换 (coordinate_transform)

**问题**: 多次调用 `quat_apply_inverse`

**优化方案**:
- 减少不必要的坐标转换
- 尽可能在同一坐标系中进行计算
- 批量处理转换操作

### 4. 数据准备 (prepare_query)

**问题**: `reshape` 和 `repeat_interleave` 可能产生额外拷贝

**优化方案**:
- 使用 `view` 替代 `reshape`（当内存连续时）
- 预先分配好形状的张量
- 使用 `contiguous()` 确保内存连续

## 进阶 Profiling

### 在自己的代码中使用

你可以在任何地方添加 profiling 标记：

```python
import torch

# 标记一个函数
with torch.profiler.record_function("my_custom_function"):
    # 你的代码
    result = expensive_computation()
```

### 手动启动和停止 Profiler

如果你想在环境中手动控制 profiling：

```python
# 在环境初始化后
env.motion_data_manager.start_profiling(
    trace_path="./my_profiler_logs",
    warmup_steps=10,
    active_steps=20
)

# 在训练循环中
for step in range(num_steps):
    # ... 训练代码 ...
    env.motion_data_manager.step_profiler()

# 停止并保存结果
env.motion_data_manager.stop_profiling()
```

### Profile 特定组件

如果只想 profile 特定部分：

```python
# 临时启用 profiling
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
) as prof:
    # 运行你想 profile 的代码
    motion_data = motion_data_term.get_motion_state(motion_ids, motion_times)

# 打印结果
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

## 性能优化 Checklist

- [ ] 识别最耗时的操作（使用 TensorBoard 的 Operator View）
- [ ] 检查是否有不必要的数据拷贝（Memory View）
- [ ] 查看 GPU 利用率（Overview）
- [ ] 分析 CPU/GPU 之间的数据传输
- [ ] 检查是否有同步点导致的等待
- [ ] 优化批处理大小
- [ ] 考虑使用 `torch.compile()` (PyTorch 2.0+)
- [ ] 检查是否可以使用 mixed precision

## 常见问题

**Q: 为什么 profiling 会让程序变慢？**

A: Profiler 会记录每个操作的详细信息，这会增加开销。这是正常的，关注相对时间分布而不是绝对时间。

**Q: Warmup 步数应该设置多少？**

A: 一般 5-10 步足够。Warmup 的目的是让 CUDA 内核编译完成，让缓存预热。

**Q: Active 步数应该设置多少？**

A: 10-20 步通常足够获得稳定的统计数据。太多会产生大量数据文件。

**Q: 如何减少 trace 文件大小？**

A: 
- 减少 active 步数
- 只 profile 关键部分
- 使用 `record_shapes=False`
- 使用 `profile_memory=False`

## 参考资源

- [PyTorch Profiler 官方文档](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [PyTorch Profiler with TensorBoard](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html)
- [性能优化最佳实践](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
