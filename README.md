# PCB 动态重调度双 Actor PPO 论文项目

基于双 Actor PPO 强化学习架构的 PCB 钻孔车间动态重调度求解系统。

## 项目结构

```text
├── agent/            # 策略网络、HGNN编码器、双Actor与Critic、Rollout Buffer
├── baselines/        # 对比基准算法与调度派发规则
├── configs/          # 实验配置与协议规范
├── data/             # 动态测试算例与生成器
├── environment/      # 钻孔车间动态仿真环境与数据模型
├── experiments/      # 实验管理与评测
├── result/           # 训练日志与模型权重
├── scripts/          # 实用脚本（如 TensorBoard 启动、Excel 结果汇总）
├── train.py          # 分阶段训练统一入口
├── train_stage1.py   # 独立阶段 1 训练入口
├── train_stage2.py   # 独立阶段 2 训练入口
└── train_stage3.py   # 独立阶段 3 训练入口
```

## 训练启动示例

```bash
# 分阶段训练（默认自动检测 GPU）
python train.py --stage 1 --seed 1
python train.py --stage 2 --seed 1
python train.py --stage 3 --seed 1

# 或直接运行阶段独立脚本
python train_stage1.py
python train_stage2.py
python train_stage3.py
```
