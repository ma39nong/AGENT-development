# LORA-AGENT-QWEN

Franka-Sharpa 长程采摘任务的独立 Qwen Agent 仓库：本地推理、LoRA、视觉记忆、任务规划和离线评测。

目标：用户指定总数量和可选顺序约束，Agent 根据真实观测决定继续采摘、调整、重试或放回，不预设 `2,2` 或最多两轮。

## 仓库边界

| 仓库 | 负责内容 |
| --- | --- |
| 本仓库 | Qwen 数据构建、LoRA、推理服务、事件与视觉记忆、条件计划、离线回放和评测 |
| π0.5 训练仓库 | 动作模型、动作与框联合训练、动作模型内部 memory 等独立实验 |
| `ma39nong/franka3-sharpa-pi05` | ROS 观测、普通/Fovea 模型加载、异步动作执行、技能交接、UI、Qwen 服务适配器 |

Agent 可以接普通四路或 Fovea，但输入必须携带实际裁剪来源。Qwen 和 V-JEPA 独立选择。模型内部特征不直接跨模型复用。

## 当前状态

- 新仓库开发分支：`qwen-harvest-agent`，基于原 `main` 初始提交 `42e9d9c`。
- 官方 `Qwen3-VL-4B-Instruct` 权重已在本机下载，两个分片经过官方 SHA-256 校验；权重不提交 Git。
- 本次建立开发要求、接口草案、基线配置和离线权重检查工具。
- 本仓库尚未完成模型推理、LoRA、实时事件识别或自动技能控制。部署仓库现有 Qwen 原型尚未迁移，不将其固定轮次结构作为新设计。

## 开始工作

Python 3.10+，仅检查权重时不需要安装深度学习依赖：

```bash
python3 scripts/check_model.py --model-dir /absolute/path/to/Qwen3-VL-4B-Instruct --sha256
```

默认配置只针对离线基线，不启动机器人或调用云 API。推理和训练需要后续建立独立环境，不能升级或覆盖 π0.5 已有环境。

## 开发依据

- [完整开发要求及验收顺序](docs/development_requirements.md)
- [部署接口草案](docs/deployment_interface.md)
- [本地模型基线配置](configs/qwen3_vl_4b_baseline.json)
- [LoRA 数据与标注约定](data/README.md)

近期优先工作：独立推理环境、真实四路样本基线、人工核查评测集。条件计划先在离线执行模拟器验证，通过后才进行旁路和受限自动执行。
