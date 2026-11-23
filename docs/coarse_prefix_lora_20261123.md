# 在线前缀粗阶段微调与观察记忆

日期：2026-11-23。开发目标：修复完整录像LoRA面对未结束录像时提前补出调整或放回的问题。本轮只做在线观察训练，不训练计数、调整决策、Prompt交接或机器人控制。

## 前缀监督

- 只读冻结的151条train、21条val及既有粗阶段标注。既有32条测试和新增5条都不加入训练/验证。
- 首轮试验在每条录像的每个粗阶段中点，生成一个从0到该时刻的物理前缀：预计755 train / 105 val，共860个视频。
- 保留原30fps帧时钟，用H264重新编码前缀；每个前缀验证帧数/FPS并保存SHA，不声称像素字节与源MP4相同。源标注和源视频SHA先与原冻结清单核对。
- 输入只包含观测截止前的真实帧，总时长是当前前缀长度，不泄露完整录像时长、当前阶段标签或未来边界。
- 目标只列已经发生的粗阶段，末段裁到当前截止时间，不补齐五段。末段结束时间不表示技能完成。
- 一个原始录像的全部前缀沿用其原有split，不能把相邻前缀拆到训练和验证两边。
- 当前版本每阶段一个中点，只是快速试验，不覆盖稠密边界附近。后续连续回放才能检查提前判段、抖动和延迟。

## 微调配置

沿用Qwen3-VL-4B-Instruct，加载已有完整录像LoRA的best作为初始adapter，另存新实验。r8/alpha16、冻结视觉与基础模型、BF16/SDPA、microbatch1/累积4、学习率5e-5、1个epoch，共189次optimizer更新。先实际smoke检查梯度和初始非零LoRA的参数变化，再正式启动。

先记录初始adapter在前缀验证上的结果，再训练；每轮对全部105个前缀计算loss，固定3条验证录像的全部15个前缀做生成对照，结束后best对全部105个前缀生成。best仅依据验证loss选择。

分别报告：完整观察历史的JSON/顺序/连续区间有效性、当前动作类别是否正确、当前阶段序号是否正确，以及有效完整历史的内部边界误差。类别harvest可能同时对应第2和第4阶段，因此不能只看类别就认定顺序正确。边界误差的有效子集与单阶段样本不等于总体在线交接质量。

配置：configs/coarse_prefix_lora_v1.json。提示词：prompts/coarse_prefix_sft_zh.txt。生成脚本：scripts/build_coarse_prefix_sft.py。训练器：scripts/train_coarse_lora.py。

## 记忆是什么

目前有两种明确分开的历史：

1. 模型每次读取截至当前的视频前缀，这是本次请求内的视觉历史；没有跨请求KV缓存，不代表模型自行永久记住了场景。
2. 本地SQLite观察记忆，保存任务ID、请求ID、源视频/哈希/截止帧、观测截止时间、结果收到时间、模型版本和原始判断。进程重启后仍在，可按任务和时间查询。

新增模块scripts/observation_memory.py；实际把此前17次因果回放导入runs/observation_memory_20261123.sqlite。所有判断标记为unverified_observation_candidate、control_eligible=false，不把之前误报“放回”的结论变成已确认事件。

查询必须同时限制“看到哪一刻”和“结果在什么时候已经返回”。例如20秒时不会读到观察20秒、但22秒才返回的判断，也不会读取后续录像的结果。迟到旧判断作为历史保留，不覆盖更新的观测；相同请求重复导入幂等，内容变化则报错，不静默改写历史。

这轮记忆用于保存、查询和复核；尚未把查询出的文字摘要喂回模型，也没有实现视觉特征缓存、模型主动检索或已确认计数记忆。这些能力不能由“存了SQLite”推导出来。本轮训练仍直接使用前缀视频，先避免错误历史结论反复强化。

导入及查询入口：

```bash
.venv/bin/python scripts/record_prefix_memory.py --run runs/causal_coarse_preview_20261123_v1 --database runs/observation_memory_20261123.sqlite
```

记忆模块已测试重启持久化、任务隔离、双时间过滤、迟到结果排序、重复请求和身份冲突、未来证据拒绝；前缀目标测试覆盖截断当前阶段、不修改源标注和排除未来阶段。

## 启动命令

```bash
.venv/bin/python scripts/build_coarse_prefix_sft.py --source data/derived/coarse_sft_20261123_v1 --output data/derived/coarse_prefix_sft_20261123_v1 --workers 4
.venv/bin/python scripts/train_coarse_lora.py --config configs/coarse_prefix_lora_v1.json --data data/derived/coarse_prefix_sft_20261123_v1 --output runs/coarse_prefix_lora_20261123_smoke_v1 --smoke
.venv/bin/python scripts/train_coarse_lora.py --config configs/coarse_prefix_lora_v1.json --data data/derived/coarse_prefix_sft_20261123_v1 --output runs/coarse_prefix_lora_20261123_v1
```

## 当前状态

860条前缀视频和目标全部生成并通过帧时钟/目标截断/录像分组检查；实际755 train、105 val。单步训练已通过：loss=0.22791、梯度有限、初始非零LoRA参数实际变化、显存峰值约8.79GiB，checkpoint保存成功。38项单元测试和git diff检查通过。

正式训练已以独立进程启动，PID和命令见runs/coarse_prefix_lora_20261123_launch.json；当前先评估初始adapter的前缀验证基线，再进行189次更新。日志runs/coarse_prefix_lora_20261123_v1.log。训练尚未完成，不能据此声称在线准确率已改善。

本轮配置与数据摘要：[数据摘要](coarse_prefix_lora_20261123_data_summary.json)。观察记忆已实际保存17次回放候选，查询与重启测试通过；未将它们用作训练答案。
