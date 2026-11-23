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
- 已建立开发要求、接口草案、基线配置、权重检查、视频样本提取和本地多图推理入口。
- 独立环境已在RTX5090跑通24次真实视频窗口调试推理，发现漏识别和视觉证据引用错误；[基线记录](docs/local_baseline_20261121.md)。
- 已完成首轮完整视频粗阶段LoRA（151 train/21 val，3轮、114次更新）；[训练配置与运行记录](docs/coarse_lora_20261123.md)。本轮不训练计数或自动控制。部署旧Qwen原型尚未迁移。
- 已启动在线前缀LoRA试验（755 train/105 val，1轮）；新增可持久查询的观察记忆，模型尚未主动检索使用。[配置与实际状态](docs/coarse_prefix_lora_20261123.md)。

## 开始工作

Python 3.10+，仅检查权重时不需要安装深度学习依赖：

```bash
python3 scripts/check_model.py --model-dir /absolute/path/to/Qwen3-VL-4B-Instruct --sha256
```

默认配置只针对离线基线，不启动机器人或调用云 API。推理使用本仓库独立`.venv`，LoRA使用本仓库已安装的peft==0.17.1，不能升级或覆盖π0.5已有环境。

本地推理环境与复现命令见[离线基线记录](docs/local_baseline_20261121.md)。当前CLI参数显式控制视角、像素预算和任务上下文，配置JSON是开发基线声明，不会隐式覆盖CLI参数。

## 开发依据

- [完整开发要求及验收顺序](docs/development_requirements.md)
- [部署接口草案](docs/deployment_interface.md)
- [本地模型基线配置](configs/qwen3_vl_4b_baseline.json)
- [LoRA 数据与标注约定](data/README.md)
- [现有204条数据接入核查](docs/data_intake_20261121.md)
- [自动计数候选与复核范围](data/event_annotation_requirements.md)
- [云API首批5条录像初标与后续路线（多图历史试跑）](docs/api_teacher_pilot_20261121.md)
- [原始完整视频直传实测（当前优先路线）](docs/direct_video_api_20261121.md)
- [Flash完整视频3次重复对照](docs/direct_video_flash_20261121.md)
- [用户粗阶段原提示词复测](docs/user_coarse_prompt_retest_20261121.md)
- [本地baseline、流式历史与Prompt切换筛查](docs/baseline_capability_20261121.md)
- [完整视频粗阶段LoRA首轮训练](docs/coarse_lora_20261123.md)
- [28号机器5条新增原始录像的首次划分与画面复核](docs/new_raw_20261123.md)
- [当前LoRA在线前缀回放诊断：提前报阶段问题](docs/coarse_prefix_probe_20261123.md)
- [在线前缀微调与持久观察记忆](docs/coarse_prefix_lora_20261123.md)

当前任务：完整视频粗阶段分段已完成首轮；现增加在线前缀观察训练与持久观察记录，计数及机器人Prompt交接留待后续。训练已完成，21条验证的边界平均绝对误差0.24秒；32条独立测试完成，边界平均绝对误差0.35秒（20条盲测0.29秒）；[详细测试结果](docs/coarse_lora_20261123_test.md)，不能把loss下降或标签顺序正确当作边界完全准确。

当前优先完整视频直传，自然语言回答后再整理候选：

```bash
.venv/bin/python scripts/test_direct_video_api.py --review runs/data-intake-204-v2/review-SOURCE_ID.json --camera cam2 --prompt-file prompts/direct_video_count_zh.txt --output runs/direct-video/SOURCE_ID
```

下面保留旧多图对照入口（关闭思考，不接机器人）：

```bash
.venv/bin/python scripts/annotate_video_api.py --review runs/data-intake-204-v2/review-SOURCE_ID.json --output runs/api-teacher-pilot/SOURCE_ID
```

脚本只接受train清单，SHA核对三路源视频，默认2fps、30秒分块、3秒重叠，三路等比缩放拼接。输出始终为待复核候选，无自动累计去重或训练导出。使用现有密钥文件，密钥不进入输出或Git。
