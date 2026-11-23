# 完整视频粗阶段LoRA：首轮训练配置与启动记录

日期：2026-11-23。状态：已启动正式训练，尚未完成；实时状态以本机runs目录为准。

## 本轮范围

本地Qwen3-VL-4B-Instruct学习“完整原始视频→完整粗阶段序列JSON”。保留拿起→采摘→调整→采摘→放回五段，动作类别仍为pickup/harvest/adjust/return。仅使用已存在的粗阶段侧文件，不增加逐颗或细动作标注，不训练计数、按数量切阶段或机器人控制。

首轮用头部cam0，便于观察双臂与全局交接；不是三路/四路联合训练，也未迁移π0.5、V-JEPA或部署代码。

## 数据与输入

- 沿用冻结划分：train151，val21，历史测试12和封存测试20未打开视频/标注。
- 172条所用头部视频全部SHA与侧文件绑定身份相符，帧数与FPS匹配，区间连续、reviewed标志为true。既有reviewed标志不等于本轮新增人工逐帧复核。
- 每条是一整段原始MP4；模型原生视频处理器内部2fps采样，保留30fps源帧时间。使用原模型的视频像素预算，不截短录像或用短窗口分类代替整段任务。
- 源帧边界换算成秒并四舍五入为两位小数，最后覆盖给定视频时长；两位小数不代表0.01秒真值。
- 数据生成于data/derived/coarse_sft_20261123_v1，原始数据未修改；只提交构建脚本和schema/摘要，不提交视频与派生清单。

所有172条均为同一种五段顺序，因此必须观察时间边界误差，不能仅以五个标签输出正确宣称学会视觉分段；其他顺序/缺段泛化尚未覆盖。

## 训练参数

| 参数 | 设置 |
| --- | --- |
| 模型 | Qwen3-VL-4B-Instruct，本地权重 |
| LoRA | r=8，alpha=16，dropout=0.05 |
| 训练范围 | 语言层attention与MLP的252个线性层，16,515,072个参数 |
| 冻结 | 原模型与视觉编码器 |
| 精度/注意力 | BF16 / SDPA，gradient checkpointing |
| 训练轮数 | 3 |
| microbatch / 累积 | 1条完整视频 / 4条 |
| optimizer更新总数 | 114 |
| 学习率 | 1e-4，warmup 5%，cosine |
| 梯度裁剪 / weight decay | 1.0 / 0.01 |
| 最大输入 | 32768 tokens，超长报错，不静默截断 |
| checkpoint | 每20步及每轮，验证loss最优另存best |
| 监督范围 | 仅assistant阶段JSON，不计算用户文字或视频占位token的loss |

为了减少长视频显存占用，只计算回答后缀及其前一个预测位置的logits。单元测试已验证其loss和梯度与全序列遮罩的标准causal loss一致，不省略视频上下文。

## 已实际通过的检查

- 1个完整视频训练step：loss约0.22968，梯度有限，LoRA B权重实际更新，显存峰值约14.48GiB。
- adapter与optimizer/scheduler/RNG状态成功保存；重新加载adapter后同样本loss有限（约0.21537）。这只是运行检查，不是独立精度结果。
- 32项单元测试与Ruff F/E9通过，覆盖回答mask、因果位移、完整区间、未知/错误标签拒绝等。
- 正式作业已完成启动前验证：21条teacher-forced loss约0.23355，固定3条完整生成有效输出0/3。不是封存测试集准确率。
- 正式第1步已完成：loss约0.23191、梯度有限、单步约22.61秒/4条视频，显存峰值约15.02GiB。

## 自动验证与产物

每轮在全部21条val计算teacher-forced loss，并在固定3条val做完整自回归输出，记录JSON有效性、阶段数量/类别/顺序、区间连续与覆盖、边界绝对误差和分段IoU。best按验证loss选择；结束后加载best，对全部21条val生成完整结果。loss下降本身不等于边界更准。

同时保存train归一化边界中位数作为不看画面的时间先验，供后续与生成边界误差对照。当前训练器输出模型误差，未将该先验冒充模型性能。

实时文件：

- runs/coarse_lora_20261123_v1.log：进程完整日志。
- runs/coarse_lora_20261123_v1/metrics.jsonl：逐步loss、梯度、显存与验证记录。
- runs/coarse_lora_20261123_v1/status.json：当前状态与更新次数。
- runs/coarse_lora_20261123_v1/checkpoint-*/、epoch-*/、best/：adapter及训练状态。
- runs/coarse_lora_20261123_v1/eval-baseline.json、eval-epoch-*.json、eval-best-full-validation.json：验证与完整回答。

作业以独立进程会话启动，PID与命令见runs/coarse_lora_20261123_launch.json。无需用户实时逐条确认输出；失败会写状态/日志。当前启动器未实现自动重启或命令行断点恢复，保存的optimizer/RNG状态用于后续恢复，不宣称已经自动恢复。

## 复现

```bash
.venv/bin/python scripts/build_coarse_sft.py --inventory runs/data-intake-204-v2/inventory.json --output data/derived/NEW-coarse-sft
.venv/bin/python scripts/train_coarse_lora.py --config configs/coarse_lora_v1.json --data data/derived/coarse_sft_20261123_v1 --output runs/NEW-smoke --smoke
.venv/bin/python scripts/train_coarse_lora.py --config configs/coarse_lora_v1.json --data data/derived/coarse_sft_20261123_v1 --output runs/NEW-full-training
```

环境仅在本仓库.venv增加peft==0.17.1，torch/transformers沿用基线；精确依赖见requirements-training.lock。配置见configs/coarse_lora_v1.json，提示词见prompts/coarse_sft_zh.txt。

启动快照：[机器记录](coarse_lora_20261123_setup.json)。训练仍在运行，不能据此宣称微调后更准确。
