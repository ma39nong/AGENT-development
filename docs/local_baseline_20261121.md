# 本地Qwen调试基线：2026-11-21

## 状态和结论

独立环境、原始视频采样和本地多图推理已跑通，未启动训练、云API或机器人。
本轮24次生成来自两个旧录像的四个已看过调试窗口，不是独立测试集。
速度降低并不代表事件识别可靠：三路高分辨率组的窗口标签和结构检查全部吻合，但引用的释放证据不成立。
因此本轮不能作为自动阶段切换准入，不能将8/8标签吻合写成100%准确率。

汇总和原始文本/证据引用见[机器可读摘要](baseline_20261121_summary.json)。
完整图像、源视频哈希、输入manifest和运行报告保存在本地忽略目录`runs/`；无权重/视频提交Git。

## 环境

- Python3.11.16，torch2.7.1+cu128，torchvision0.22.1+cu128，transformers4.57.6。
- RTX5090 32GB，BF16，SDPA，Instruct，不开启深度思考，不采样，输出上限160token。
- 官方模型revision：`ebb281ec70b05090aa6165b016eac8ec08e71b17`；两个分片此前已核验官方SHA256。
- 安装输入为`requirements-runtime.in`，实际依赖固定为`requirements-runtime.lock`，未改π0.5或V-JEPA环境。

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-runtime.lock \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match
```

## 样本和处理

两个成功投放窗口、一个尚持果窗口和一个空手回程/抓取尝试窗口，分别重复两次贪心生成。
数据为旧批次Sharpa录像，四个窗口的标签来自此前人工观看，未做独立复核。
取完整窗口每0.5秒一帧，各视角按源时间排序，不用已知事件位置挑选关键帧。
映射为cam0头部、cam1右腕、cam2左腕；三路6秒窗口39图，3秒窗口21图。
四路增加同帧头部裁剪，分别52/28图。

第四路采用归一化固定框`[0.25,0.22,0.83,0.86]`，原图直接矩形裁剪，标记`fixed_diagnostic`。
它只验证当前真实像素的同源裁剪输入，不是定位器输出、Fovea输出，也不复现Fovea的1.5倍正方形预处理。
按数据集帧号/FPS记录源时间，不将其当作在线ROS时间或验证过的跨相机同步。
标签文件未传入模型。任务目标5/已有4是合成上下文，用于接口测试，不是本次真实执行进度。

## 结果

| 配置 | 生成次数 | 图像文件读取至解码完成中位数 | 范围 | torch峰值分配显存 |
| --- | --- | --- | --- | --- |
| 四路，每图最多65536像素 | 8 | 0.943秒 | 0.615–1.305秒 | 9.53GiB |
| 四路，每图最多262144像素 | 8 | 2.189秒 | 1.208–2.378秒 | 11.44GiB |
| 三路，每图最多262144像素 | 8 | 1.948秒 | 1.038–2.088秒 | 11.02GiB |

最大像素是处理器预算而非每张图片实际像素；实际token网格在运行报告中。
torch分配峰值不包含全部驱动/其他进程占用；本轮未与π0.5并发。
首次生成计入范围，模型加载约1.5秒单列；每次请求计时包含PNG读取、预处理、传GPU和完整生成/解码。
不包含从原视频解码提取PNG、实时采样等待、事件证据出现时间或动作交接；不能当作事件到反应时间。
不同视角配置改变了图片数及总token预算，不能据此因果断言第四路有害。

### 识别与证据

- 四路低像素：8次中4次未知，成功片段分别返回未知或无成功；成功事件均未获得有效确认。
- 四路高像素：一个成功片段返回无成功，另一个报成功但证据时间不严格递增，被结构校验拒绝；重复结果相同。
- 三路高像素：8次窗口标签及结构通过，但人工查看所引证据后，两个成功片段的释放证据均不成立；重复结果相同。
- success_a引用16.5秒左腕画面作释放，该画面果实仍在手上；success_b引用61秒主图作释放，早于此前观察到的投放。
- 所有输出均建议observe；下一技能决策质量尚未完成评测，动态计划能力未得到证明。

结构通过只证明字段、引用存在和时间递增；不证明画面语义。程序不会把建议发送到机器人。
初始四路低像素报告使用`decision_usable`旧字段，语义仅为完整输出及结构通过；随后改名为`output_complete_and_structural`并显式记录`control_eligible=false`，避免误解为动作许可。
该初跑未记录runner源码SHA，后两组已记录；三组均记录prompt SHA、环境、图像网格及源图哈希。

## 复现入口

所有输出目录必须不存在，工具拒绝覆盖旧运行。

```bash
.venv/bin/python scripts/prepare_replay.py \
  --dataset /absolute/path/to/package --episode 29 --start 16 --end 22 \
  --period 0.5 --fixed-crop 0.25 0.22 0.83 0.86 \
  --output runs/new-inputs/success_a

.venv/bin/python scripts/local_baseline.py \
  --model-dir /absolute/path/to/Qwen3-VL-4B-Instruct \
  --manifest runs/new-inputs/success_a/manifest.json \
  --include-crop --max-pixels 262144 --target 5 --confirmed 4 \
  --repeat 2 --output runs/new-local-baseline

.venv/bin/python -m unittest discover -s tests -v
```

下一步先扩充并独立核查事件/证据数据，保留失败样本，建立按完整录像划分的评测；再决定LoRA范围。
不靠更换提示词并反复查看这四个窗口来宣称泛化提升，不进入自动控制。
