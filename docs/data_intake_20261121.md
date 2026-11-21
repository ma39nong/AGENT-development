# 数据接入核查：2026-11-21

## 结论

可以从已有原三路subtask数据开始，不需要用户重新提供同一批，也不要求先生成局部图。
含ROI包仍保留原三路时直接读取原图，现阶段不消费离线未来辅助ROI。
本地候选根目录：`/home/user/datasets/tomato_inhand204_20261120/source`。
本轮没有读取视频像素或启动训练，也没有修改源包、V-JEPA代码及其划分。

## 已核对的材料

| 内容 | 结果 |
| --- | --- |
| 源包与录制 | 13包，204条，588734帧（元数据） |
| 已锁定划分 | train151、val21、historical_test12、blind_test20 |
| train/val粗阶段 | 172条侧文件SHA与既有lock一致；源ID/episode/帧数/FPS、连续覆盖检查通过 |
| train/val三路视频 | 516个可访问文件，172条均有三路 |
| 封存数据 | 32条保留；只做录制身份和文件存在检查，未读其标注内容、未解码或评分 |
| 首批标注队列 | 5条train，跨A/B与5个包；全部为unreviewed空模板 |

镜像目录本身只有291个原三路视频文件可访问，不能把元数据存在当成所有视频已经同步完成。
对train/val缺失视频，检查粗标注记录的原始导入目录，仅在允许的fallback root下定位；238个文件SHA与标注内预期一致后才引用。
保留文件原位置，不自动下载封存视频，不修改或补写源目录。
尚未全量校验所有已有视频的哈希、解码、PTS和跨相机同步，也未全量检查Parquet行数，因此不宣布完整数据验收通过。

根manifest仍声明training_ready=false/subtask_labels_generated=false。后续生成的粗阶段侧文件可单独核验，但不能据此把整个根目录自动认定为Qwen SFT就绪。
抽查train Parquet的列为state/action/时间/帧号等，task_index对应整段任务Prompt；实际五阶段标注在annotations/subtasks/coarse。
train/val目前可见标注目录为coarse，尚未核验单颗成功/失败/未知和决策真值。

机器可读[摘要](data_intake_20261121_summary.json)保留范围限制；完整inventory及标注模板在本地`runs/data-intake-204-v2/`。

## 用户需要提供什么

同一批数据无需重复上传。如果有更新批次，只需提供可访问的服务器/目录，保持完整LeRobot结构：

- 头部、左腕、右腕的原始完整视频。
- meta、Parquet状态和源时间，以及源录制ID。
- 原subtask标注侧文件和Prompt定义，不只提供三段视频或一个task_index。

不必先导出删除第四路的副本，原始数据不动。

目前主要缺的是单颗标注与验证：持果/摘离、盘内释放、随后空手、抓空/盘外掉落/未知，以及首次足够确认的时刻。
还需区分实际用户目标、示范做了调整、当时是否确实需要调整，不能从整串示范编造目标数量或反事实技能真值。
具体要求见[事件标注约定](../data/event_annotation_requirements.md)。

## 接下来

先对5条train完成完整尝试序列标注和核查，再抽取训练窗口；冻结独立评测，不要求立即重标全部204条。
补采只针对清点后确实缺失的失败、遮挡、不同目标提前结束和调整路径。Qwen标签与π0.5技能执行能力分别验收。

## 工具

```bash
.venv/bin/python scripts/audit_subtask_data.py \
  --root /absolute/path/to/source \
  --source-lock /absolute/path/to/frozen/dataset.lock.json \
  --fallback-root /absolute/path/to/allowed-original-imports \
  --review-limit 5 --output runs/new-data-intake
```

输出目录必须不存在，避免覆盖人工标注。省略fallback-root时仅报告原位置，不隐式到任意目录寻找数据。
该工具仅做数据清点和待标注队列；没有自动生成真实事件标签，不输出可直接训练的已审核数据。
