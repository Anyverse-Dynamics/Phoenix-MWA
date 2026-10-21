# 编码器预训练

[English](README.md) | **简体中文** | [仓库概览](../../README_zh.md)

视觉编码器由 DINOv3-L/16 和 8 个共享时空 Adapter block 组成。冻结的 VGGT-Ω 提供几何监督，冻结的 V-JEPA 2.1 提供时序特征目标。该模块负责编码器预训练，导出的学生特征可供 CS-VAE 或世界动作模型使用。

## 模型与目标

RGB 片段先中心裁成方形，再缩放到 256×256。DINO 每帧产生 16×16 patch 网格，每个 patch 为 1,024 维。Adapter 包含 16 个 attention head 和 16 个 register token，在帧内做空间注意力、跨帧做时序注意力，并通过相对帧位置保留采样步长。输出为 `[B,T,256,1024]`。

默认配置从 VGGT-Ω checkpoint 的 `aggregator.patch_embed.` 初始化 DINO。两个教师按优化器更新交替；同一次累积更新中的全部 microbatch 使用同一教师。

- **VGGT-Ω：** Adapter 的 1、3、5、7 号 block 接收中间特征监督，学生特征通过冻结的深度 head 解码。最终关系项在前 5% 训练进度内升至 0.1。
- **V-JEPA 2.1：** 通过 DINO mask-token 输入施加空间 tube mask，保留完整 Adapter patch 网格。预测器将遮蔽位置替换为可学习 mask token，并预测时序池化后的教师特征；预测与目标分别 layer-normalize 后计算 L1。上下文损失权重在前 20% 优化器更新内升至 0.5；前 20% V-JEPA 更新内，mask 覆盖率从 35–55% 过渡到 50–90%，交替使用大小空间块。
- **EMA：** 每次优化器更新后以 momentum 0.9999 更新部署学生。导出保留学生，训练教师、projector 和 predictor 分别保留在训练模块中。图像归一化等固定常量直接复制。

训练与部署共享 `wam.encoder.multi_teacher_distillation`，采用 dense mask curriculum 和交替教师目标。

## 安装与资源

使用仓库的 Python 3.10 环境及兼容的 PyTorch。依赖包含 Accelerate、OmegaConf、PyAV 和 PyArrow。可选 W&B 依赖通过 `pip install -e '.[logging]'` 安装。默认视频后端为 PyAV；Decord、OpenCV、TorchCodec 需按使用情况另行准备。

DINO 源码已随附。教师实现和预训练权重通过外部路径配置，使用以下源码版本：

| 教师／骨干 | 源码版本 |
| --- | --- |
| VGGT-Ω | `39a0cb8af88554f15ddcb5354cd52bde588fa014` |
| V-JEPA 2.1 | `204698b45b3712590f06245fbfba32d3be539812` |
| DINOv3 | `6876159a11b4df116f30f667f8c9888617df0751` |

设置资源环境变量，或在本地配置中替换 [dinov3_multiteacher.yaml](../../configs/encoder_pretrain/dinov3_multiteacher.yaml) 的环境插值：

```bash
export PHOENIX_VGGT_SOURCE="$PWD/resources/vggt-omega"
export PHOENIX_VJEPA_SOURCE="$PWD/resources/vjepa2"
export PHOENIX_VGGT_CHECKPOINT="$PWD/checkpoints/vggt_omega_1b_512.pt"
export PHOENIX_VJEPA_CHECKPOINT="$PWD/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt"
export PHOENIX_EGO_DATA="$PWD/data/ego_example"
export PHOENIX_ROBOT_DATA="$PWD/data/robot_example"
export PYTHONPATH="$PWD/src"
```

数据项可以设置单个 `path`、多个 `paths` 或显式 `subdatasets` 模式。要求原生 LeRobot v3 `meta/info.json`、episode Parquet 元数据及视频。`video_key` 指定主相机，`sampling_group` 使用 `ego_general` 或 `robot`；读取器发现所有 `dtype: video` 特征，辅助相机作为独立片段流采样。

默认长度为 4／8／12／16，步长为 1／2，每设备每个 microbatch 的帧预算为 32。温度采样目标为 ego/general:robot = 40:60、primary:auxiliary = 50:50。数据源需要提供对应的有效流；应按实际数据组成调整比例。

## 配置检查与训练

`preflight` 解析配置、检查源码版本和权重文件是否存在，并读取视频元数据：

```bash
python -m wam.encoder_pretrain preflight \
  --config configs/encoder_pretrain/dinov3_multiteacher.yaml
```

使用新的 run ID 启动训练：

```bash
accelerate launch --num_processes 8 --mixed_precision bf16 \
  -m wam.encoder_pretrain train \
  --config configs/encoder_pretrain/dinov3_multiteacher.yaml \
  run_id=encoder_run_01
```

默认训练 160,000 次更新。只有 `train` 子命令执行优化。W&B 默认关闭；启用时设置 `tracking.enabled=true`、`wandb_project` 和可选 `wandb_entity`。本地 JSONL 指标独立保存。

## 恢复与导出

输出位于 `run_root_dir/run_id`：

| 输出 | 内容 |
| --- | --- |
| `config.full.yaml` | 解析后的配置 |
| `metrics.jsonl` | 损失、学习率与采样组成 |
| `checkpoints/step_XXXXXXXX.pt` | 学生、EMA、训练目标、优化器、调度器、采样器和各 rank RNG |
| `ema_student_final.pt` | 部署学生权重、元数据与配置 |

非空运行目录需要显式提供 `trainer.resume_from=...`。恢复要求 world size、梯度累积、模型、数据和训练日程一致；日志与保存间隔可调整。教师从配置路径重新加载，因此应保留所选权重。目录缓存由本模块在输出位置生成；输入元数据变化时更新 `catalog_cache_version`。多机共享缓存需要文件系统支持跨节点锁。

完整恢复格式为 `phoenix.encoder_pretrain.resume.v1`，学生导出格式为 `phoenix.encoder.student.v1`。在 `OnlineDinov3DistilledEncoder` 中，`ema_student_final.pt` 使用 `checkpoint_state="student"`，完整训练 checkpoint 使用 `checkpoint_state="ema_student"`。导出包含现有加载器需要的 Adapter 配置。

## 模块映射

| 模块 | 职责 |
| --- | --- |
| `wam.encoder_pretrain.__main__` | preflight／train 命令 |
| `config.py`、`build.py` | 资源配置与模型构建 |
| `model.py` | 共享编码器上的训练 checkpoint 接口 |
| `data/` | LeRobot v3、元数据目录、多视角采样与视频解码 |
| `trainer.py`、`multiview_trainer.py` | 交替目标、梯度累积、EMA、恢复和指标 |
| `wam.encoder.multi_teacher_distillation` | 共享学生、Adapter 和教师损失 |
| `wam.encoder.external_sources` | 固定版本教师源码加载 |
| `wam.third_party.dino_runtime.dinov3` | DINO 源码 |
