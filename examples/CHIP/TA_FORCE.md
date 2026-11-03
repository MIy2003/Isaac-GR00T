# TA-VLA → GR00T：三维 Force，保留 base policy 训练配置

独立于 Force MoT，只保留 `EXPERT_HIS_C_FUT`（历史单 token + 未来力联合预测）。参考 [TA-VLA](https://github.com/ZZongzheng0918/TA-VLA)，版本 `149d4be296530172eb09a5ca7ddd72413502924a`。

## 模型与数据

- 只使用传感器坐标系 `[Fx,Fy,Fz]`，单位 N；不读取或监督 Tx/Ty/Tz。
- 最近 10 个可用历史采样，按时间拼成 30 维；`Linear(30,2W) → SiLU → Linear(2W,W)` 编为一个 token。当前历史不是论文的固定 2 秒均匀采样。
- 动作与未来力共享 DiT、flow 时间并联合去噪；损失为动作损失 + 0.1 × 力损失，各自按有效元素平均。
- state 使用真实 replay LowState 经 FK 得到的 37 维，历史 `[-6,-3,0]`，去掉自身初始 IMU yaw；`ideal_state=False`。action、RGB、归一化和训练/验证划分保持 base policy 不变。
- 使用原 `ChipReplayDataset`、`ShardedMixtureDataset` 和 `Gr00tTrainer`，仅在原分片样本中追加力字段，不更换采样器。
- Force 来自原始 `source_raw/ft_data.csv` 和 `sync_frames.jsonl`，按 source timeline 配 replay；它不是 replay 物理执行时重新测得的力。历史严格因果，过期（500 ms）、缺失、未来越界及 padding 均标记无效。
- Force 的 mean/std 仅由训练 episode 计算并保存为模型 buffer；state/action 统计仍由原训练流程计算。

## 与 base 完全对齐的训练入口

直接读取各任务 base 作业保存的 `experiment_cfg/config.yaml`。只允许 `model.ta_force` 和 `training.output_dir` 发生变化，并保存差异报告。保留 base 的初始权重来源，不从任务已训练的 checkpoint-20000 继续训练。

```bash
PYTHONPATH=. .venv/bin/python -m examples.CHIP.train_ta_force \
  --base-config /path/to/base/checkpoint-20000/experiment_cfg/config.yaml \
  --output-dir /path/to/new_ta_force_run
```

可加 `--audit-only` 只检查配置。当前两个 base 的共同配置：batch 160、梯度累积 1、RTC 最大延迟 6、20,000 steps、lr 1e-4、cosine、warmup 0.05、weight decay 1e-5、max grad norm 1、每 1,000 steps 保存、保留 5 个 checkpoint、2 个数据 worker、BF16/TF32。视觉/语言骨干冻结，训练的 action head 部分与 base 一致，另外训练 Force 模块。

作业脚本：`finetune_ta_force_a100_1gpu.slurm`，使用 `TA_TASK=dough|pivot` 和 `TA_PROJECT_ROOT` 指定独立快照。实验名与 base 相同，放在新的输出根目录下。

## RTC

复用 base 的 `condition_training_prefix`：逐样本抽取 0–6 帧干净动作前缀，action token time 设为 1，并屏蔽对应动作损失。历史 Force token 和 state token 保持当前采样时间。

已提交动作不代表已知未来真实力：其对应的未来力输入设为零，力损失也屏蔽；推理固定动作前缀，并将对应力预测标记为无效。不会把未来实测力作为 RTC 条件泄漏给模型。

```python
from gr00t.policy.ta_force_policy import TAForcePolicy
policy = TAForcePolicy("NEW_EMBODIMENT", "/path/to/checkpoint", device="cuda:0")
observation["force"] = force_history      # float32 [B,10,3]，物理单位
observation["force_valid"] = force_valid  # bool [B,10]
actions, info = policy.get_action(observation, options={
    "rtc_mode": "trained",
    "rtc_prefix": committed_physical_actions,  # 与 base 相同的 action 字典 [B,d,37]
})
# info['force_pred']: 物理单位的未来力；info['force_pred_valid']: 排除已提交前缀。
```

调用方负责因果历史、坐标系/单位一致、episode 重置。未来力是预测观测，不是力控制指令。普通无前缀推理也支持。

## 验证与历史作业

```bash
.venv/bin/python -m pytest tests/gr00t/model/test_ta_force.py tests/gr00t/model/test_training_rtc.py -q
```

测试覆盖真实小型 DiT 的训练/采样、RTC 前缀保持与未来力无泄漏、参数恢复、三维力读取、历史因果性，以及新增 Force 前后原分片顺序/state/action/统计一致。

**作业 5824 / 5825 已取消，不作为有效对照。** 它们曾使用有效 batch 32、关闭 RTC，并从已训练任务权重初始化，另有 Trainer 默认参数差异。其日志和输出保留用于审计；旧记录在 `assets/ta_force_xyz/`，已被此次重训取代。

新快照目录：`/share/ml/yangmin/gr00t-chip/runs/ta_force_xyz_rtc_base_20261030/`。base 配置归档位于 [assets/ta_force_xyz_rtc_base](assets/ta_force_xyz_rtc_base/)。

正式重训作业：**5840（削面）、5841（翻盒）**。二者只读取 base 的配置，使用 base 原始初始化权重；不是从旧 Force 作业恢复。容量验证 job 5835 在 A800-SXM4-80GB 完成两步前反向和 checkpoint 保存，峰值 allocated 69.37 GiB / reserved 75.49 GiB。31 项本地测试通过。

| 任务 | Job | 新输出根目录（`/share/ml/yangmin/gr00t-chip/outputs/` 下） |
|---|---|---|
| 削面 | 5840 | dough_ta_force_xyz_rtc_base_5840/dough_shaving_replay_yaw_1gpu |
| 翻盒 | 5841 | pivot_ta_force_xyz_rtc_base_5841/pivot_box_replay_yaw_1gpu |

每个输出根目录的上一级保存 `base_config_comparison.json`；checkpoint 按原训练流程保存完整配置、processor 和统计。容量验证的 processor 配置及 state/action 统计已与削面 base 作业逐字段比较，完全一致。

```bash
ssh lambda 'squeue -j 5840,5841'
```

正式启动后两个任务的 processor 配置、state/action 统计均与各自 base 完全一致，最终模型配置仅新增 ta_force；详见 [effective_audit.json](assets/ta_force_xyz_rtc_base/effective_audit.json)。5840 已记录 loss 1.4957 / 1.4961，5841 已记录 loss 1.4911，梯度有限，均处于原定 warmup 阶段。
