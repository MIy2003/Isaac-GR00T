# GR00T N1.7：CHIP 数据微调

## Robofarm wiping-car 实机回放数据（2026-10-05）

本次新训练使用 `chip_replay`，数据源为 Robofarm 的
`/home/yixuan/Project/HIL_Dagger/data/wiping_car_new_replay_merged_v001`。
原有 `chip_native` 配置仍按下文的旧相对动作协议读取旧数据。

- `state_action_30hz.npz/state` 原始为 65 维。输入策略前，由实测 29 关节位置和
  IMU 四元数通过 FK 转为 **37 维**：下肢 12 关节、anchor wxyz、三点位置、三点 wxyz。
  不输入关节速度、IMU 角速度。state 历史仍为 `[t-6,t-3,t]`，anchor 去掉 episode 初始 yaw。
- 三点依次是左腕、右腕、躯干，均在 pelvis/anchor 坐标系中；关节按名称映射。
  FK 树来自该数据使用的 MimicLite 977/24500 模型，打包在
  `gr00t/data/chip_replay_kinematics.json`，含原始 scene SHA256。
  各 episode 的 `chip_manifest.json/point_offsets_m` 决定点偏移；本数据右腕偏移为
  `[0.0719,-0.003,0]`，不能套用旧右腕偏移。
- action **独立读取** `state_action_30hz.npz/action` 的 37 维绝对源参考，输出键为
  `reference_absolute`，不作相对差分或 yaw 变换，不使用 `applied_reference_action`。
  数据 state 为动作前状态，因此动作窗口为 `[t,...,t+23]`，不是旧格式的 `[t+1,...,t+24]`。
- RGB 仍为 `[t-6,t]`；按 NPZ `image_file` 读取。历史不足重复首帧，未来不足或窗口
  内存在无效 state/action/timing 时不采样。seed=42，按 episode 留出 5%，归一化仅用训练集。
- Lambda 全量检查通过：98 episodes / 34,110 帧；训练 93 episodes / 10,132 样本，
  留出 5 episodes / 517 样本。留出集未参与梯度或统计量，当前训练器不计算在线验证指标。
- 全量 72,312 文件已校验 SHA256；新数据保留所有原始模态与 provenance。

```bash
python -m examples.CHIP.inspect_replay_dataset --dataset-path /path/to/wiping_car_new_replay_merged_v001
python -m unittest discover -s tests/gr00t/data -p test_chip_replay.py -v
# Lambda: from a writable run/log directory
sbatch /path/to/Isaac-GR00T/examples/CHIP/finetune_replay_a100_1gpu.slurm
```

Lambda 代码在 `/share/ml/yangmin/gr00t-chip/runs/wiping_car_replay_20261005/Isaac-GR00T`，
数据在 `/share/ml/yangmin/gr00t-chip/data/wiping_car_new_replay_merged_v001`。
单卡默认 batch=160、20,000 steps、每 1,000 steps 保存；冻结视觉/语言骨干，训练
projector 和 diffusion head。`GLOBAL_BATCH_SIZE`、`MAX_STEPS`、`SAVE_STEPS` 可覆盖。
任务文本为 `Wipe the car.`。新模型输出是绝对动作，推理端必须使用保存的 replay
modality 配置及同一实测 FK state，不能通过旧 `predict_chip_relative` 解码。

本次正式训练 Job ID：`1391`，节点 `dgx-hyperplane12`，1 × A800 80GB。
已观察到 20 步以上实际更新，loss 为 `1.2957 → 1.2926`，约 2.9 秒/步。
日志位于 `runs/wiping_car_replay_20261005/chip-replay-1391.{out,err}`；输出位于
`/share/ml/yangmin/gr00t-chip/outputs/wiping_car_replay_1gpu_1391/wiping_car_replay_1gpu`。
回归检查为 9 passed / 1 skipped（原 DP 可选对照未配置）。

## 旧 CHIP-native 协议


直接读取现有 `g1_chip_episode_folder_v001` 数据，无需转换 LeRobot。
数据格式和输入/动作语义沿用 `HIL_Dagger/config/live_diffusion_chip_native_relative_live.yaml`，
动作预测长度扩展为 24 步（30 Hz 下 0.8 秒）；原部署 YAML 未修改。已核对
SSH `5090` 上的 `HIL_GentleHumanoid/third_party/diffusion_policy` 训练实现。

## 数据与输入输出

| 项目 | 定义 |
| --- | --- |
| 数据时钟 | 30 Hz；每 3 帧取一个训练样本，即 10 Hz |
| RGB | `camera0_rgb`，`[t-6,t]`，两帧 RGB，原图 `224×400` |
| 状态 | `[t-6,t-3,t]`，三帧绝对 native reference，去除 episode 首帧 yaw |
| 动作 | `[t+1,...,t+24]`，24 步相对 native reference，每步 37 维 |
| 37 维顺序 | 下肢关节位置 12 + anchor wxyz 4 + 三点位置 9 + 三点 wxyz 12 |
| 三点顺序 | `left_wrist, right_wrist, torso` |
| 四元数相对量 | `inverse(q_current) * q_future`，单位化并取 `w>=0` 半球 |
| 位置和关节相对量 | `future-current`；三点位置不额外旋转 |
| 边界 | 历史不足时重复 episode 首帧；未来不足时丢弃样本 |

只去除首帧的全局 yaw，保留 roll/pitch 和后续 yaw 变化，三点在 anchor
坐标系内的数据保持不变。动作参考是本次观测的最后一帧，不是每一步前一帧。

`ft_sensor` 是当前部署里的 CR-DAgger 残差/采集输入，原始 CHIP DP 主策略没有
把它作为观测；这里也保留这一定义。`lower_joint_vel` 不进入 37 维输出。

GR00T 额外需要任务文本。本适配器对整个数据集使用固定文本，默认
`Perform the demonstrated task.`，可设为具体任务，推理时须保持一致。
因此传感器和动作语义一致，模型结构及内部预处理不等同于原 DP：

- 图像使用 GR00T 的视觉处理器；启动脚本保留全图，之后仍经过 VLM 的尺寸调整。
- 状态/动作采用 GR00T 的逐维归一化，时间步共享统计量；不复用 DP 的逐时间步 normalizer。
- 保留预训练模型的内部动作维度和 horizon，超出 37 维/24 步的训练目标被 mask，
  推理处理器仅返回 `native_relative: [B,24,37]`。
- 预训练状态编码器从一帧扩展到三帧：当前帧权重原样保留，前两帧新增权重初始化为零。
  保存的模型配置包含 `state_history_length=3`，加载微调 checkpoint 无需再次扩展。

模态配置中的 `ActionRepresentation.ABSOLUTE` 表示 GR00T 不再自动变换已编码的
`native_relative` 数值，**不代表物理动作变成绝对动作**。这样也避免推理处理器自动
把相对动作解码成绝对动作。不要自行改为 `RELATIVE`。

## 训练

在包含本次修改的 Isaac-GR00T 仓库中，先按仓库说明准备 Python 3.12 / GPU 环境，
并准备 `nvidia/GR00T-N1.7-3B` 权重或本地权重目录。

`5090` 上已找到的数据路径：

```bash
export CHIP_DATASET_ROOT=/media/raid/workspace/yangmin/data/drag_collection/flip_noft/chip
```

该路径属于远端机器；若在本地训练，应设置成本地可访问的数据路径。
远端仓库为 `/media/raid/workspace/yangmin/code/Isaac-GR00T`，使用仓库内独立 `.venv`。

5090 四卡启动（先确认所选卡空闲）：

```bash
cd /media/raid/workspace/yangmin/code/Isaac-GR00T
source examples/CHIP/activate_5090.sh
export CHIP_DATASET_ROOT=/media/raid/workspace/yangmin/data/drag_collection/flip_noft/chip
export CUDA_VISIBLE_DEVICES=4,5,6,7
export NUM_GPUS=4
bash examples/CHIP/finetune.sh
```

脚本自动通过 `torchrun` 启动四个进程，并设置 `--num-gpus 4`。
默认每卡 batch 为 1，梯度累积 8 次，每次参数更新使用 32 个样本。
也可用 `--num-gpus 4` 替代 `NUM_GPUS=4`，两者均会控制实际进程数。
激活脚本将下载、编译和测试缓存放到 RAID 盘，保留现有 Hugging Face 登录。
需先取得 `nvidia/Cosmos-Reason2-2B` 访问权限并在远端执行 `hf auth login`。

短训练验证命令（包含第 10 步 checkpoint 保存）：

```bash
OUTPUT_DIR=./outputs/chip_debug_4gpu bash examples/CHIP/finetune.sh \
  --max-steps 10 --save-steps 10 --save-total-limit 1
```

2026-09-06 在 5090 上的验证记录：

- 独立 Python 3.12 环境按 `uv.lock` 安装；PyTorch `2.9.0+cu128`、
  Transformers `4.57.3`、DeepSpeed `0.17.6`、Flash Attention `2.8.3`。
- 实际 CHIP 数据：237 个训练 episode、18,608 个样本；24 步动作归一化往返通过。
- 4–7 号 RTX 5090 的 NCCL all-reduce、BF16 Flash Attention 前向/反向全部通过。
- CHIP、RTC、状态/动作处理器和分布式初始化回归：36 passed、1 skipped
  （跳过的是需要额外 DP 源码环境的可选一致性测试）。
- 使用真实 N1.7 权重、真实数据、RTC 延迟上限 6 和 ZeRO-2 完成 10 次参数更新，
  每卡 batch 1、梯度累积 8；loss `1.1328`、梯度范数 `0.0924084`，
  运行中采样到每卡显存约 18–19 GiB，进程退出码为 0。
- `outputs/chip_debug_4gpu/checkpoint-10` 已保存模型、处理器、优化器及四个 rank 的 RNG 状态。
  重载后普通/RTC 推理均返回 `[24,37]`，RTC 前 6 步与输入前缀逐元素完全相同。
- 完整日志和汇总位于远端 `outputs/chip_debug_4gpu_checks/`。

这是训练链路短跑验证，不用于判断任务成功率或长期训练收敛情况。

先执行 CPU 数据检查：

```bash
python -m examples.CHIP.inspect_dataset --dataset-path "$CHIP_DATASET_ROOT"
```

再启动微调：

```bash
export CHIP_TASK_DESCRIPTION='Perform the demonstrated task.'
export OUTPUT_DIR=./outputs/chip_native_relative
# 可选：export BASE_MODEL_PATH=/path/to/GR00T-N1.7-3B
bash examples/CHIP/finetune.sh
```

脚本默认单 GPU、microbatch 4、梯度累积 8、10000 步，并开启下述 training-time RTC。
实际显存用量需在训练机验证；
可在脚本后追加 `--global-batch-size 1 --gradient-accumulation-steps 32` 等已有参数。
脚本本身不安装依赖。原 LeRobot 微调入口保持可用；新增开关为
`--dataset-format chip_native`，它会自动注册 `NEW_EMBODIMENT` 的 CHIP 配置。

默认以 seed 42 按 episode 留出 5%，与原 DP 的拆分算法一致；留出集不参与统计量。
当前 GR00T sharded trainer 不执行定期 validation，留出数据可用
`ChipNativeDataset(..., split="val")` 读取。不要使用要求 LeRobot 数据格式的
`open_loop_eval.py` 直接读取 CHIP 文件夹。

## A100 单机四卡

`finetune_a100_4gpu.sh` 面向 A100 80GB，复用相同的
CHIP 数据处理和 `torchrun` 入口。默认每卡 batch 128、梯度累积 1、有效 batch 512，
学习率 `1e-4`，BF16 + DeepSpeed ZeRO-2；冻结视觉/语言骨干，训练动作头及投影层。
CHIP action chunk 为 24，RTC 最大延迟默认 6。已在 lambda A100 80GB 实测单卡训练；
四卡通信尚待验证。40GB 卡需要显式降低 GLOBAL_BATCH_SIZE。

冻结策略与官方 `gr00t/configs/finetune_config.py` 的默认值一致：
`tune_llm=false`、`tune_visual=false`、`tune_projector=true`、
`tune_diffusion_model=true`，模型默认 `tune_vlln=true`。
这包括 action head 内的状态/动作编码解码、位置嵌入、DiT、VLLN 和视觉语言自注意力。
官方 G1 仿真工作流则另行推荐打开 `tune_visual`；若采用该策略，需要重新测量 batch 显存。
参考：[官方默认配置](https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/configs/finetune_config.py)、
[G1 仿真工作流](https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/simulation-workflow/groot-fine-tuning-sim.html)。

在 A100 机器按本仓库 `uv.lock` 安装环境，准备好 Cosmos 权限或缓存，再执行：

```bash
cd /path/to/Isaac-GR00T
source .venv/bin/activate
export CHIP_DATASET_ROOT=/path/to/chip
# 可选：export BASE_MODEL_PATH=/path/to/GR00T-N1.7-3B
CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/CHIP/finetune_a100_4gpu.sh
```

路径使用 A100 机器上的实际位置，无需 source `activate_5090.sh`。
若调度器已经设置 `CUDA_VISIBLE_DEVICES`，保留它并直接执行脚本。
先做 10 步短跑及保存检查：

```bash
OUTPUT_DIR=./outputs/chip_a100_smoke bash examples/CHIP/finetune_a100_4gpu.sh \
  --max-steps 10 --save-steps 10 --save-total-limit 1
```

四卡 A100 80GB 默认每卡 batch 128、累积 1，有效 batch 512。
若需复现旧的有效 batch 32，可显式使用较小的配置：

```bash
GLOBAL_BATCH_SIZE=8 GRADIENT_ACCUMULATION_STEPS=4 \
  bash examples/CHIP/finetune_a100_4gpu.sh
```

增大 microbatch 可能提高吞吐，需要实测；不应把更多显存直接理解为更高任务成功率。
5090/A100 对比应保持数据划分、有效 batch、更新步数、学习率和冻结设置一致。
改变 microbatch 或硬件后不保证数值逐位一致。
当前只有 10 步链路验证，尚无验证集或实机成功率对比；默认 trainer 不进行周期验证。

## Training-time RTC

实现依据 [Training-Time Action Conditioning for Efficient Real-Time Chunking](https://arxiv.org/abs/2512.05964)：
对每个样本随机采样延迟 `d`，前 `d` 步使用干净的真实动作，后面的动作正常加噪；
前缀的 flow timestep 设为 1，后缀沿用原先采样的 timestep，损失只计算后缀。
动作编码器和 DiT/AlternateVLDiT 的 adaLN、输出调制均支持逐 token 时间。
参数数量和权重形状不因此改变，仍可从原 N1.7 checkpoint 开始微调。

CHIP 脚本默认 `--rtc-training-max-delay 6`，均匀采样 **0–6 步，包含两端**。
这是 30 Hz 动作序列上的 0–200 ms；不是 10 Hz 重规划次数或 50 Hz 控制器 tick。
零前缀样本用于学习首次启动。每个样本最多条件化有效 horizon 减一行，
补齐的时间步不计入可采样延迟，补齐维度仍不参与损失。
冻结视觉/语言骨干、训练动作头的设置保持原配置。

```bash
# 默认开启；应根据实测推理延迟调整预算
RTC_MAX_DELAY=6 bash examples/CHIP/finetune.sh
# 关闭 RTC，作为对照实验
RTC_MAX_DELAY=0 bash examples/CHIP/finetune.sh
```

通用微调入口的默认值仍是 `0`。该字段保存到模型配置，训练加载 checkpoint 时
使用命令行值覆盖；继续 RTC 微调时也要传正数。只修改旧 checkpoint 的配置数字
不会让旧权重学会前缀条件。现有底层 overlap/ramp 推理路径仍保留，新增模式使用
训练好的模型进行硬前缀条件生成，不使用梯度 guidance 或软 ramp。

## 推理接口

```python
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.chip_native_policy import predict_chip_relative

policy = Gr00tPolicy(
    embodiment_tag="NEW_EMBODIMENT",
    model_path="./outputs/chip_native_relative/checkpoint-10000",
    device="cuda:0",
)
policy.model.action_head.num_inference_timesteps = 8  # 对应 live YAML 的推理步数

# images: uint8 [2,224,400,3]，时间顺序 t-200ms、t
# native_history: float32 [3,37]，已按 episode/A-time yaw canonicalization 处理
relative_action = predict_chip_relative(policy, images, native_history)
# relative_action: float32 [24,37]；任务文本非默认时传入同样的 task_description
```

若已有 DP 张量 `camera0_rgb` 为 `[2,3,224,400]`、float `[0,1]`，需先转 HWC uint8；
四组 native 状态按上表顺序拼接成 `[3,37]`。不要把历史转换成对当前帧的 delta，
也不要在每个滑动窗口重新去除 yaw。几何编码/解码工具位于 `gr00t/data/chip_native.py`。

当前 live YAML 的有效值是 `relative_decode_anchor: actual_observation`。
HIL 应保留本次请求的观测，返回后用该观测解码 `q_abs=q_current*q_relative`，
位置和关节则相加，再交给已有 CHIP/CR-DAgger 流程。YAML 顶部关于 planned frontier
的注释与当前有效值不同。

GR00T checkpoint 是模型目录，不能直接替换 HIL 当前使用的 DP `.ckpt`。
这里提供了 Isaac-GR00T 侧的输入输出桥接函数；HIL 的策略加载入口和实时调度尚未切换。
10 Hz 重规划、每次执行 3 行和 50 Hz CHIP 控制仍由部署端管理，GR00T 的实际推理延迟
必须在目标 GPU 上测量。

### RTC 推理前缀

普通请求仍使用上面的 `predict_chip_relative`，首次启动无需前缀。
连续请求必须把已经承诺执行的动作转换到**本次请求参考帧**：

```python
from gr00t.policy.chip_native_policy import rebase_chip_prefix

# 示例：两次请求的观测时间相隔 3 个 30 Hz 帧，预计推理延迟为 3 帧。
# previous_reference/current_reference 都是对应请求状态历史的最后一行 [37]，
# 并且处于同一个 episode/A-time yaw canonical 坐标系。
prefix = rebase_chip_prefix(
    previous_chunk, previous_reference, native_history[-1],
    elapsed_steps=3, delay_steps=3,
)
chunk = predict_chip_relative(policy, images, native_history, rtc_prefix=prefix)
# chunk[:3] 是前缀；按时间戳切换到尚未执行的后缀，不重复执行前缀。
```

`rebase_chip_prefix` 先按已过去的帧数移动旧 chunk，再用旧参考还原绝对动作，最后
相对于新参考重新编码（包括四元数组合），不能直接截取旧 relative 数值。
如果安全限幅、CR-DAgger 或其它控制层改变了已承诺执行的轨迹，应以实际提交给队列的
轨迹为准生成前缀，不能把修正前的模型输出当作已承诺动作。

通用 `Gr00tPolicy`/客户端接口也支持：

```python
actions, info = policy.get_action(observation, options={
    "rtc_mode": "trained",
    "rtc_prefix": {"native_relative": prefix[None]},  # 物理单位 [B,d,37]
})
```

同一 batch 的前缀长度相同；`d` 必须不超过 checkpoint 的训练延迟配置且小于有效
horizon。已承诺动作归一化时不裁剪到训练分位数，生成过程逐步固定前缀，返回值保留
传入的物理前缀。提供的前缀必须覆盖新请求 `t+1...t+d`；机器人继续执行旧队列时才
适用。需要 `elapsed_steps + d <= 24`，实际到达时间超过预算时须由调度器处理。

本次只接入模型训练和推理接口，没有实现 HIL 的异步动作队列，也没有修改当前
`sequential_open_loop_sync` 调度。开启训练参数本身不会把同步部署变成异步 RTC。

## 回归检查

```bash
python -m pytest tests/gr00t/data/test_chip_native.py -q
python -m pytest tests/gr00t/model/test_training_rtc.py -q
# 可选：与原训练类逐样本对照；此项需要原 DP 环境依赖
CHIP_DP_DATASET_SOURCE=/path/to/diffusion_policy/dataset/chip_native_dataset.py \
  python -m pytest tests/gr00t/data/test_chip_native.py -q
```

覆盖历史与未来帧对齐、episode 边界、yaw 和非交换四元数乘法、相对动作还原、
统计量拆分与归一化还原、推理接口形状，以及状态编码器扩展后的前向/反向行为。
RTC 测试额外覆盖逐 token 时间与原标量时间的等价性、实际动作头的前向/反向、
每次去噪的前缀固定、padding/loss mask、旧权重加载、零前缀生成、相对动作重定参考帧，
以及策略接口的归一化/前缀传递。CPU 测试不加载 VLM 骨干或预训练模型权重。

## Hugging Face 私有资源包与一键部署

目标仓库为 `mimiclite-chip/GR00T-CHIP`（model repo，私有），统一保存：
全部 249 个 CHIP episode、完整 GR00T N1.7 / Cosmos-Reason2-2B 官方快照、
CHIP/RTC 修改后的训练源码与 `uv.lock`、原始模型许可证和声明。
GR00T 权重按其许可证仅用于非商业研究/评估；Cosmos 使用独立的 NVIDIA Open Model License。
原始数据没有因备份而获得公开使用许可。上传过程不包含 token、虚拟环境或调试 checkpoint。

在有 CUDA 12.8 兼容驱动、CUDA toolkit（nvcc）和 FFmpeg 4–7 运行库的 Linux x86_64
服务器上，先用 `hf auth login` 登录能读取该私有仓库的账号。预留约 100 GB 空间。
以下一行命令会安装独立 Python 3.12 环境、下载/校验/恢复数据和权重，再启动四卡训练：

```bash
hf download mimiclite-chip/GR00T-CHIP deploy.sh --local-dir ./gr00t-bootstrap && bash ./gr00t-bootstrap/deploy.sh
```

默认使用当前目录下 `gr00t-chip-deployment`，可通过 `DEPLOY_ROOT` 指定大容量磁盘。
`CUDA_VISIBLE_DEVICES` 沿用外部设置，未设置时为 `0,1,2,3`；`NUM_GPUS` 默认 4。
适用 5090 或 A100，数据格式、24 步 action chunk、RTC 最大延迟 6、有效 batch 32 不变。

```bash
# 只准备资源与环境，并执行真实数据检查
DEPLOY_ROOT=/large-disk/gr00t-chip DEPLOY_MODE=prepare bash ./gr00t-bootstrap/deploy.sh
# 10 步四卡短训练及 checkpoint 保存
DEPLOY_ROOT=/large-disk/gr00t-chip DEPLOY_MODE=smoke CUDA_VISIBLE_DEVICES=4,5,6,7 \
  bash ./gr00t-bootstrap/deploy.sh
# 训练参数可以继续追加；默认正式训练 10000 步
DEPLOY_ROOT=/large-disk/gr00t-chip bash ./gr00t-bootstrap/deploy.sh --max-steps 10000
```

部署会先解析一个确定的 Hub commit，逐文件核对 `bundle.json` 中的 SHA-256，
恢复 episode 原目录，并将镜像权重填入隔离的、按 bundle 版本区分的 HF 缓存。
下载后启用 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`；模型加载不再访问 NVIDIA 仓库。
许可证与原始模型卡保留在 `models/`。这不是 Hugging Face Space，也不包含实机控制器部署。

打包和上传工具（应在有原始数据与已授权模型访问权限的机器上执行）：

```bash
python -m examples.CHIP.prepare_hf_bundle \
  --dataset-path "$CHIP_DATASET_ROOT" \
  --staging-dir /large-disk/gr00t-chip-upload \
  --repo-id mimiclite-chip/GR00T-CHIP --upload
```

不加 `--upload` 只生成资源包；上传工具会拒绝将原始数据写入公开仓库。
模型版本固定为已通过四卡短训练的上游 commit，具体版本及校验值见 `bundle.json`。

### lambda：先准备共享环境，再提交 Slurm

当前两组对照训练使用独立脚本，按以下顺序提交（提交顺序不保证调度启动顺序）：

| 脚本 | GPU 数 | 每卡 batch | 有效 batch（累积 1） | 可训练模块 |
|---|---:|---:|---:|---|
| `finetune_a100_1gpu_head.slurm` | 1 | 160 | 160 | action head / projector / VLLN |
| `finetune_a100_4gpu_visual.slurm` | 4 | 24 | 96 | 视觉骨干 + action head / projector / VLLN |

两组都冻结 LLM，采用 research/lv0b、48 小时、20000 步、每 1000 步保存。
两份脚本均允许 DGX,HGX，作业名及实验名分别为 train1、train2。输出目录分别为
`outputs/a100_head_1gpu_<jobid>/train1` 和 `outputs/a100_visual_4gpu_<jobid>/train2`。
它们使用节点本地临时目录，避免 NFS 多进程临时目录清理冲突。
验证基于单卡 A100 80GB，四卡 NCCL 仍需实际分配节点后检查。

```bash
module add slurm
sbatch /home/yangmin/gr00t-bootstrap/finetune_a100_1gpu_head.slurm
sbatch /home/yangmin/gr00t-bootstrap/finetune_a100_4gpu_visual.slurm
```

`finetune_a100_4gpu.slurm` 使用 DGX 分区、research 账号、lv0b QoS，申请
单节点四卡 A100 80GB、16 CPU、128GB 内存、两天时限。默认有效 batch 512，
每卡 128、梯度累积 1，训练 10000 步。作业内不下载或安装依赖。

lambda 的个人 `/home` 硬配额为 10GB，应将资源、环境和训练输出放在
`/share/ml/yangmin/gr00t-chip`，不能只根据 `df` 的文件系统空闲量选目录。
在登录节点先执行部署入口的 `DEPLOY_MODE=prepare`，成功后验证：

lambda 的 Ubuntu 20.04 不能加载官方 FlashAttention wheel 所需的 GLIBC_2.32。
首次 prepare 后，先运行 `install_flash_attn_lambda.sh`，安装社区项目
`mjun0812/flash-attention-prebuild-wheels` v0.9.0 提供的同版本 2.8.3
manylinux wheel（Python 3.12、Torch 2.9、CUDA 12.8）。脚本固定下载地址并验证
SHA256，无需更换 Python/Torch。`build_flash_attn_lambda.sh` 保留作为源码编译备选。
安装后再执行下面的验证。不要在正式作业内重新 `uv sync`，否则会
重新安装不兼容的官方 wheel。

预检和 Slurm 脚本设置 `JE_ARROW_MALLOC_CONF=background_thread:false`：
lambda 上组合导入 DeepSpeed、TorchCodec、PyArrow 时，Arrow 的 jemalloc 后台线程
会发生段错误；此设置关闭该后台线程。手动启动训练时也需要导出这个变量。

```bash
module add cuda12.8/12.8
export DEPLOY_ROOT=/share/ml/yangmin/gr00t-chip
source "$DEPLOY_ROOT/deployment.env"
source "$CHIP_PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="$CHIP_PROJECT_ROOT" NO_ALBUMENTATIONS_UPDATE=1
python /home/yangmin/gr00t-bootstrap/validate_prepared_env.py
```

验证检查实际样本、离线图像/tokenizer 处理、权重索引、编译库导入和 sm_80
CUDA 编译，全部通过后才生成 `environment-verified.json`。这不包含 GPU 运算或
NCCL 通信实测；这些需要分配到计算节点后验证。提交方式：

```bash
module add slurm
sbatch /home/yangmin/gr00t-bootstrap/finetune_a100_4gpu.slurm
```

## Replay episode-yaw-canonical 重训（2026-10-05）

旧 `chip_native` 先对整段 reference 去掉初始 yaw，再将未来 action 编码为
相对当前 reference 的增量。DP 的 `ChipNativeDataset._prepare_fields` 同样会对
state 与 action 执行 episode yaw canonicalization。

实测 replay 的 state 与 source action 来自不同世界朝向，因此新格式
`chip_replay_yaw_canonical` 分别以各自首帧为参考：

```text
q_state_canonical(t) = yaw(q_measured(0))^-1 * q_measured(t)
q_action_canonical(t) = yaw(q_reference(0))^-1 * q_reference(t)
```

实现复用 `canonicalize_anchor_by_episode_yaw`，按四元数左乘移除 yaw，保留
roll/pitch 和片段内的 yaw 变化。不是每帧归零，也不是相邻帧差分。
只改变 action 的 anchor 四元数，三点在 anchor 局部系的位置/姿态和下半身关节
保持原值。实测 state 仍由 FK 独立构建；历史 `[-6,-3,0]`，目标 `t..t+23`。

Action key 为 `reference_yaw_canonical`，其含义是初始朝向坐标系下的绝对姿态，
不加当前 state。这个独立 key 会让旧 absolute replay 推理入口拒绝加载，防止
静默混用参考系。部署新模型时必须将 canonical 零朝向映射到启动时实际 heading，
不可复用带源 yaw 的旧 bootstrap 偏置；训练本身不修改正在运行的真机配置。

检查及训练：

```bash
python -m examples.CHIP.inspect_replay_dataset --dataset-path "$CHIP_DATASET_ROOT" --yaw-canonical
sbatch examples/CHIP/finetune_replay_yaw_a100_1gpu.slurm
```

Lambda job **1458**：单卡，batch 160，20,000 steps，每 1,000 steps 保存；与
1391 相同的基础模型初始化，不恢复旧优化器或旧 replay checkpoint。
代码快照：`/share/ml/yangmin/gr00t-chip/runs/wiping_car_replay_yaw_20261005/Isaac-GR00T`。
输出：`/share/ml/yangmin/gr00t-chip/outputs/wiping_car_replay_yaw_1gpu_1458/wiping_car_replay_yaw_1gpu`。

5 项 replay 单测通过；全量检查包含 93 个训练/5 个验证片段、10,132/517 个
采样窗口、图像读取和 action 标准化往返。训练 action yaw 范围为
`[-4.3322, 9.4618]°`，P1/P99 为 `[-2.0838, 8.3427]°`。
state 的片段内异常朝向变化未被裁剪或过滤。上述范围为数据统计，不代表模型效果。

启动验证：1458 在 `dgx-hyperplane18` 的一张 A800 80GB 上完成至少 30 steps，
前 3 次日志 loss 为 1.2797、1.2749、1.2601，约 2.9 秒/step。
确认新任务正常后停止旧任务 1391，保留其 checkpoint-16000 等现有产物。

## Sim measured-state 对照实验（2026-10-06，job 1485）

真机 measured-state 对照任务 1458 已正常完成 20,000 steps。新实验只替换
数据中的 measured state，沿用原 RGB、source action、image→50Hz 行映射、98 个
片段及其顺序。仍使用 `chip_replay_yaw_canonical`：state/action 分别去各自
episode 初始 yaw；不是用 reference 假装 measured state。

Robofarm 生成器：
`/home/yixuan/Project/My_GentleHumanoid_Deploy_low_level/tools/generate_gr00t_sim_replay_states.py`。
对应源码为本目录 `generate_sim_replay_states.py`。生成器复用该工程的
`BatchPolicy`、`ProbePolicy` 和 `replay_mimic_6d_aligned_sim.simulate`，不启动 DDS。

- 使用真实录制 run `chip862_batch_20261004_221415_238877/setup/policy.yaml`。
- CHIP 862/11500；右腕平移 `[0.005,0.005,0.005] m/N`，旋转为 0，actor 缩放值为 0.25。
- 与真机相同的参考轨迹、heading 对齐、历史输入、PD gains、全身 8 rad/s 限速；
  50 Hz 逐行执行原 `chip_motion_50hz.npz`，不插值 30 Hz action。
- 记录执行每行 action **之前**的 q29/dq29/IMU4/gyro3，再按原有索引配对到 RGB。
- 仿真沿用原工程的 200 Hz MuJoCo、5 ms servo delay、alpha 0.95 filter、
  mild-wrist 初始化、3 秒初始保持和 4 秒 lead-in。
- 每片段独立 cold start，没有车辆接触或外力；这与真机连续 batch 的历史条件不同。
  RGB/FT 仍来自原始采集，不是仿真渲染/测量。FT 不输入当前 GR00T。

生成结果：98/98 完整，34,110 帧；全部 action 与原数据逐元素一致。对每条仿真
轨迹验证行号 `0..N-1`、pre-action state 和模拟器保存的 q、actor command、最终
target 一致。RGB 在各主机上通过硬链接复用原文件；不通过跨目录图片符号链接
绕过 loader 的路径检查。旧真机时间戳、扭矩和 next-state 别名未写入新 NPZ。

全 98 片段数据对比：sim canonical state yaw 为 `[-5.3508,5.1385]°`，真机为
`[-26.1223,5.4609]°`；关节位置总体 RMSE 为 0.03752 rad。仿真全过程最低 pelvis
高度 0.7615 m，最大倾斜 11.19°。这些是数据统计，不是模型性能结论。

数据集（文件 schema 兼容旧 replay，manifest 中 `state_source_kind=mujoco`）：

```text
Robofarm: /home/yixuan/Project/HIL_Dagger/data/wiping_car_sim_replay_862_c005_v001
Lambda:   /share/ml/yangmin/gr00t-chip/data/wiping_car_sim_replay_862_c005_v001
```

`generation_protocol.json` 保存代码/actor/config 校验信息，`comparison_audit.json`
保存对比统计，`simulation_summary.json` 保存逐片段结果。`SHA256SUMS` 的 197 个
训练相关文件在两端均验证通过。完整仿真 NPZ 保存在 Robofarm，新训练只传送所需数据。
全量训练入口检查通过：93/5 个片段、10,132/517 个窗口，所有 state FK、action
yaw 变换、图像读取和标准化往返一致。

Lambda job **1485** 使用 `finetune_sim_replay_yaw_a100_1gpu.slurm`，单卡、batch 160、
20,000 steps、每 1,000 steps 保存；与 1458 相同的基础模型初始化与训练参数，
不从 1458 权重续训。代码与输出分别为：

```text
/share/ml/yangmin/gr00t-chip/runs/wiping_car_sim_replay_yaw_20261006/Isaac-GR00T
/share/ml/yangmin/gr00t-chip/outputs/wiping_car_sim_replay_yaw_1gpu_1485/wiping_car_sim_replay_yaw_1gpu
```

判断优劣仍需在相同 held-out 数据和部署回放上比较；仅训练 loss 不能证明 sim
state 能改善真机表现。此次未执行真机运动，也未切换已部署模型。

启动检查：1485 已进入实际优化，首个 10-step 日志 loss 为 1.2798，梯度范数
0.38279。保存的 action normalization statistics 与 1458 **逐项完全一致**，
state statistics 则按 sim measured state 重新计算。

## Robofarm 两个 canonical checkpoint 的异步部署（2026-10-08）

1458（real replay state）和 1485（sim replay state）的 `checkpoint-20000` 已复制到
`/home/yixuan/Project/HIL_Dagger/models/gr00t_chip/checkpoints/` 下的
`real_replay_yaw_1458/checkpoint-20000` 与 `sim_replay_yaw_1485/checkpoint-20000`。
三个权重分片、索引、模型/processor 配置和统计均通过与 Lambda 源文件的 SHA256 校验。

在 Robofarm 的 HIL_Dagger 根目录运行：

```bash
# --model 选择训练 state 的来源，不是选择运行环境
bash run_gr00t_yaw_sim.sh --model real
bash run_gr00t_yaw_sim.sh --model sim

# 不初始化机器人 DDS 的配置/模型检查
bash run_gr00t_yaw_real.sh --model real --check
bash run_gr00t_yaw_real.sh --model sim --check

# 真机；ROBOT_INTERFACE 替换为实际机器人网卡
bash run_gr00t_yaw_real.sh --model real --real --net ROBOT_INTERFACE
bash run_gr00t_yaw_real.sh --model sim --real --net ROBOT_INTERFACE
```

脚本自动启动对应模型的服务，real-state 使用 8004 端口，sim-state 使用 8005。
`GR00T_YAW_PORT` 可覆盖端口；preflight 校验服务实际 checkpoint，拒绝连接到错误模型。
默认 4 步去噪，RTC 提交前缀 6 帧。服务 PID/日志在 `records/gr00t_yaw_server_real/`
和 `records/gr00t_yaw_server_sim/`。同一张 GPU 上建议逐个运行实验。

sim 是实际异步 DDS/MuJoCo 控制，推理不暂停物理时间；使用原录制 RGB 和 MuJoCo
实测 state，无车辆接触/仿真视觉闭环。终端按 `s → a → b` 对应 START → A → B，
`y` 复位，`x` 停止。默认 DDS domain 为 90/91，可用 `--domain` 覆盖。无界面模式：

```bash
touch /tmp/gr00t_yaw_keys.txt
bash run_gr00t_yaw_sim.sh --model sim --headless --key-file /tmp/gr00t_yaw_keys.txt
# 在另一终端、控制器各阶段就绪后，分别向文件追加 s、a、b；追加 x 结束。
```

新代码位于 `third_party/gr00t_rtc/src/gr00t_yaw/`、`src/deploy_gr00t_yaw.py`，没有
覆盖现有 replay/旧 GR00T 入口。复用当前 mild-wrist 初始化和 CHIP 862，右手平移
0.005 m/N、旋转 0，保留 8 rad/s slew 与原有 yaw guard。

Yaw 处理：state 在 A 时锁定实测初始 yaw，并从之后的观测中去掉它。模型输出和
RTC 队列始终是 `reference_yaw_canonical`，`t..t+23`，不加 state/last action。
只在送低层时执行 `yaw(q_bootstrap) * q_prediction`；低层已有 A 时对齐
`yaw(q_measured_A) * inverse(yaw(q_bootstrap))`，组合后目标等效于
`yaw(q_measured_A) * q_prediction`。只左乘 anchor，三点局部位置/姿态保持原值。
不把首个预测 yaw 减掉，不在每次请求时重置朝向，不向 RTC 回填已旋转的 action。
协议字段 `actions_absolute` 在这里表示 canonical 系下的绝对目标，而非世界系。

沿用的 yaw guard 会将参考 yaw 限制在 A-relative ±3°、3°/s；实测偏离 8°时
停止高层并要求 Y 复位。它可能裁剪模型预测中较大的转向。没有因换模型而移除保护。

验证：33 项协议/控制测试通过，包含首个非零 yaw 保留、任意启动 heading 的复合
旋转、局部三点不变、RTC canonical 前缀不随 state 变化。两个模型均通过不发送
真机控制指令的 preflight；另使用独立 DDS domain 跑异步仿真，日志保存在
`records/gr00t_yaw_{real,sim}_async_check_20261008*`。未执行真机运动。

异步复测（`*_v2/smoke_result.json`）：各 46 次推理、约 9 秒高层执行。
real-state 的耗时 P50/P95 为 59.98/108.26 ms，sim-state 为 60.82/63.46 ms，
两次复测均无 deadline miss、prefix hold、yaw stop 或 traceback。首次 real-state
短测曾出现一次 deadline miss 并随后触发 prefix hold，已保留原始日志；复测未
重现，不据此保证长期运行无超时。测试进程和这次启动的模型服务已退出，入口会自动重启。


## Perfect-controller state ablation (2026-10-09, Lambda 1679)

`chip_ideal_yaw_canonical` uses the existing merged real-replay dataset's images
and action labels unchanged. After canonicalizing the full action episode, it
sets `state[t] = action[max(t-1, 0)]` at the GR00T 30-Hz rate. Thus the three
history rows use action indices `max([t-7,t-4,t-1],0)`, while targets remain
`action[t:t+24]`. The first action is the assumed already-achieved initial hold;
this is explicit startup padding, not a cross-episode predecessor. This is a
one-30-Hz-action-step idealization, not a 50-Hz controller latency simulation.
No measured replay pose contributes to the model's state values. Source validity
masks and split seed 42 are retained for comparison. State and action share the
initial action heading and its continuous positive-initial-w quaternion branch.

Validation: six replay contract tests passed; all 10,132 train and 517 validation
windows checked, 93/5 episodes, images decoded at each episode's first/last sample.
An independent SciPy yaw calculation verifies state/action values within 2e-6.

Launch: `examples/CHIP/finetune_ideal_replay_yaw_a100_1gpu.slurm`, single GPU,
global batch 160, 20,000 optimizer steps, save every 1,000 steps, RTC max delay 6,
original base initialization and the same tuning settings as job 1485.
Lambda job: **1679**, isolated code snapshot:
`/share/ml/yangmin/gr00t-chip/runs/wiping_car_ideal_replay_yaw_20261009/Isaac-GR00T`.
Output root:
`/share/ml/yangmin/gr00t-chip/outputs/wiping_car_ideal_replay_yaw_1gpu_1679/wiping_car_ideal_replay_yaw_1gpu`.
Slurm logs are in the parent run directory as `chip-ideal-yaw-1679.out/.err`.
Existing datasets, checkpoints and deployments are not overwritten.

Startup verified: job 1679 reached 20 optimizer steps; logged loss 1.2798 -> 1.2749, finite gradient norms. Saved action statistics are exactly identical to job 1485; state statistics differ as intended. Steady startup throughput is about 2.9 seconds/step, so allow roughly 16–18 hours for 20,000 steps including checkpoints.
