# GLM-5.2 8×A100 GPU 基线实验操作手册

## 1. 目的与适用范围

本文档供拿到一台基本为空的 Linux x86_64、8×NVIDIA A100 服务器的同事使用，
目标是从零建立 TorchTitan GPU 环境，并完成当前 GLM debug 模型的可复现实验。

当前里程碑支持 GLM 单设备训练和 8 卡数据并行训练（DDP/HSDP 与 FSDP）。8 张
A100 中单卡完成模型精度和训练验收；8 卡完成 DDP 与 FSDP 训练验收（见 G6）。
当前代码仍主动拒绝 TP、CP、PP、EP 及其他混合并行配置。

本文档验收的是：

1. 当前源码能在 A100 上导入和构建；
2. TorchTitan 缩小模型能与相同缩小配置的 Transformers 模型完成 FP32 组件和
   BF16 logits、loss、MoE block、代表性梯度对齐；
3. TorchTitan 单卡 10 步训练能正常结束；
4. 8 张 A100 能完成基本 NCCL all-reduce；
5. TorchTitan GLM 8 卡 DDP 与 FSDP 各 10 步训练能正常结束。

本文档不验收正式完整 GLM-5.2 checkpoint，不验收 TP/CP/PP/EP 等混合并行训练，
也不产生 A3/A100 性能结论。

除非小节另有说明，命令均在 Linux Bash 中执行。文中的 `/path/to/...` 是路径示例，
执行者必须替换为本机真实绝对路径；其他命令可以按门禁顺序直接复制执行。

## 2. 固定基线与停止条件

| 项目 | 基线 |
| --- | --- |
| 代码仓库 | `https://github.com/loveofguoke/torchtitan.git` |
| 开发分支 | `feat/glm5-model-distributed` |
| 最低模型提交 | `4bc6832c1f01761811e6e5695a24297f992953b4` |
| Python | 3.12 |
| PyTorch | 与当前 TorchTitan 源码匹配的 CUDA Nightly；首次通过后冻结实际版本 |
| CUDA wheel | `cu130` |
| NVIDIA 驱动 | Linux R580 或更新版本 |
| Transformers | 5.14.1 |
| GPU | 8×A100；模型实验固定使用物理 GPU 0 |

出现以下任一情况时停止，不要继续执行后续门禁：

- `nvidia-smi` 不可用、GPU 数量不是 8，或者型号不是 A100；
- 驱动主版本低于 580；
- 团队远程仓库没有 `feat/glm5-model`；
- 当前提交不包含上表所列最低模型提交；
- `pip check` 失败；
- PyTorch 看不到 8 张 GPU，或者 A100 BF16 不可用；
- GPU parity 出现失败、错误或跳过；
- 训练 loss 出现 NaN/Inf、CUDA OOM 或进程异常退出。

CUDA 13.x 需要 R580 或更新驱动。PyTorch 官方 TorchTitan 当前源码安装说明使用
PyTorch Nightly，并以 `cu130` 为默认 CUDA wheel。PyTorch wheel 已包含运行时所需
CUDA 库，本手册的测试不要求额外安装完整 CUDA Toolkit 或 `nvcc`。

## 3. 负责人先决操作：发布开发分支

当前开发分支最初建立在开发机本地。其他机器执行前，仓库维护者必须在开发机确认
变更范围并把分支推送到团队 fork。该操作会修改远程仓库，只能由获得发布授权的
维护者执行：

```bash
git -C /path/to/pytorch-torchtitan-glm5 status --short --branch
git -C /path/to/pytorch-torchtitan-glm5 log --oneline origin/main..HEAD
git -C /path/to/pytorch-torchtitan-glm5 push -u team feat/glm5-model
```

如果团队分支没有推送，GPU 机器执行者应停止并联系维护者，不要从聊天附件复制零散
源码，也不要基于 `origin/main` 猜测补丁。

## 4. G0：裸机预检

### 4.1 系统工具

机器至少需要以下系统工具：

- `bash`、`git`、`curl`、`tar`；
- 可访问 GitHub、PyPI 和 `download.pytorch.org`；
- 可写的个人目录和足够的环境、源码、日志空间；
- 已安装并加载 NVIDIA 数据中心驱动。

如果系统工具或驱动缺失，由机器管理员安装。本手册不提供发行版相关的 root/驱动
安装命令，避免在共享训练服务器上误改内核或驱动。

### 4.2 GPU 与驱动检查

```bash
set -o pipefail
nvidia-smi
nvidia-smi -L
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
  --format=csv,noheader
nvidia-smi topo -m
```

验收要求：

- 恰好列出 8 张 A100；
- 8 张卡状态正常，没有不可恢复的 ECC/Xid 错误；
- Linux 驱动为 R580 或更新版本；
- 记录 A100 是 40 GB 还是 80 GB 版本及拓扑。

驱动低于 R580 时停止并申请升级。不要在同一次基线实验中自行换用其他 CUDA wheel；
如项目决定使用 `cu126` 等非默认环境，应先作为新的环境基线记录并重新评审。

## 5. 获取代码

选择有足够空间的工作目录执行：

```bash
git clone https://github.com/loveofguoke/torchtitan.git torchtitan-glm5
cd torchtitan-glm5
git remote add upstream https://github.com/pytorch/torchtitan.git
git ls-remote --exit-code --heads origin feat/glm5-model
git fetch origin feat/glm5-model
git switch --track -c feat/glm5-model origin/feat/glm5-model
git merge-base --is-ancestor \
  4bc6832c1f01761811e6e5695a24297f992953b4 HEAD
git status --short --branch
git rev-parse HEAD
```

验收要求：

- `git ls-remote` 和 `git merge-base --is-ancestor` 返回 0；
- 当前分支是 `feat/glm5-model`；
- `git status --short` 没有源码改动；
- 记录完整 HEAD。实验期间不得切换分支或拉取新提交。

## 6. 从零创建 Python 环境

### 6.1 安装 Miniforge

如果机器已有可用 Conda/Miniforge，可以跳过安装，只执行环境创建。否则：

```bash
MINIFORGE_ROOT="${HOME}/miniforge3"
MINIFORGE_INSTALLER="/tmp/Miniforge3-Linux-x86_64.sh"
curl -fsSL \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
  -o "${MINIFORGE_INSTALLER}"
bash "${MINIFORGE_INSTALLER}" -b -p "${MINIFORGE_ROOT}"
source "${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
conda create -y -n torchtitan-glm5-gpu python=3.12
conda activate torchtitan-glm5-gpu
python --version
```

以后打开新 shell 时，需要重新执行：

```bash
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate torchtitan-glm5-gpu
cd /path/to/torchtitan-glm5
```

### 6.2 安装 PyTorch Nightly 和项目依赖

必须在仓库根目录和新环境中执行：

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install --pre --upgrade torch \
  --index-url https://download.pytorch.org/whl/nightly/cu130
python -m pip install -r requirements.txt
python -m pip install --pre --upgrade torchdata \
  --index-url https://download.pytorch.org/whl/nightly/cpu
python -m pip install \
  "transformers==5.14.1" \
  "pytest==7.3.2" \
  "expecttest>=0.2.0" \
  "pre-commit" \
  "pyrefly==0.45.1"
python -m pip install -e . --no-deps
python -m pip check
```

不要直接 `pip install torchtitan` 替代 editable source install，否则运行的可能是 PyPI
版本而不是 `feat/glm5-model` 源码。

Nightly 会随时间变化。首次 GPU 验收通过后，应把 `torch.__version__`、wheel CUDA
版本和 `pip freeze` 归档；后续复现实验优先使用该已验证版本组合。

## 7. 建立本次实验目录

在激活环境并进入仓库根目录后执行：

```bash
RUN_ID="glm5_a100_$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_ROOT="${TORCHTITAN_EXPERIMENT_ROOT:-${PWD}/../torchtitan-experiments}"
RUN_DIR="${EXPERIMENT_ROOT}/${RUN_ID}"
mkdir -p "${RUN_DIR}"
git rev-parse HEAD | tee "${RUN_DIR}/git-head.txt"
git status --short --branch | tee "${RUN_DIR}/git-status.txt"
nvidia-smi -q > "${RUN_DIR}/nvidia-smi-q.txt"
nvidia-smi topo -m > "${RUN_DIR}/nvidia-topology.txt"
python --version 2>&1 | tee "${RUN_DIR}/python-version.txt"
python -m pip freeze > "${RUN_DIR}/pip-freeze.txt"
```

后续所有命令必须在同一个 shell 中运行，以保留 `RUN_DIR`。如果开启新 shell，先根据
实际路径重新设置 `RUN_DIR`。

## 8. G1：CUDA、BF16 和 Transformers 导入检查

```bash
python - <<'PY' 2>&1 | tee "${RUN_DIR}/environment-check.log"
import torch
import transformers
from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM

print("torch:", torch.__version__)
print("torch wheel CUDA:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

assert transformers.__version__ == "5.14.1"
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 8
assert torch.cuda.is_bf16_supported()

for device_index in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(device_index)
    capability = torch.cuda.get_device_capability(device_index)
    print(device_index, name, capability)
    assert "A100" in name
    assert capability == (8, 0)

x = torch.randn(256, 256, device="cuda:0", dtype=torch.bfloat16, requires_grad=True)
loss = (x @ x.T).float().square().mean()
loss.backward()
torch.cuda.synchronize()
print("BF16 forward/backward OK; loss=", loss.item())
print("GLM config class:", GlmMoeDsaConfig.__name__)
print("GLM model class:", GlmMoeDsaForCausalLM.__name__)
PY
```

所有断言必须通过。这里不下载 Hugging Face checkpoint。

## 9. G2：CPU 功能回归

CPU 门禁用于先排除模型结构、state-dict adapter、mask、训练配置和反向传播错误：

```bash
python -m pytest \
  tests/unit_tests/test_glm5.py \
  tests/unit_tests/test_config_manager.py \
  -v --junitxml="${RUN_DIR}/cpu-functional.xml" \
  2>&1 | tee "${RUN_DIR}/cpu-functional.log"
```

验收要求：退出码为 0，没有 failed/error。CPU BF16 不支持相关 warning 可以记录，但不能
出现测试失败。

## 10. G3：单卡 Transformers 精度对齐

清空 GPU 0 上的其他任务，确认显存空闲后执行：

```bash
nvidia-smi -i 0
CUDA_VISIBLE_DEVICES=0 python -m pytest \
  tests/unit_tests/test_glm5_parity.py \
  -v -rs --junitxml="${RUN_DIR}/gpu-parity.xml" \
  2>&1 | tee "${RUN_DIR}/gpu-parity.log"
```

验收要求：

- 命令退出码为 0；
- 该文件的 7 个测试全部通过；
- 不能出现 skipped、failed 或 error；
- FP32 indexer/router/attention 组件比较通过；
- BF16 logits、causal loss、MoE block 和代表性梯度比较通过；
- Transformers 与 TorchTitan 的 indexer 参数都保持无梯度。

测试中的容差就是当前验收标准。数值断言失败时不得擅自扩大容差或改成 skip；应归档
日志、commit 和环境版本后定位首个偏差组件。

## 11. G4：单卡 10 步 TorchTitan 训练

当前 debug config 使用仓库内测试 tokenizer 和 `c4_test` 数据，不需要下载正式模型或
外部训练集。执行：

```bash
nvidia-smi -i 0
CUDA_VISIBLE_DEVICES=0 \
NGPU=1 \
MODULE=glm5 \
CONFIG=glm5_debugmodel \
LOG_RANK=0 \
./run_train.sh --training.steps 10 \
  2>&1 | tee "${RUN_DIR}/single-gpu-train.log"
```

验收要求：

- 只启动 1 个 worker；
- 完成 10 步并以退出码 0 结束；
- 每步 loss 都是有限值，没有 NaN/Inf；
- 没有 CUDA OOM、非法显存访问或 NCCL error；
- 日志记录模型参数量、每步时间、峰值显存和 loss。

本步骤证明 TorchTitan 单卡训练入口可用，不替代 G3 的 Transformers 数值对齐。

## 12. G5：8 卡 NCCL 健康检查

本步骤只验证机器的 8 卡通信环境，不运行 GLM。先在实验目录生成临时脚本：

```bash
cat > "${RUN_DIR}/nccl_smoke.py" <<'PY'
import os

import torch
import torch.distributed as dist


rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")

value = torch.tensor(float(rank + 1), device=f"cuda:{local_rank}")
dist.all_reduce(value)
torch.cuda.synchronize()
assert value.item() == 36.0, (rank, value.item())
print(f"rank={rank} local_rank={local_rank} all_reduce={value.item()}")

dist.barrier()
dist.destroy_process_group()
PY

NCCL_DEBUG=INFO torchrun --standalone --nproc_per_node=8 \
  "${RUN_DIR}/nccl_smoke.py" \
  2>&1 | tee "${RUN_DIR}/nccl-smoke.log"
```

验收要求：8 个 rank 均输出 `all_reduce=36.0`，命令退出码为 0，没有 NCCL timeout、
unhandled system error 或进程异常退出。

GLM 的 8 卡训练已由 G6 覆盖；本步骤只做通信环境健康检查。

## 12.5. G6：8 卡 GLM DDP/FSDP 训练

在 G4 与 G5 都通过后执行。两条命令都在 8 张 A100 上运行 GLM debug 模型 10 步：
DDP（复本式权重）与 FSDP（分片权重）各一次。

```bash
# DDP/HSDP: replicate=8, shard=1
NGPU=8 MODULE=glm5 CONFIG=glm5_debugmodel LOG_RANK=0 \
  ./run_train.sh --parallelism.data_parallel_replicate_degree 8 \
    --parallelism.data_parallel_shard_degree 1 --training.steps 10 \
    2>&1 | tee "${RUN_DIR}/8gpu-ddp-train.log"

# FSDP: replicate=1, shard=8
NGPU=8 MODULE=glm5 CONFIG=glm5_debugmodel LOG_RANK=0 \
  ./run_train.sh --parallelism.data_parallel_replicate_degree 1 \
    --parallelism.data_parallel_shard_degree 8 --training.steps 10 \
    2>&1 | tee "${RUN_DIR}/8gpu-fsdp-train.log"
```

验收要求（两条命令分别满足）：

- 各启动 8 个 worker，完成 10 步并以退出码 0 结束；
- 每步 loss 都是有限值，没有 NaN/Inf；
- 所有 rank 打印相同的全局 loss（trainer 会在 DP mesh 上做跨卡归约）；
- 没有 CUDA OOM、非法显存访问或 NCCL timeout；
- 第 2 步不报 `set_timeout` 相关 AttributeError（torch 2.12 已移除该 API）。

可选：给两条命令都加 `--debug.seed 42 --debug.deterministic`，对比两条 loss/grad_norm
曲线——同为数据并行，结果应当一致。

## 13. 结果摘要与归档

完成后创建 `${RUN_DIR}/result-summary.md`，至少填写：

```markdown
# GLM A100 experiment result

- Date:
- Operator:
- Host/cluster:
- GPU model and memory:
- NVIDIA driver:
- Git commit:
- Python:
- PyTorch:
- PyTorch wheel CUDA:
- Transformers:
- G1 environment check: PASS/FAIL
- G2 CPU functional tests: PASS/FAIL
- G3 one-GPU parity: PASS/FAIL
- G4 one-GPU 10-step training: PASS/FAIL
- G5 eight-GPU NCCL smoke: PASS/FAIL
- First failure or blocker:
- Log directory:
- Notes:
```

归档整个 `${RUN_DIR}`。把结果摘要和稳定存储位置链接追加到
`glm5_2_ascend_training_plan.md` 的“执行跟踪”和“精度与性能证据索引”。大日志、
profile 和 checkpoint 不应直接提交到 Git。

## 14. 常见故障处理

### 14.1 团队远程没有开发分支

现象：`git ls-remote --exit-code --heads origin feat/glm5-model` 非零退出。

处理：停止，联系维护者推送分支。不要改用 upstream main。

### 14.2 `torch.cuda.is_available()` 为 False

依次确认：

1. `nvidia-smi` 是否正常；
2. 是否激活了 `torchtitan-glm5-gpu`；
3. `torch.__version__` 是否包含 CUDA build；
4. `torch.version.cuda` 是否为 13.x；
5. 是否错误安装了 CPU-only torch。

修复环境后重新执行 G1，不要直接跳到训练。

### 14.3 Transformers parity 被跳过

在 A100 基线机上，parity skip 视为验收失败。确认 Transformers 恰好为 5.14.1、
GLM classes 可导入且 `CUDA_VISIBLE_DEVICES=0` 可见，然后重跑完整 parity 文件。

### 14.4 数值断言失败

不要放宽容差。保留 `gpu-parity.log`、JUnit XML、git commit、pip freeze 和 GPU/驱动
信息。先判断失败属于 indexer、router、attention、MoE block、logits/loss 或梯度，
再在相同环境下做最小复现。

### 14.5 单卡训练 OOM

debug 模型在 A100 上不应 OOM。检查 GPU 0 是否有其他进程、是否误用了完整模型配置、
是否启动了多于 1 个 worker，以及环境中是否存在遗留进程。不要通过改变模型规格掩盖
基线异常。

### 14.6 NCCL 健康检查失败

保存 `nccl-smoke.log` 和 `nvidia-smi topo -m`，检查所有 GPU 是否可见、驱动是否一致、
容器是否开放 GPU/P2P、共享服务器是否有残留进程。模型分布式适配尚未开始，因此
NCCL 环境失败应先由集群/系统侧解决。

## 15. 官方参考

- TorchTitan 源码安装与 Nightly 要求：
  https://github.com/pytorch/torchtitan#installation
- PyTorch 安装入口：
  https://pytorch.org/get-started/locally/
- NVIDIA CUDA driver 兼容性：
  https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html
- Transformers 安装说明：
  https://huggingface.co/docs/transformers/installation
