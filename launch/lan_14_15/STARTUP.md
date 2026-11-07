# 启动教程（.14 / .15）

本文只覆盖两台目标机（192.168.1.14、192.168.1.15）的**离线**部署与启动。两台都不能连外网，所有代码通过 bundle 同步。

## 0. 各机的工作区

| | .14 | .15 |
|---|---|---|
| 仓库 | `~/project/Lsglang-egram-nvme-kv-nvme/lsglang165` | 同左 |
| 分支 | `feat/dsv41-665535-12-14-15` | 同左 |
| 内核 | **6.14.0-37-generic**（2026-10-25 从 6.11.0-17 升级，已 hold） | 同左 |
| 模型 | `/media/hello/Windows/models/DeepSeek-V4.1-Flash-1017`（NTFS 分区） | `~/models/DeepSeek-V4.1-Flash-1017` |
| Engram | NVMe 直查（`LK_ENGRAM_NVME=1`） | 主机表（`SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`，约 190 GiB） |
| 卡 / TP | 4 x RTX 4090 24G / TP4 | 4090 48G + RTX PRO 5000 / TP2 |
| 解码 | plain | DSPARK |
| 日志 | `logs/serve_14.log` | `logs/serve_15.log` |

两边都以 `git log --oneline -1` 显示的提交为准（写作时为 `73d64cccd3`），包含我们全部改动：

| 文件 | 改动 |
|---|---|
| `python/sglang/srt/layers/engram.py` | host 表按 1 GiB 粒度 `cudaHostRegister`（驱动 >2 GiB 必 EINVAL） |
| `python/sglang/srt/mem_cache/pool_host/common.py` | HiCache host 池：1 GiB 截断 + 按拷贝粒度对齐 + 失败二分重试 + 错误信息修复 |
| `python/sglang/kernels/ops/gemm/fp8_kernel.py` | 按当前 rank 的卡名查 FP8 调优表（异构 TP 组） |
| `python/sglang/srt/distributed/parallel_state.py` | `create_custom_parallel_group` 的对象集合通信用 CPU(gloo) 组 |
| `launch/lan_14_15/*` | 启动/测试脚本与实测参数 |

核对命令（两台都跑，应输出 2 / 7 / 1 / 1）：

```bash
cd ~/project/Lsglang-egram-nvme-kv-nvme/lsglang165
git log --oneline -1
grep -c _PIN_GRANULE python/sglang/srt/layers/engram.py
grep -c register_range python/sglang/srt/mem_cache/pool_host/common.py
grep -c current_device python/sglang/kernels/ops/gemm/fp8_kernel.py
grep -c 'get_world_group().all_gather_object' python/sglang/srt/distributed/parallel_state.py
```

## 1. 一次性检查

```bash
# conda 环境（两台同名）
source ~/anaconda3/etc/profile.d/conda.sh && conda activate Lsglang165

# CUDA toolkit：conda 环境里 nvcc 13.4 与 cudart 13.0 头文件不匹配，
# 启动脚本已自动指向 /usr/local/cuda（13.2），手工跑 JIT 时也要设：
export CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda PATH=$CUDA_HOME/bin:$PATH

# .14 的模型在 NTFS 分区，确认已挂载（fstab 已配 nofail）：
ls /media/hello/Windows/models            # 应看到 DeepSeek-V4.1-Flash-1017

# /dev/shm 大页（start.sh 会自动处理，权限不足时会 WARN）：
mount | grep ' /dev/shm '
```

## 2. 启动

```bash
cd ~/project/Lsglang-egram-nvme-kv-nvme/lsglang165

# .14 —— plain 解码 + L2 host 层(ratio 4) + L3 NVMe + text-only，全部已是默认
bash launch/lan_14_15/restart.sh 14

# .15 —— DSPARK + L2 host 层(ratio 2) + L3 NVMe + Engram host 表，全部已是默认
bash launch/lan_14_15/restart.sh 15
```

脚本默认值（两台一致，`DSV41_*` 可覆盖）：

```
--chunked-prefill-size 8192   --mem-fraction-static 0.88
--max-total-tokens 666112     --max-running-requests 1
LK_POOL_SLOTS 40(.14) / 24(.15)   LK_POOL_LOG=100 可看池命中
```

`restart.sh` = `stop.sh` + `start.sh` + 等 `/model_info` 返回正确模型名；冷启动约 9–14 分钟（`.15` 的 host 表更慢）。

## 3. 验证

```bash
curl -s http://127.0.0.1:8070/model_info | head -c 200
# 期望: {"model_path": "...", "served_model_name": "DeepSeek-V4.1-Flash-1017" ...}
```

跑一个 32K 请求（单发、400 token 才是稳态吞吐，见第 5 节）：

```bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
source ~/anaconda3/etc/profile.d/conda.sh && conda activate Lsglang165
export PYTHONPATH=$PWD/python
M=$HOME/models/DeepSeek-V4.1-Flash-1017
[ "$(hostname -I | grep -c 192.168.1.14)" = 1 ] && M=/media/hello/Windows/models/DeepSeek-V4.1-Flash-1017
python launch/lan_14_15/client.py gen --model $M --max-new 400 --tokens 32768 --nonce 1
```

预期量级（单发、400 token）：

| | .14 | .15 |
|---|---|---|
| 32K decode | ~35 t/s | ~47 t/s（DSPARK） |
| 131K decode | ~35 t/s | ~32 t/s |
| 524K decode | ~35 t/s | ~31 t/s |
| 524K prefill | ~1550 t/s | ~1580 t/s |

## 4. 停止 / 重启

```bash
bash launch/lan_14_15/stop.sh 14 --all      # 必须带 --all：只 SIGTERM 父进程会留下持显存的 worker
bash launch/lan_14_15/restart.sh 14
```

## 5. 已知限制与排错

- **decode 必须用 ≥400 token 的输出测量**。64 token 的窗口里每请求启动开销（plain ~0.4 s、spec ~1.1 s）占三成以上，会把 600K 误读成“断崖”。
- **前缀复用要热身**：同一会话前 3 轮 `#cached-token=0`，第 4 轮起整段命中（SWA 门控）。单轮重复提示词永远不命中。
- **双并发会掉**：两路并发聚合只有单路的 ~40%，脚本因此固定 `--max-running-requests 1`。
- **OOM 之后要重启机器**：崩溃会把 pinned 内存留在驱动里（曾见 335–360 GiB 不回收），此时再启动会以 `CUDA error: invalid argument` 失败。`sudo systemctl reboot` 后即恢复。
- **6.14 后 BMC 显示器分辨率**：两台显示器接在主板 ast 上，6.14 的 ast 驱动拒绝 Wayland 的 atomic 提交，会黑屏/掉到 1024x768。修复脚本在 `10-kernel-6.14/fix-display-6.14.sh`（.14 用 `DP-1`、.15 用 `VGA-1`）：强制 Xorg、`AutoAddGPU false`、`Atomic off`、`PreferredMode 1920x1080`。
- **`text-only` 与图像**：`.14` 为省内存开了 `DSV41_TEXT_ONLY=1`，此时图像输入不可用；去掉它会多占约 40–50 GiB 主机内存。
- **L3（NVMe KV）暂时不要开**：能写不能读回，且 decode 掉到 ~4 t/s，详见 README 的 L3 章节。
- **spec + 大池子会 OOM**：`.14` 上 DSPARK + pool 40 在 524K 解码约 200 token 后 OOM，长上下文要么 plain、要么把池子降到 24。
## 输出预算与参数透传

**输出没有被启动命令限制。** 服务端没有 max_new_tokens 上限，请求里的
max_new_tokens 会照常执行；唯一的硬约束是上下文长度检查：

`
input_tokens + max_new_tokens <= --context-length
`

两个脚本都写死了 --context-length 665535（模型上限），并且
--allow-auto-truncate 默认关闭——超限时服务端返回 400
（The input (N tokens) is longer than the model's context length），
客户端现在会把这个错误原样抛出，而不是假装成功返回 0 个 token。

单请求的输入/输出预算（--context-length 665535）：

| max_new | 单请求最大 input | 两路并发时每路最大 input（KV 池 666112） |
|---|---|---|
| 262 144（256K） | 403 391 | ≤ 70 912 |
| 8 192 | 657 343 | ≤ 324 864 |
| 400 | 665 135 | ≤ 332 656 |

两路并发那列按 2 x (input + max_new) <= --max-total-tokens 算：两路各跑满
256K 输出时，各自上下文只能到约 70K；否则会排队（不是 OOM）。

### 透传

`ash
# 追加任意 sglang 参数（同名参数后写覆盖前写）
bash launch/lan_14_15/start.sh 14 --allow-auto-truncate --context-length 700000

# 追加 LK_* 环境变量
DSV41_EXTRA='LK_POOL_LOG=100 LK_POOL_WINDOW=64' bash launch/lan_14_15/start.sh 14

# 已有的 DSV41_* 旋钮依然生效
DSV41_MAX_TOTAL=800000 DSV41_POOL_SLOTS=40 DSV41_CHUNK=8192 bash launch/lan_14_15/start.sh 14
`

start.sh ... [args] → serve_XX.sh "$@"，参数追加在脚本自带参数之后，因此
重复的 flag 以你传入的为准；DSV41_EXTRA 是追加在 env 链尾部的赋值，
同样后写覆盖。