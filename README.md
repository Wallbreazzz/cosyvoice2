# CosyVoice2 多进程流式 TTS 服务 - 一键部署

基于华为昇腾 NPU 的 CosyVoice2 多进程流式语音合成 HTTP 服务一键部署方案。

## 前置依赖

运行环境由 [CosyVoice2-0.5B](https://ai.gitcode.com/Ascend-SACT/CosyVoice2-0.5B) 仓库提供，包含：
- 昇腾 NPU 驱动及 toolkit
- conda 环境（cosyvoice）
- CosyVoice2 模型代码及依赖

请先按照该仓库说明完成环境搭建和模型权重下载。

## 一键部署

在 CosyVoice 项目根目录下（比如按前置依赖中的步骤执行的话，根目录就是/home/mind/model/CosyVoice）执行：

```bash
bash deploy.sh
```

脚本将自动生成以下 5 个文件，并修改 2 个模型代码文件：

| 文件 | 说明 |
|------|------|
| `server.py` | 多进程流式 TTS 服务（支持并发控制） |
| `client.py` | 测试客户端 |
| `run_server.sh` | 启动脚本（自动查找 conda + 加载 Ascend NPU 环境） |
| `bench_client.py` | HTTP 并发压测脚本 |
| `test_seeds.sh` | 音色调优脚本（遍历不同 seed 寻找最佳音色） |

模型代码修改：
- `cosyvoice/cli/model.py`：在 `CosyVoice2Model.tts()` 中添加 seed 支持
- `cosyvoice/cli/cosyvoice.py`：在所有推理方法中添加 seed 参数透传

## 启动服务

```bash
bash run_server.sh [model_dir] [num_workers] [port] [load_concurrency] [timeout] [cpu_bind] [seed]
```

示例（8 进程、端口 50000）：

```bash
bash run_server.sh ../weight/CosyVoice2-0.5B 8 50000
```

指定 seed（默认 9）：

```bash
bash run_server.sh ../weight/CosyVoice2-0.5B 8 50000 1 1800 "" 42
```

手动指定 CPU 绑核（绑定到 NUMA node 0 的 0~11 号核）：

```bash
bash run_server.sh ../weight/CosyVoice2-0.5B 8 50000 1 1800 0,1,2,3,4,5,6,7,8,9,10,11
```

参数说明：
- `model_dir`：模型目录路径
- `num_workers`：进程数，默认 2
- `port`：服务端口，默认 50000
- `load_concurrency`：最大并发加载数，默认 1（单 NPU 环境建议设为 1）
- `timeout`：每个 worker 启动超时（秒），默认 1800
- `cpu_bind`：可选，逗号分隔的 CPU 核编号。为空时自动探测 NUMA 拓扑并绑核
- `seed`：随机种子，默认 9。固定 seed 可确保相同文本生成完全一致的音频

> 如需指定 NPU 卡号，启动前设置 `ASCEND_RT_VISIBLE_DEVICES` 环境变量，如 `export ASCEND_RT_VISIBLE_DEVICES=0`。

## API 接口

服务启动后提供以下接口：

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/list_spks` | GET | 查询可用说话人列表 |
| `/inference_sft` | POST | SFT 模式推理（指定说话人） |
| `/inference_zero_shot` | POST | Zero-shot 模式推理（上传参考音频克隆声音） |
| `/inference_cross_lingual` | POST | 跨语言推理 |
| `/inference_instruct2` | POST | 指令控制推理 |

所有推理接口返回流式 PCM 音频（int16），响应头包含：
- `X-Sample-Rate`：采样率
- `X-Format`：音频格式（pcm-int16）

### 请求参数

**SFT 模式**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `tts_text` | string | 是 | 待合成文本 |
| `spk_id` | string | 是 | 说话人 ID（如"中文女"） |
| `stream` | bool | 否 | 是否流式，默认 true |
| `speed` | float | 否 | 语速，默认 1.0 |
| `seed` | int | 否 | 随机种子，默认 9。固定 seed 可确保相同文本生成完全一致的音频 |

**Zero-shot 模式**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `tts_text` | string | 是 | 待合成文本 |
| `prompt_text` | string | 是 | 参考音频对应文本 |
| `prompt_wav` | file | 是 | 参考音频文件 |
| `stream` | bool | 否 | 是否流式 |
| `speed` | float | 否 | 语速 |
| `seed` | int | 否 | 随机种子，默认 9 |

**Cross-lingual 模式**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `tts_text` | string | 是 | 待合成文本 |
| `prompt_wav` | file | 是 | 参考音频文件 |
| `stream` / `speed` | - | 否 | 同上 |
| `seed` | int | 否 | 随机种子，默认 9 |

**Instruct2 模式**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `tts_text` | string | 是 | 待合成文本 |
| `instruct_text` | string | 是 | 指令文本 |
| `prompt_wav` | file | 是 | 参考音频文件 |
| `stream` / `speed` | - | 否 | 同上 |
| `seed` | int | 否 | 随机种子，默认 9 |

## 客户端测试

```bash
# 健康检查
python3 client.py --mode health

# 查看可用说话人
python3 client.py --mode list_spks

# SFT 模式
python3 client.py --mode sft --tts_text '你好世界' --spk_id '中文女'

# SFT 模式（指定 seed）
python3 client.py --mode sft --tts_text '你好世界' --spk_id '中文女' --seed 42

# Zero-shot 模式
python3 client.py --mode zero_shot \
  --tts_text '收到好友从远方寄来的生日礼物' \
  --prompt_text '希望你以后能够做的比我还好呦。' \
  --prompt_wav asset/zero_shot_prompt.wav
```

## 音色调优

CosyVoice2 模型的 LLM 采样和 Flow 模型噪声生成依赖随机数。通过固定 `seed` 参数可以消除随机性，确保相同文本每次生成完全一致的音频。

不同 seed 值会产生不同的音色表现。使用 `test_seeds.sh` 可以批量测试不同 seed，找到最适合业务的音色：

```bash
# 测试 seed 0~20，每个 seed 生成 1 条音频
bash test_seeds.sh 0 20

# 音频保存在 seed_test/ 目录下
# 逐一试听，选择音色最佳的 seed 值
```

找到满意的 seed 后，将其设为服务启动的默认值：

```bash
# 例如最佳 seed 为 15
bash run_server.sh ../weight/CosyVoice2-0.5B 8 50000 1 1800 "" 15
```

> **注意**：seed 与说话人（`spk_id`）是独立的。同一个 seed 在不同说话人上的音色表现可能不同。如需多个说话人都达到最佳效果，建议分别为每个说话人测试最佳 seed。

## 并发压测

```bash
# 基础压测（8 并发，16 请求）
python3 bench_client.py --concurrency 8 --num_requests 16 --output_dir bench_output

# 混合文本长度压测
python3 bench_client.py --concurrency 8 --num_requests 16 --text_len mixed --output_dir bench_mixed

# Zero-shot 模式压测
python3 bench_client.py --concurrency 8 --num_requests 16 \
  --mode zero_shot --text_len medium \
  --prompt_text "希望你以后能够做的比我还好呦。" \
  --prompt_wav asset/zero_shot_prompt.wav \
  --output_dir bench_zero_shot

# 指定 seed 压测
python3 bench_client.py --concurrency 8 --num_requests 16 --seed 42 --output_dir bench_seed42
```

压测结果输出指标：
- **TTFT**：首包到达时间（Time To First Token）
- **RTF**：实时率（Real-Time Factor），越小越好
- **QPS**：每秒处理请求数

## 架构说明

服务采用多进程架构：
- 主进程运行 FastAPI + uvicorn，接收 HTTP 请求并分发
- 每个 Worker 进程独立加载一份模型实例，通过共享队列接收任务
- Worker 进程顺序预热（避免多进程同时推理导致 NPU 资源竞争）

### 性能优化

服务内置以下性能优化机制：

**CPU 绑核（CPU Affinity）**

自动探测系统 NUMA 拓扑，将所有 Worker 进程和主进程绑定到同一 NUMA node 内的 CPU 核上，避免跨 NUMA 访存和跨 L3 cache 的 IPC 开销。也可通过 `cpu_bind` 参数手动指定核列表。

**NUMA 内存绑定**

启动脚本自动检测 `numactl`，若可用则将进程内存分配绑定到对应的 NUMA node（`--membind`），确保内存访问走本地路径，降低跨 node 访存延迟。

**固定大小线程池**

流式响应的 `run_in_executor` 使用固定大小的 `ThreadPoolExecutor`（`max_workers = num_workers + 4`），避免线程数随 CPU 核数增长导致上下文切换开销增大。

**控制队列优化**

Worker 控制信号（ready/warmup/load_semaphore）使用 `mp.Queue` 替代 `Manager.Queue`，消除 Manager Server 单点 IPC 瓶颈。任务分发和结果回传仍使用 `Manager.Queue` 以支持跨进程 pickle 传递。

## 性能参考

以下为 8 进程并发、Ascend 910B3 NPU 环境下的参考性能数据（已启用 CPU 绑核 + NUMA 内存绑定优化）：

| 场景 | 并发数 | 请求数 | 成功率 | 平均RTF | 平均TTFT | QPS |
|------|--------|--------|--------|---------|----------|-----|
| SFT 短文本 | 8 | 16 | 100% | 0.567 | 0.505s | 3.78 |
| SFT 混合文本 | 8 | 16 | 100% | 0.692 | 0.525s | 1.45 |
| Zero-shot | 8 | 16 | 100% | 0.809 | 1.715s | 1.62 |

> **注意**：在多 NUMA node 的机器上（如 24 核双路），扩容 CPU/内存后若未启用绑核，性能可能反而下降。建议保持默认自动绑核行为，或通过 `cpu_bind` 参数显式指定同一 NUMA node 内的核列表。
