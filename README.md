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

脚本将自动生成以下 4 个文件：

| 文件 | 说明 |
|------|------|
| `server.py` | 多进程流式 TTS 服务（动态库发现 + 预加载 + 顺序预热） |
| `client.py` | 测试客户端 |
| `run_server.sh` | 启动脚本（conda 环境查找 + Ascend 环境加载 + modelscope 补丁） |
| `bench_client.py` | HTTP 并发压测脚本 |

## 启动服务

```bash
bash run_server.sh <模型目录> <进程数> <端口> <预热次数>
```

示例（8 进程、端口 50000）：

```bash
bash run_server.sh ../weight/CosyVoice2-0.5B 8 50000
```

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

**Zero-shot 模式**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `tts_text` | string | 是 | 待合成文本 |
| `prompt_text` | string | 是 | 参考音频对应文本 |
| `prompt_wav` | file | 是 | 参考音频文件 |
| `stream` | bool | 否 | 是否流式 |
| `speed` | float | 否 | 语速 |

**Cross-lingual 模式**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `tts_text` | string | 是 | 待合成文本 |
| `prompt_wav` | file | 是 | 参考音频文件 |
| `stream` / `speed` | - | 否 | 同上 |

**Instruct2 模式**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `tts_text` | string | 是 | 待合成文本 |
| `instruct_text` | string | 是 | 指令文本 |
| `prompt_wav` | file | 是 | 参考音频文件 |
| `stream` / `speed` | - | 否 | 同上 |

## 客户端测试

```bash
# 健康检查
python3 client.py --mode health

# 查看可用说话人
python3 client.py --mode list_spks

# SFT 模式
python3 client.py --mode sft --tts_text '你好世界' --spk_id '中文女'

# Zero-shot 模式
python3 client.py --mode zero_shot \
  --tts_text '收到好友从远方寄来的生日礼物' \
  --prompt_text '希望你以后能够做的比我还好呦。' \
  --prompt_wav asset/zero_shot_prompt.wav
```

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
- 动态发现并预加载昇腾 HCCL、OpenFst 等关键共享库

## 测试结果
测试一
```
[root@23a42ac44c1d CosyVoice]# python3 bench_client.py --concurrency 8 --num_requests 16 --output_dir bench_output

============================================================
  并发压测: concurrency=8, total_requests=16
  mode=sft, spk_id=中文女
  target=127.0.0.1:50000
============================================================

  req-003 OK ttft=0.518s elapsed=1.91s audio=3.16s rtf=0.603
  req-006 OK ttft=0.541s elapsed=1.95s audio=3.16s rtf=0.617
  req-002 OK ttft=0.529s elapsed=1.97s audio=3.16s rtf=0.622
  req-001 OK ttft=0.534s elapsed=1.98s audio=3.16s rtf=0.626
  req-005 OK ttft=0.541s elapsed=1.98s audio=3.16s rtf=0.627
  req-004 OK ttft=0.542s elapsed=2.00s audio=3.16s rtf=0.632
  req-000 OK ttft=0.537s elapsed=2.00s audio=3.16s rtf=0.633
  req-007 OK ttft=0.535s elapsed=2.01s audio=3.16s rtf=0.635
  req-008 OK ttft=0.454s elapsed=1.98s audio=4.12s rtf=0.480
  req-009 OK ttft=0.468s elapsed=2.02s audio=4.12s rtf=0.490
  req-011 OK ttft=0.484s elapsed=2.04s audio=4.12s rtf=0.495
  req-010 OK ttft=0.478s elapsed=2.13s audio=4.12s rtf=0.517
  req-015 OK ttft=0.477s elapsed=2.12s audio=4.12s rtf=0.514
  req-014 OK ttft=0.481s elapsed=2.14s audio=4.12s rtf=0.520
  req-012 OK ttft=0.490s elapsed=2.17s audio=4.12s rtf=0.527
  req-013 OK ttft=0.475s elapsed=2.22s audio=4.12s rtf=0.540

============================================================
  压测结果汇总
============================================================
  总请求数:       16
  并发数:         8
  成功:           16
  失败:           0
  总耗时:         4.23s
  平均响应时间:   2.04s
  最大响应时间:   2.22s
  最小响应时间:   1.91s
  平均音频时长:   3.64s
  平均RTF:        0.567
  平均TTFT:       0.505s
  最大TTFT:       0.542s
  最小TTFT:       0.454s
  QPS:            3.78
```
测试二
```
[root@23a42ac44c1d CosyVoice]# python3 bench_client.py --concurrency 8 --num_requests 16 --text_len mixed --output_dir bench_mixed

============================================================
  并发压测: concurrency=8, total_requests=16
  mode=sft, spk_id=中文女
  文本池大小=13, 模式=轮询取不同文本
  target=127.0.0.1:50000
============================================================

  req-001 OK text_len=3 ttft=0.598s elapsed=0.60s audio=0.60s rtf=1.001
  req-000 OK text_len=3 ttft=0.500s elapsed=0.94s audio=1.04s rtf=0.900
  req-002 OK text_len=3 ttft=0.489s elapsed=0.95s audio=1.12s rtf=0.849
  req-004 OK text_len=3 ttft=0.508s elapsed=0.99s audio=1.20s rtf=0.828
  req-003 OK text_len=4 ttft=0.497s elapsed=1.04s audio=1.32s rtf=0.792
  req-005 OK text_len=14 ttft=0.513s elapsed=1.76s audio=3.00s rtf=0.585
  req-013 OK text_len=3 ttft=0.514s elapsed=1.01s audio=1.08s rtf=0.935
  req-006 OK text_len=28 ttft=0.506s elapsed=2.81s audio=5.76s rtf=0.488
  req-014 OK text_len=3 ttft=0.576s elapsed=0.58s audio=0.52s rtf=1.111
  req-008 OK text_len=24 ttft=0.446s elapsed=2.84s audio=5.76s rtf=0.493
  req-007 OK text_len=31 ttft=0.493s elapsed=3.68s audio=7.56s rtf=0.486
  req-015 OK text_len=3 ttft=0.508s elapsed=0.97s audio=1.12s rtf=0.865
  req-009 OK text_len=28 ttft=0.565s elapsed=3.19s audio=6.60s rtf=0.483
  req-010 OK text_len=72 ttft=0.568s elapsed=7.31s audio=16.92s rtf=0.432
  req-011 OK text_len=100 ttft=0.537s elapsed=9.65s audio=24.16s rtf=0.400
  req-012 OK text_len=90 ttft=0.576s elapsed=10.01s audio=23.72s rtf=0.422

============================================================
  压测结果汇总
============================================================
  总请求数:       16
  并发数:         8
  成功:           16
  失败:           0
  总耗时:         11.07s
  平均响应时间:   3.02s
  最大响应时间:   10.01s
  最小响应时间:   0.58s
  平均音频时长:   6.34s
  平均RTF:        0.692
  文本长度范围:   3~100 字
  平均文本长度:   25.8 字
  平均TTFT:       0.525s
  最大TTFT:       0.598s
  最小TTFT:       0.446s
  QPS:            1.45
```
测试三
```
[root@23a42ac44c1d CosyVoice]# python3 bench_client.py --concurrency 8 --num_requests 16 --mode zero_shot --text_len medium --prompt_text "希望你以后能够做的比我还好呦。" --prompt_wav asset/zero_shot_prompt.wav --output_dir bench_zero_shot

============================================================
  并发压测: concurrency=8, total_requests=16
  mode=zero_shot, spk_id=中文女
  文本池大小=5, 模式=轮询取不同文本
  target=127.0.0.1:50000
============================================================

  req-000 OK text_len=14 ttft=1.886s elapsed=3.48s audio=3.56s rtf=0.977
  req-005 OK text_len=14 ttft=2.099s elapsed=3.88s audio=3.72s rtf=1.043
  req-006 OK text_len=28 ttft=1.925s elapsed=4.83s audio=5.72s rtf=0.844
  req-002 OK text_len=31 ttft=2.074s elapsed=5.06s audio=6.00s rtf=0.844
  req-001 OK text_len=28 ttft=2.103s elapsed=5.14s audio=5.80s rtf=0.886
  req-003 OK text_len=24 ttft=1.950s elapsed=5.69s audio=7.08s rtf=0.804
  req-007 OK text_len=31 ttft=1.890s elapsed=5.78s audio=7.08s rtf=0.817
  req-004 OK text_len=28 ttft=2.068s elapsed=6.22s audio=7.28s rtf=0.854
  req-008 OK text_len=24 ttft=1.424s elapsed=4.41s audio=6.04s rtf=0.731
  req-010 OK text_len=14 ttft=1.290s elapsed=3.10s audio=3.60s rtf=0.860
  req-009 OK text_len=28 ttft=1.708s elapsed=4.64s audio=6.00s rtf=0.774
  req-011 OK text_len=28 ttft=1.652s elapsed=4.06s audio=5.72s rtf=0.711
  req-015 OK text_len=14 ttft=1.496s elapsed=3.14s audio=3.92s rtf=0.800
  req-013 OK text_len=24 ttft=1.267s elapsed=3.79s audio=5.64s rtf=0.672
  req-014 OK text_len=28 ttft=1.182s elapsed=3.72s audio=5.72s rtf=0.650
  req-012 OK text_len=31 ttft=1.425s elapsed=4.73s audio=7.04s rtf=0.672

============================================================
  压测结果汇总
============================================================
  总请求数:       16
  并发数:         8
  成功:           16
  失败:           0
  总耗时:         9.88s
  平均响应时间:   4.48s
  最大响应时间:   6.22s
  最小响应时间:   3.10s
  平均音频时长:   5.62s
  平均RTF:        0.809
  文本长度范围:   14~31 字
  平均文本长度:   24.3 字
  平均TTFT:       1.715s
  最大TTFT:       2.103s
  最小TTFT:       1.182s
  QPS:            1.62
```
