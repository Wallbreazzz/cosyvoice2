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

---

## 非 root 用户部署改造指南（去 root 化）

### 问题背景

当前镜像内所有文件（Python site-packages、conda 环境、Ascend toolkit 等）均以 root 用户安装，运行时也默认以 root 身份启动。当客户安全规范要求**禁止以 root 用户运行任何脚本和服务**时，切换到普通用户（如 HwHiAiUser）后会遇到一系列权限问题，典型报错如下：

```
ERROR: Fst::WriteFile: Can't open file: /usr/local/lib/python3.11/site-packages/tn/zh_tn_tagger.fst
_pywrapfst.FstIOError: Write failed: '/usr/local/lib/python3.11/site-packages/tn/zh_tn_tagger.fst'
```

**根因**：WeTextProcessing（tn 库）在初始化中文/英文 Normalizer 时，默认将 FST 缓存文件写入 Python site-packages 目录（root 所有），非 root 用户无法写入。

### 权限问题全景清单

切换到非 root 用户（如 HwHiAiUser）后，预计会遇到以下 5 类权限问题：

| # | 问题描述 | 报错路径/命令 | 严重程度 | 触发时机 |
|---|---------|-------------|---------|---------|
| 1 | WeTextProcessing 中文 Normalizer 写 FST 缓存到 site-packages | `/usr/local/lib/python3.11/site-packages/tn/zh_tn_tagger.fst` 等 | **致命** | 每次启动（`overwrite_cache=True`） |
| 2 | WeTextProcessing 英文 Normalizer 写 FST 缓存到 site-packages | 同上目录，`en_tn_tagger.fst` 等 | **致命** | 首次启动（`overwrite_cache=False`） |
| 3 | run_server.sh 中 `sed -i` 修改 modelscope 的 ast_utils.py | `/usr/local/lib/python3.11/site-packages/modelscope/utils/ast_utils.py` | **严重** | 每次启动 |
| 4 | run_server.sh 中 `pip install` 安装 FastAPI 等依赖 | `pip install fastapi uvicorn python-multipart requests` | **严重** | 每次启动 |
| 5 | 镜像内所有目录/文件为 root 所有，非 root 用户无写权限 | `/opt/conda/`、`/usr/local/lib/` 等全局 | **基础** | 全局影响 |

### 问题分类：模型代码 vs 服务脚本

| 问题 | 来源 | 说明 |
|------|------|------|
| #1 ZhNormalizer `overwrite_cache=True` 写 site-packages | **模型代码**（`cosyvoice/cli/frontend.py:71`） | CosyVoice 上游代码的设计缺陷：默认 cache_dir 指向 site-packages |
| #2 EnNormalizer 首次写 site-packages | **模型代码**（`cosyvoice/cli/frontend.py:72`） | 同上，首次运行也需写 FST 文件 |
| #3 `sed -i` 修改 modelscope | **服务脚本**（`deploy.sh` 内嵌的 `run_server.sh` Step 2） | 我们在启动脚本中引入的运行时补丁 |
| #4 `pip install` FastAPI | **服务脚本**（`run_server.sh` Step 3） | 我们在启动脚本中引入的运行时安装 |
| #5 镜像文件全部 root 所有 | **镜像构建**（Dockerfile 无 USER 指令） | 镜像层面的问题 |

**结论**：当前报错的核心原因是**模型代码本身**的问题（#1），不是服务脚本造成的。但服务脚本（#3、#4）和镜像构建（#5）引入了额外的权限问题，需要一并解决。

---

### 改造方案：三层改造

以下方案从镜像构建、模型代码、服务脚本三个层面彻底消除 root 权限依赖。建议三层全部完成，确保在任何非 root 环境下都能正常运行。

#### 第一层：Dockerfile 改造（最关键）

在 Dockerfile 中完成所有需要 root 权限的操作（安装、补丁、预构建），然后切换到非 root 用户。这样运行时不再需要任何 root 权限。

在现有 Dockerfile 的所有安装命令**之后**，添加以下内容：

```dockerfile
# ====== 去 root 化改造 ======

# Step A: 预构建 WeTextProcessing FST 缓存（构建时以 root 完成，运行时无需再写）
# 这一步会生成 zh_tn_tagger.fst / zh_tn_verbalizer.fst / en_tn_tagger.fst / en_tn_verbalizer.fst
RUN conda activate cosyvoice && \
    python3 -c "from tn.chinese.normalizer import Normalizer as ZhNormalizer; ZhNormalizer(overwrite_cache=True)" && \
    python3 -c "from tn.english.normalizer import Normalizer as EnNormalizer; EnNormalizer()"

# Step B: 预打 modelscope ast_utils.py 补丁（不再需要在运行时 sed -i）
# 注意：路径需要根据实际 Python 版本和安装位置调整
# 查找方法：python3 -c "import modelscope; print(modelscope.__file__)"
RUN AST_UTILS=$(python3 -c "import os, modelscope; print(os.path.join(os.path.dirname(modelscope.__file__), 'utils', 'ast_utils.py'))") && \
    if [ -f "$AST_UTILS" ] && ! grep -q 'getattr(node, field, None)' "$AST_UTILS"; then \
        sed -i 's/attr = getattr(node, field)$/attr = getattr(node, field, None)/' "$AST_UTILS"; \
        echo "Patched modelscope ast_utils.py: $AST_UTILS"; \
    fi

# Step C: 将关键目录的 ownership 赋予 HwHiAiUser，确保运行时可写
# 注意：根据实际镜像中的用户名和目录调整，以下为昇腾镜像典型路径
ARG RUNTIME_USER=HwHiAiUser
RUN chown -R ${RUNTIME_USER}:${RUNTIME_USER} /opt/conda \
 && chown -R ${RUNTIME_USER}:${RUNTIME_USER} /home/${RUNTIME_USER} \
 && if [ -d /usr/local/lib/python3.11/site-packages ]; then \
        chown -R ${RUNTIME_USER}:${RUNTIME_USER} /usr/local/lib/python3.11/site-packages; \
    fi

# Step D: 切换到非 root 用户
USER ${RUNTIME_USER}
WORKDIR /home/${RUNTIME_USER}/CosyVoice
```

**注意事项**：
- `Step C` 中的 chown 路径需要根据实际镜像结构调整。如果 Python 包安装在 `/opt/conda/envs/cosyvoice/lib/python3.10/site-packages/` 而非 `/usr/local/lib/python3.11/site-packages/`，则 chown 那条路径即可（因为 `/opt/conda` 已被整体 chown）。
- 如果镜像中尚未创建 HwHiAiUser 用户，需在 Dockerfile 中先添加 `RUN useradd -m -u 1000 HwHiAiUser`。
- 如果客户使用其他用户名（如 hw），将 `ARG RUNTIME_USER` 的默认值改为对应用户名即可。

#### 第二层：模型代码改造（消除运行时对 site-packages 的写入）

修改 `cosyvoice/cli/frontend.py`，将 WeTextProcessing 的 FST 缓存目录从 site-packages 改为用户 home 下的可写目录。

**修改文件**：`cosyvoice/cli/frontend.py`

**修改前**（第 71-72 行）：

```python
        else:
            self.zh_tn_model = ZhNormalizer(remove_erhua=False, full_to_half=False, overwrite_cache=True)
            self.en_tn_model = EnNormalizer()
            self.inflect_parser = inflect.engine()
```

**修改后**：

```python
        else:
            tn_cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'cosyvoice', 'tn')
            os.makedirs(tn_cache_dir, exist_ok=True)
            self.zh_tn_model = ZhNormalizer(remove_erhua=False, full_to_half=False, overwrite_cache=True, cache_dir=tn_cache_dir)
            self.en_tn_model = EnNormalizer(cache_dir=tn_cache_dir)
            self.inflect_parser = inflect.engine()
```

**原理说明**：
- `ZhNormalizer` 和 `EnNormalizer` 默认 `cache_dir=files("tn")`，指向 site-packages 内的 tn 包目录（root 所有，不可写）
- 改为 `~/.cache/cosyvoice/tn/` 后，FST 缓存文件写到用户 home 目录下，任何用户都有写权限
- `overwrite_cache=True` 保持不变（中文 Normalizer 每次启动仍会重写缓存，但写入位置改为用户目录）
- 这个改动兼容 root 和非 root 用户：root 的 home 是 `/root`，HwHiAiUser 的 home 是 `/home/HwHiAiUser`，各自可写

#### 第三层：服务脚本改造（消除运行时 pip install 和 sed -i）

修改 `deploy.sh` 中内嵌的 `run_server.sh` 部分，删除运行时修改系统文件的操作。

**修改文件**：`deploy.sh`

**删除 Step 2（modelscope 补丁）**——整个 for 循环块（约第 674-679 行）：

```bash
# ===== Step 2: Patch modelscope for Python 3.11 compatibility (fallback) =====
# ... 删除整个 Step 2 ...
```

> 此补丁已在 Dockerfile Step B 中预完成，运行时无需再执行。

**删除 Step 3（pip install）**——整行（约第 683 行）：

```bash
pip install fastapi uvicorn python-multipart requests 2>/dev/null | tail -1
```

> FastAPI 等依赖应在镜像构建时安装（在 Dockerfile 中添加 `RUN pip install fastapi uvicorn python-multipart requests`），运行时无需再安装。

如果确实需要在运行时安装 pip 包（如临时补丁），应改用 `pip install --user`：

```bash
pip install --user fastapi uvicorn python-multipart requests 2>/dev/null | tail -1
```

---

### 改造验证步骤

完成三层改造后，按以下步骤验证非 root 用户能否正常启动服务：

```bash
# 1. 确认当前用户不是 root
id
# 期望输出：uid=1000(HwHiAiUser) ... 而非 uid=0(root)

# 2. 确认 FST 缓存目录可写
ls -la ~/.cache/cosyvoice/tn/
# 期望：目录存在且归属当前用户

# 3. 确认 modelscope 补丁已生效（Dockerfile 预打补丁）
python3 -c "import modelscope.utils.ast_utils; print('OK')"
# 期望：无报错

# 4. 确认 tn 缓存写到用户目录而非 site-packages
python3 -c "
from tn.chinese.normalizer import Normalizer as ZhNormalizer
import os
cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'cosyvoice', 'tn')
n = ZhNormalizer(overwrite_cache=True, cache_dir=cache_dir)
print('FST cache written to:', cache_dir)
ls = os.listdir(cache_dir)
print('Files:', ls)
"
# 期望：输出 zh_tn_tagger.fst / zh_tn_verbalizer.fst，且路径在 ~/.cache/ 下

# 5. 启动服务并观察日志
bash run_server.sh ../weight/CosyVoice2-0.5B 2 50000
# 期望：无 Write failed / Permission denied 报错，Worker 正常加载模型

# 6. 健康检查
python3 client.py --mode health
# 期望：返回 {"status":"ok","workers":2,...}
```

---

### 最小化改造路径（紧急修复）

如果时间紧迫，**只做第一层（Dockerfile 改造）** 就能解决当前报错，因为：

- Step A 预构建了 FST 缓存 → 运行时 tn 包目录中已有缓存文件
- Step C chown 了 site-packages → 即使 `overwrite_cache=True` 仍尝试写入 site-packages，但 HwHiAiUser 现在有写权限
- Step B 预打了 modelscope 补丁 → 运行时不再需要 `sed -i`

**但此方案有隐患**：`overwrite_cache=True` 仍然会在每次启动时重写 site-packages 中的 FST 文件，依赖 chown 赋予的写权限。一旦客户环境对 site-packages 有只读保护（如安全策略强制挂载为只读），仍会失败。

**推荐做法**：紧急修复用第一层快速上线，后续补上第二层（改 cache_dir）彻底根治。三层全部完成后，服务不再依赖任何 site-packages 写权限，完全符合"去 root"要求。

---

### 补充说明

**torchair 缓存**：`torch.compile` 会在当前工作目录下创建 `.torchair_cache/` 目录。如果 CWD 是 root 所有的目录，非 root 用户无法写入。确保 WORKDIR 设置在用户 home 下（如 `/home/HwHiAiUser/CosyVoice`），或在 `run_server.sh` 中添加：

```bash
export TORCHAIR_CACHE_DIR=~/.cache/torchair
```

**modelscope 模型下载缓存**：`snapshot_download()` 默认写入 `~/.cache/modelscope/`，属于用户 home 目录，非 root 用户可写，无需改造。

**临时文件**：服务上传音频时使用 `tempfile.gettempdir()`（通常为 `/tmp`），Linux 默认 `/tmp` 有 sticky bit 且所有用户可写，无需改造。但如果客户环境将 `/tmp` 设为只读，需设置 `export TMPDIR=~/.tmp` 并预先创建该目录。
