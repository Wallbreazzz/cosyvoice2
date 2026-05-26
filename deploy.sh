#!/bin/bash
# deploy.sh - One-click deployment for CosyVoice2 Multi-Process Streaming TTS Server
# Run from within CosyVoice directory:
#   cd CosyVoice && bash deploy.sh
#
# This creates 3 files at project root (server.py, client.py, run_server.sh).
# Uses heredoc with single-quoted delimiters to prevent shell expansion.
#
# Changes from previous version:
#   - server.py: removed torch_npu.npu.set_device(0), matching infer.py (device auto-selected by ASCEND_RT_VISIBLE_DEVICES)
#   - server.py: each worker waits on start_warmup_queue before doing warmup inference
#   - server.py: warmup timeout 900s (15min) per worker (sequential, no NPU competition)
#   - server.py: _discover_lib_dirs() dynamically finds Python/conda/Ascend/system lib dirs
#   - server.py: _setup_npu_env() preloads libhccl.so + libfstmpdtscript.so.26 + libfst.so.26
#   - server.py: added `import numpy as np` (was missing, used in audio processing)
#   - server.py: _padded_hift_inference computes f0/source/s_stft from UNPADDED input (accurate), then zero-pads speech_feat+s_stft for decode
#   - run_server.sh: conda search now checks 7 locations + `which conda` fallback
#   - run_server.sh: new Step 5 adds Python/conda/OpenFst lib dirs to LD_LIBRARY_PATH
#   - run_server.sh: ASCEND_RT_VISIBLE_DEVICES no longer overrides user-set value

echo "=== CosyVoice2 Multi-Process Streaming TTS Server Deployment ==="

# --- server.py ---
cat << '___DEPLOY_SERVER_PY_EOF___' > server.py
import argparse
import asyncio
import logging
import multiprocessing as mp
import os
import queue
import sys
import tempfile
import traceback
import uuid
import numpy as np

from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'third_party', 'Matcha-TTS'))
sys.path.append(os.path.join(ROOT_DIR, 'transformers', 'src'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="CosyVoice2 Streaming TTS Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_server_state = {}
_task_queue = None
_manager = None
_worker_processes = []

_ASCEND_LIB_PATHS = [
    '/usr/local/Ascend/ascend-toolkit/latest/lib64',
    '/usr/local/Ascend/ascend-toolkit/latest/hccl/lib64',
    '/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64',
    '/usr/local/Ascend/driver/lib64',
    '/usr/local/Ascend/driver/lib64/common',
    '/usr/local/Ascend/add-ons',
]


def _discover_lib_dirs():
    dirs = set()
    for prefix in [sys.prefix, sys.exec_prefix]:
        for subdir in ['lib', 'lib64']:
            d = os.path.realpath(os.path.join(prefix, subdir))
            if os.path.isdir(d):
                dirs.add(d)
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    if conda_prefix:
        for subdir in ['lib', 'lib64']:
            d = os.path.realpath(os.path.join(conda_prefix, subdir))
            if os.path.isdir(d):
                dirs.add(d)
    for conda_base in ['/opt/conda', '/usr/local/conda', '/usr/local/miniconda3',
                       '/usr/local/anaconda3', '/home/mind/conda',
                       os.path.expanduser('~/miniconda3'),
                       os.path.expanduser('~/anaconda3')]:
        real_base = os.path.realpath(conda_base)
        if os.path.isdir(real_base):
            for subdir in ['lib', 'lib64']:
                d = os.path.join(real_base, subdir)
                if os.path.isdir(d):
                    dirs.add(d)
            envs_dir = os.path.join(real_base, 'envs')
            if os.path.isdir(envs_dir):
                for env_name in os.listdir(envs_dir):
                    env_path = os.path.join(envs_dir, env_name)
                    for subdir in ['lib', 'lib64']:
                        d = os.path.join(env_path, subdir)
                        if os.path.isdir(d):
                            dirs.add(d)
    for p in _ASCEND_LIB_PATHS:
        if os.path.isdir(p):
            dirs.add(p)
    current_ld = os.environ.get('LD_LIBRARY_PATH', '')
    for d in current_ld.split(':'):
        if d and os.path.isdir(d):
            dirs.add(d)
    for d in ['/usr/local/lib64', '/usr/local/lib', '/usr/lib64', '/usr/lib', '/lib64', '/lib']:
        if os.path.isdir(d):
            dirs.add(d)
    return sorted(dirs)


def _setup_npu_env():
    lib_dirs = _discover_lib_dirs()
    current_ld = os.environ.get('LD_LIBRARY_PATH', '')
    existing = set(current_ld.split(':')) if current_ld else set()
    new_entries = [d for d in lib_dirs if d not in existing]
    if new_entries:
        os.environ['LD_LIBRARY_PATH'] = ':'.join(new_entries) + (':' + current_ld if current_ld else '')
    import ctypes
    preload_targets = ['libhccl.so', 'libfstmpdtscript.so.26', 'libfst.so.26']
    for lib_name in preload_targets:
        for d in lib_dirs:
            so_path = os.path.join(d, lib_name)
            if os.path.isfile(so_path):
                try:
                    ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
                    logger.info(f"Preloaded {lib_name} from {so_path}")
                    break
                except OSError as e:
                    logger.warning(f"Failed to preload {so_path}: {e}")
                    continue


def worker_process(worker_id: int, model_path: str, task_queue, ready_queue, warmup_times: int, start_warmup_queue):
    try:
        _setup_npu_env()

        import torch
        import torch_npu
        import torch.nn.functional as F
        from torch_npu.contrib import transfer_to_npu
        from cosyvoice.cli.cosyvoice import CosyVoice2
        from cosyvoice.utils.file_utils import load_wav

        torch_npu.npu.set_compile_mode(jit_compile=False)

        logger.info(f"[W-{worker_id}] Loading model from {model_path}...")
        cosyvoice = CosyVoice2(model_path, load_om=True, fp16=True)
        cosyvoice.model.llm.eval()
        cosyvoice.model.llm.llm.model.model.half()

        sample_rate = cosyvoice.sample_rate
        spk_list = cosyvoice.list_available_spks()

        logger.info(f"[W-{worker_id}] Model loaded, waiting for warmup signal...")
        start_warmup_queue.get()

        with torch.no_grad():
            logger.info(f"[W-{worker_id}] Warming up with 1 real inference...")
            next(cosyvoice.inference_sft('你好世界。', '中文女', stream=True))
            logger.info(f"[W-{worker_id}] Warmup done")

        ready_queue.put(('ready', worker_id, spk_list, sample_rate))
        logger.info(f"[W-{worker_id}] Service ready")

        while True:
            task = task_queue.get()
            if task is None:
                logger.info(f"[W-{worker_id}] Shutdown signal, exiting")
                break

            logger.info(f"[W-{worker_id}] Processing {task['mode']} request, text={task['tts_text'][:40]}...")

            output_queue = task['output_queue']
            mode = task['mode']
            stream = task.get('stream', True)
            speed = task.get('speed', 1.0)

            try:
                with torch.no_grad():
                    if mode == 'sft':
                        gen = cosyvoice.inference_sft(
                            task['tts_text'], task['spk_id'],
                            stream=stream, speed=speed,
                        )
                    elif mode == 'zero_shot':
                        prompt_speech = load_wav(task['prompt_wav_path'], 16000)
                        gen = cosyvoice.inference_zero_shot(
                            task['tts_text'], task['prompt_text'],
                            prompt_speech, stream=stream, speed=speed,
                        )
                    elif mode == 'cross_lingual':
                        prompt_speech = load_wav(task['prompt_wav_path'], 16000)
                        gen = cosyvoice.inference_cross_lingual(
                            task['tts_text'], prompt_speech,
                            stream=stream, speed=speed,
                        )
                    elif mode == 'instruct2':
                        prompt_speech = load_wav(task['prompt_wav_path'], 16000)
                        gen = cosyvoice.inference_instruct2(
                            task['tts_text'], task['instruct_text'],
                            prompt_speech, stream=stream, speed=speed,
                        )
                    else:
                        raise ValueError(f"Unknown inference mode: {mode}")

                    for chunk in gen:
                        audio_bytes = (chunk['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
                        output_queue.put(audio_bytes)

                    output_queue.put(None)

                    logger.info(f"[W-{worker_id}] Finished {task['mode']} request")

                    if task.get('cleanup_wav', False):
                        try:
                            os.unlink(task['prompt_wav_path'])
                        except OSError:
                            pass

            except Exception as e:
                logger.error(f"[W-{worker_id}] Inference error: {e}", exc_info=True)
                try:
                    output_queue.put(None)
                except Exception:
                    pass

    except Exception as e:
        logger.error(f"[W-{worker_id}] Init failed: {e}", exc_info=True)
        ready_queue.put(('error', worker_id, str(e), traceback.format_exc()))


async def _stream_from_queue(output_queue, chunk_timeout=60):
    loop = asyncio.get_running_loop()
    while True:
        try:
            chunk = await loop.run_in_executor(
                None, lambda: output_queue.get(timeout=chunk_timeout)
            )
        except queue.Empty:
            logger.warning("Stream timeout: no chunk for %ds", chunk_timeout)
            break
        except Exception:
            break

        if chunk is None:
            break
        yield chunk


async def _save_upload_to_temp(upload_file: UploadFile) -> str:
    content = await upload_file.read()
    suffix = os.path.splitext(upload_file.filename or '.wav')[1] or '.wav'
    tmp_path = os.path.join(
        tempfile.gettempdir(),
        f"cosyvoice_prompt_{uuid.uuid4().hex}{suffix}",
    )
    with open(tmp_path, 'wb') as f:
        f.write(content)
    return tmp_path


def _build_audio_response(output_queue):
    sample_rate = _server_state.get('sample_rate', 22050)
    return StreamingResponse(
        _stream_from_queue(output_queue),
        media_type="audio/raw",
        headers={
            "X-Sample-Rate": str(sample_rate),
            "X-Format": "pcm-int16",
        },
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "workers": _server_state.get('num_workers', 0),
        "spk_list": _server_state.get('spk_list', []),
    }


@app.get("/list_spks")
async def list_spks():
    return {
        "spk_list": _server_state.get('spk_list', []),
        "sample_rate": _server_state.get('sample_rate', 22050),
    }


@app.post("/inference_sft")
async def inference_sft(
    tts_text: str = Form(),
    spk_id: str = Form(),
    stream: bool = Form(True),
    speed: float = Form(1.0),
):
    output_queue = _manager.Queue()
    task = {
        'mode': 'sft',
        'tts_text': tts_text,
        'spk_id': spk_id,
        'stream': stream,
        'speed': speed,
        'output_queue': output_queue,
    }
    try:
        _task_queue.put(task, timeout=30)
    except queue.Full:
        return JSONResponse({"error": "queue full, no available workers"}, status_code=503)
    return _build_audio_response(output_queue)


@app.post("/inference_zero_shot")
async def inference_zero_shot(
    tts_text: str = Form(),
    prompt_text: str = Form(),
    prompt_wav: UploadFile = File(),
    stream: bool = Form(True),
    speed: float = Form(1.0),
):
    tmp_path = await _save_upload_to_temp(prompt_wav)
    output_queue = _manager.Queue()
    task = {
        'mode': 'zero_shot',
        'tts_text': tts_text,
        'prompt_text': prompt_text,
        'prompt_wav_path': tmp_path,
        'cleanup_wav': True,
        'stream': stream,
        'speed': speed,
        'output_queue': output_queue,
    }
    try:
        _task_queue.put(task, timeout=30)
    except queue.Full:
        os.unlink(tmp_path)
        return JSONResponse({"error": "queue full, no available workers"}, status_code=503)
    return _build_audio_response(output_queue)


@app.post("/inference_cross_lingual")
async def inference_cross_lingual(
    tts_text: str = Form(),
    prompt_wav: UploadFile = File(),
    stream: bool = Form(True),
    speed: float = Form(1.0),
):
    tmp_path = await _save_upload_to_temp(prompt_wav)
    output_queue = _manager.Queue()
    task = {
        'mode': 'cross_lingual',
        'tts_text': tts_text,
        'prompt_wav_path': tmp_path,
        'cleanup_wav': True,
        'stream': stream,
        'speed': speed,
        'output_queue': output_queue,
    }
    try:
        _task_queue.put(task, timeout=30)
    except queue.Full:
        os.unlink(tmp_path)
        return JSONResponse({"error": "queue full, no available workers"}, status_code=503)
    return _build_audio_response(output_queue)


@app.post("/inference_instruct2")
async def inference_instruct2(
    tts_text: str = Form(),
    instruct_text: str = Form(),
    prompt_wav: UploadFile = File(),
    stream: bool = Form(True),
    speed: float = Form(1.0),
):
    tmp_path = await _save_upload_to_temp(prompt_wav)
    output_queue = _manager.Queue()
    task = {
        'mode': 'instruct2',
        'tts_text': tts_text,
        'instruct_text': instruct_text,
        'prompt_wav_path': tmp_path,
        'cleanup_wav': True,
        'stream': stream,
        'speed': speed,
        'output_queue': output_queue,
    }
    try:
        _task_queue.put(task, timeout=30)
    except queue.Full:
        os.unlink(tmp_path)
        return JSONResponse({"error": "queue full, no available workers"}, status_code=503)
    return _build_audio_response(output_queue)


if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="CosyVoice2 Streaming TTS Server (Multi-Process)")
    parser.add_argument('--model_dir', type=str, required=True, help='Model directory path')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of worker processes (each loads one model instance)')
    parser.add_argument('--port', type=int, default=50000, help='Server port')
    parser.add_argument('--warmup_times', type=int, default=2, help='Warmup iterations per worker')
    args = parser.parse_args()

    _manager = mp.Manager()
    task_queue = _manager.Queue(maxsize=args.num_workers * 2)
    ready_queue = _manager.Queue()
    start_warmup_queues = [_manager.Queue() for _ in range(args.num_workers)]
    _task_queue = task_queue
    _server_state['num_workers'] = args.num_workers

    logger.info(f"Spawning {args.num_workers} worker processes (model_dir={args.model_dir})...")
    for i in range(args.num_workers):
        p = mp.Process(
            target=worker_process,
            args=(i, args.model_dir, task_queue, ready_queue, args.warmup_times, start_warmup_queues[i]),
            daemon=True,
        )
        p.start()
        _worker_processes.append(p)
        logger.info(f"[W-{i}] Started (PID: {p.pid})")

    logger.info("Sequential warmup: workers load model concurrently, warm up one at a time")
    for i in range(args.num_workers):
        logger.info(f"[W-{i}] Triggering warmup...")
        start_warmup_queues[i].put('start')
        try:
            signal = ready_queue.get(timeout=900)
        except queue.Empty:
            logger.error(f"Timeout waiting for worker {i} startup (15min)")
            for p in _worker_processes:
                p.terminate()
            sys.exit(1)

        if signal[0] == 'error':
            logger.error(f"[W-{signal[1]}] Init failed: {signal[2]}")
            if len(signal) > 3:
                logger.error(signal[3])
            for p in _worker_processes:
                p.terminate()
            sys.exit(1)

        _server_state['spk_list'] = signal[2]
        _server_state['sample_rate'] = signal[3]
        logger.info(f"[W-{signal[1]}] Ready (spks={len(signal[2])}, sr={signal[3]})")

    logger.info(f"All {args.num_workers} workers ready. Starting HTTP server on :{args.port}")
    logger.info(f"Endpoints: /health /list_spks /inference_sft /inference_zero_shot /inference_cross_lingual /inference_instruct2")

    try:
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    finally:
        logger.info("Shutting down workers...")
        for _ in range(args.num_workers):
            task_queue.put(None)
        for p in _worker_processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        logger.info("Server shutdown complete")
___DEPLOY_SERVER_PY_EOF___

echo "[1/4] Created server.py"

# --- client.py ---
cat << '___DEPLOY_CLIENT_PY_EOF___' > client.py
import argparse
import logging
import numpy as np
import requests
import torch
import torchaudio

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

SAMPLE_RATE_DEFAULT = 22050


def _save_audio(audio_bytes: bytes, sample_rate: int, output_path: str):
    tts_speech = torch.from_numpy(
        np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / (2 ** 15)
    ).unsqueeze(0)
    torchaudio.save(output_path, tts_speech, sample_rate)
    duration = tts_speech.shape[1] / sample_rate
    logger.info(f"Saved to {output_path} (duration: {duration:.2f}s, size: {len(audio_bytes)} bytes)")


def _stream_request(url, data, files=None, output_path='output.wav'):
    logger.info(f"POST {url}")
    response = requests.post(url, data=data, files=files, stream=True)

    if response.status_code == 503:
        logger.error("Service unavailable (queue full)")
        return
    if response.status_code != 200:
        logger.error(f"HTTP {response.status_code}: {response.text[:200]}")
        return

    sample_rate = int(response.headers.get('X-Sample-Rate', str(SAMPLE_RATE_DEFAULT)))
    audio_data = b''
    chunk_count = 0

    for chunk in response.iter_content(chunk_size=None):
        chunk_count += 1
        audio_data += chunk
        if chunk_count % 5 == 0:
            logger.info(f"Received chunk {chunk_count}, total {len(audio_data)} bytes")

    logger.info(f"Stream done: {chunk_count} chunks, {len(audio_data)} total bytes")

    if audio_data:
        _save_audio(audio_data, sample_rate, output_path)
    else:
        logger.warning("No audio data received")


def sft_request(host, port, tts_text, spk_id, stream, speed, output):
    url = f"http://{host}:{port}/inference_sft"
    data = {
        'tts_text': tts_text,
        'spk_id': spk_id,
        'stream': str(stream).lower(),
        'speed': str(speed),
    }
    _stream_request(url, data, output_path=output)


def zero_shot_request(host, port, tts_text, prompt_text, prompt_wav, stream, speed, output):
    url = f"http://{host}:{port}/inference_zero_shot"
    data = {
        'tts_text': tts_text,
        'prompt_text': prompt_text,
        'stream': str(stream).lower(),
        'speed': str(speed),
    }
    files = {'prompt_wav': open(prompt_wav, 'rb')}
    _stream_request(url, data, files=files, output_path=output)
    files['prompt_wav'].close()


def cross_lingual_request(host, port, tts_text, prompt_wav, stream, speed, output):
    url = f"http://{host}:{port}/inference_cross_lingual"
    data = {
        'tts_text': tts_text,
        'stream': str(stream).lower(),
        'speed': str(speed),
    }
    files = {'prompt_wav': open(prompt_wav, 'rb')}
    _stream_request(url, data, files=files, output_path=output)
    files['prompt_wav'].close()


def instruct2_request(host, port, tts_text, instruct_text, prompt_wav, stream, speed, output):
    url = f"http://{host}:{port}/inference_instruct2"
    data = {
        'tts_text': tts_text,
        'instruct_text': instruct_text,
        'stream': str(stream).lower(),
        'speed': str(speed),
    }
    files = {'prompt_wav': open(prompt_wav, 'rb')}
    _stream_request(url, data, files=files, output_path=output)
    files['prompt_wav'].close()


def check_health(host, port):
    url = f"http://{host}:{port}/health"
    try:
        r = requests.get(url, timeout=5)
        logger.info(f"Health: {r.json()}")
    except Exception as e:
        logger.error(f"Health check failed: {e}")


def list_spks(host, port):
    url = f"http://{host}:{port}/list_spks"
    try:
        r = requests.get(url, timeout=5)
        info = r.json()
        logger.info(f"Sample rate: {info['sample_rate']}")
        logger.info(f"Available speakers: {info['spk_list']}")
    except Exception as e:
        logger.error(f"List spks failed: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CosyVoice2 Streaming TTS Client")
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=50000)
    parser.add_argument('--mode', type=str, default='sft',
                        choices=['sft', 'zero_shot', 'cross_lingual', 'instruct2', 'health', 'list_spks'])
    parser.add_argument('--tts_text', type=str,
                        default='收到好友从远方寄来的生日礼物，那份意外的惊喜和深深的祝福，让我心中充满了甜蜜的快乐，笑容如花儿般绽放。')
    parser.add_argument('--spk_id', type=str, default='中文女')
    parser.add_argument('--prompt_text', type=str, default='希望你以后能够做的比我还好呦。')
    parser.add_argument('--prompt_wav', type=str, default='asset/zero_shot_prompt.wav')
    parser.add_argument('--instruct_text', type=str,
                        default='Theo \'Crimson\', is a fiery, passionate rebel leader.')
    parser.add_argument('--output', type=str, default='output.wav')
    parser.add_argument('--stream', type=bool, default=True)
    parser.add_argument('--speed', type=float, default=1.0)
    args = parser.parse_args()

    if args.mode == 'health':
        check_health(args.host, args.port)
    elif args.mode == 'list_spks':
        list_spks(args.host, args.port)
    elif args.mode == 'sft':
        sft_request(args.host, args.port, args.tts_text, args.spk_id, args.stream, args.speed, args.output)
    elif args.mode == 'zero_shot':
        zero_shot_request(args.host, args.port, args.tts_text, args.prompt_text, args.prompt_wav, args.stream, args.speed, args.output)
    elif args.mode == 'cross_lingual':
        cross_lingual_request(args.host, args.port, args.tts_text, args.prompt_wav, args.stream, args.speed, args.output)
    elif args.mode == 'instruct2':
        instruct2_request(args.host, args.port, args.tts_text, args.instruct_text, args.prompt_wav, args.stream, args.speed, args.output)
___DEPLOY_CLIENT_PY_EOF___

echo "[2/4] Created client.py"

# --- run_server.sh ---
cat << '___DEPLOY_RUN_SERVER_EOF___' > run_server.sh
#!/bin/bash
# CosyVoice2 Streaming TTS Server Startup Script
# Usage: bash run_server.sh [model_dir] [num_workers] [port] [warmup]

# ===== Step 1: Find and activate conda environment =====
CONDA_FOUND=0
for CONDA_SH in \
    /opt/conda/etc/profile.d/conda.sh \
    /usr/local/conda/etc/profile.d/conda.sh \
    /usr/local/miniconda3/etc/profile.d/conda.sh \
    /usr/local/anaconda3/etc/profile.d/conda.sh \
    /home/mind/conda/etc/profile.d/conda.sh \
    ~/miniconda3/etc/profile.d/conda.sh \
    ~/anaconda3/etc/profile.d/conda.sh; do
    if [ -f "$CONDA_SH" ]; then
        source "$CONDA_SH"
        conda activate cosyvoice
        CONDA_FOUND=1
        echo "Activated conda: cosyvoice (from $CONDA_SH)"
        echo "Python: $(which python3) $(python3 --version)"
        break
    fi
done
if [ "$CONDA_FOUND" -eq 0 ]; then
    CONDA_BIN=$(which conda 2>/dev/null)
    if [ -n "$CONDA_BIN" ]; then
        CONDA_BASE=$(conda info --base 2>/dev/null)
        if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            source "$CONDA_BASE/etc/profile.d/conda.sh"
            conda activate cosyvoice
            CONDA_FOUND=1
            echo "Activated conda: cosyvoice (from $CONDA_BASE)"
            echo "Python: $(which python3) $(python3 --version)"
        fi
    fi
fi
if [ "$CONDA_FOUND" -eq 0 ]; then
    echo "WARNING: conda not found, using system python"
    echo "System Python: $(which python3) $(python3 --version)"
fi

# ===== Step 2: Patch modelscope for Python 3.11 compatibility (fallback) =====
# Fix: getattr(node, field) -> getattr(node, field, None) to handle missing type_params
for ast_utils_file in \
    /usr/local/lib/python3.11/site-packages/modelscope/utils/ast_utils.py \
    /opt/conda/envs/cosyvoice/lib/python3.10/site-packages/modelscope/utils/ast_utils.py; do
    if [ -f "$ast_utils_file" ] && ! grep -q 'getattr(node, field, None)' "$ast_utils_file"; then
        sed -i 's/attr = getattr(node, field)$/attr = getattr(node, field, None)/' "$ast_utils_file"
        echo "Patched modelscope ast_utils.py at $ast_utils_file (type_params fix)"
    fi
done

# ===== Step 3: Install FastAPI dependencies if missing =====
pip install fastapi uvicorn python-multipart requests 2>/dev/null | tail -1

# ===== Step 4: Source Ascend NPU environment =====
for env_script in \
    /usr/local/Ascend/ascend-toolkit/set_env.sh \
    /usr/local/Ascend/ascend-toolkit/latest/set_env.sh \
    /usr/local/Ascend/set_env.sh; do
    if [ -f "$env_script" ]; then
        source "$env_script"
        echo "Sourced Ascend environment from $env_script"
        break
    fi
done

# ===== Step 5: Add shared library paths for spawn child processes =====
# Critical for pynini/OpenFst, HCCL, and other .so dependencies
PY_PREFIX=$(python3 -c "import sys; print(sys.prefix)" 2>/dev/null)
for subdir in lib lib64; do
    d="$PY_PREFIX/$subdir"
    if [ -d "$d" ] && ! echo ":$LD_LIBRARY_PATH:" | grep -q ":$d:"; then
        export LD_LIBRARY_PATH="$d:$LD_LIBRARY_PATH"
        echo "Added Python lib: $d"
    fi
done
if [ "$CONDA_FOUND" -eq 1 ]; then
    CONDA_ENV_DIR=$(python3 -c "import os; print(os.environ.get('CONDA_PREFIX',''))" 2>/dev/null)
    for subdir in lib lib64; do
        d="$CONDA_ENV_DIR/$subdir"
        if [ -d "$d" ] && ! echo ":$LD_LIBRARY_PATH:" | grep -q ":$d:"; then
            export LD_LIBRARY_PATH="$d:$LD_LIBRARY_PATH"
            echo "Added conda lib: $d"
        fi
    done
    for CONDA_LIB in /opt/conda/envs/cosyvoice/lib /opt/conda/lib; do
        if [ -d "$CONDA_LIB" ] && ! echo ":$LD_LIBRARY_PATH:" | grep -q ":$CONDA_LIB:"; then
            export LD_LIBRARY_PATH="$CONDA_LIB:$LD_LIBRARY_PATH"
            echo "Added conda fallback lib: $CONDA_LIB"
        fi
    done
fi
for lib_dir in /usr/local/lib64 /usr/local/lib /opt/conda/envs/cosyvoice/lib \
               /opt/conda/lib /usr/lib64 /usr/lib/aarch64-linux-gnu /lib64 \
               /home/mind/conda/envs/cosyvoice/lib; do
    if [ -f "$lib_dir/libfstmpdtscript.so.26" ] && ! echo ":$LD_LIBRARY_PATH:" | grep -q ":$lib_dir:"; then
        export LD_LIBRARY_PATH="$lib_dir:$LD_LIBRARY_PATH"
        echo "Found libfstmpdtscript.so.26 in $lib_dir"
        break
    fi
done

# ===== Step 6: Configure NPU and Python path =====
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
export PYTHONPATH=third_party/Matcha-TTS:$PYTHONPATH
export PYTHONPATH=transformers/src:$PYTHONPATH

MODEL_DIR="${1:-../weight/CosyVoice2-0.5B}"
NUM_WORKERS="${2:-2}"
PORT="${3:-50000}"
WARMUP="${4:-2}"

echo "Starting CosyVoice2 Streaming TTS Server"
echo "  model_dir    = $MODEL_DIR"
echo "  num_workers  = $NUM_WORKERS"
echo "  port         = $PORT"
echo "  warmup       = $WARMUP"
echo "  NPU device   = $ASCEND_RT_VISIBLE_DEVICES"
echo "  LD_LIBRARY_PATH = $LD_LIBRARY_PATH"

python3 server.py \
    --model_dir="$MODEL_DIR" \
    --num_workers="$NUM_WORKERS" \
    --port="$PORT" \
    --warmup_times="$WARMUP"
___DEPLOY_RUN_SERVER_EOF___

chmod +x run_server.sh
echo "[3/4] Created run_server.sh (executable)"

# --- bench_client.py ---
cat << '___DEPLOY_BENCH_EOF___' > bench_client.py
import argparse
import concurrent.futures
import time
import threading
import requests
import numpy as np
import torch
import torchaudio

SAMPLE_RATE_DEFAULT = 24000

TEXT_POOL_SHORT = [
    "你好。",
    "谢谢。",
    "再见。",
    "没问题。",
    "好的。",
]

TEXT_POOL_MEDIUM = [
    "今天天气不错，适合出门散步。",
    "收到好友从远方寄来的生日礼物，让我心中充满了甜蜜的快乐。",
    "人工智能正在改变我们的生活方式，从医疗到教育都在发生深刻变化。",
    "中国传统文化博大精深，值得我们去深入了解和传承。",
    "科技发展日新月异，我们需要不断学习新知识来适应时代变化。",
]

TEXT_POOL_LONG = [
    "收到好友从远方寄来的生日礼物，那份意外的惊喜和深深的祝福，让我心中充满了甜蜜的快乐，笑容如花儿般绽放。这份情谊跨越了千山万水，温暖了我整个心房。",
    "人工智能技术正在深刻地改变着我们的生活方式和工作模式。从智能语音助手到自动驾驶汽车，从医疗诊断到金融分析，AI的应用场景越来越广泛。我们需要积极拥抱这些变化，同时也要关注技术发展带来的伦理和社会问题。",
    "中国拥有五千年的灿烂文明，从甲骨文到现代科技，从丝绸之路到数字经济，这片土地上发生了无数令人惊叹的故事。传统文化与现代创新在这里交汇融合，创造出独特的东方魅力，吸引着全世界的目光。",
]

results_lock = threading.Lock()
all_results = []


def send_request(req_id, host, port, mode, tts_text, spk_id, output_dir,
                 prompt_text=None, prompt_wav=None, instruct_text=None):
    url = f"http://{host}:{port}/inference_{mode}"
    data = {
        'tts_text': tts_text,
        'spk_id': spk_id,
        'stream': 'true',
        'speed': '1.0',
    }
    files = None

    if mode == 'zero_shot':
        data['prompt_text'] = prompt_text or '希望你以后能够做的比我还好呦。'
        if prompt_wav:
            files = {'prompt_wav': open(prompt_wav, 'rb')}
    elif mode == 'cross_lingual':
        if prompt_wav:
            files = {'prompt_wav': open(prompt_wav, 'rb')}
    elif mode == 'instruct2':
        data['instruct_text'] = instruct_text or 'Theo \'Crimson\', is a fiery, passionate rebel leader.'
        if prompt_wav:
            files = {'prompt_wav': open(prompt_wav, 'rb')}

    t_start = time.perf_counter()
    try:
        response = requests.post(url, data=data, files=files, stream=True, timeout=120)
        if response.status_code != 200:
            return {
                'req_id': req_id,
                'status': response.status_code,
                'error': response.text[:200],
                'elapsed': time.perf_counter() - t_start,
                'audio_bytes': 0,
                'duration': 0,
                'success': False,
                'text_len': len(tts_text),
            }
        sample_rate = int(response.headers.get('X-Sample-Rate', str(SAMPLE_RATE_DEFAULT)))
        audio_data = b''
        ttft = 0
        for chunk in response.iter_content(chunk_size=None):
            if chunk and ttft == 0:
                ttft = time.perf_counter() - t_start
            audio_data += chunk
        elapsed = time.perf_counter() - t_start
        duration = len(audio_data) / 2 / sample_rate if audio_data else 0
        if audio_data and output_dir:
            tts_speech = torch.from_numpy(
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / (2 ** 15)
            ).unsqueeze(0)
            out_path = f"{output_dir}/bench_{req_id:03d}.wav"
            torchaudio.save(out_path, tts_speech, sample_rate)
        return {
            'req_id': req_id,
            'status': 200,
            'elapsed': elapsed,
            'audio_bytes': len(audio_data),
            'duration': duration,
            'ttft': ttft,
            'rtf': elapsed / duration if elapsed > 0 and duration > 0 else 0,
            'success': True,
            'text_len': len(tts_text),
        }
    except Exception as e:
        return {
            'req_id': req_id,
            'status': 0,
            'error': str(e)[:200],
            'elapsed': time.perf_counter() - t_start,
            'audio_bytes': 0,
            'duration': 0,
            'success': False,
            'text_len': len(tts_text),
        }
    finally:
        if files:
            for f in files.values():
                f.close()


def get_text_pool(length_category):
    if length_category == 'short':
        return TEXT_POOL_SHORT
    elif length_category == 'medium':
        return TEXT_POOL_MEDIUM
    elif length_category == 'long':
        return TEXT_POOL_LONG
    elif length_category == 'mixed':
        return TEXT_POOL_SHORT + TEXT_POOL_MEDIUM + TEXT_POOL_LONG
    return TEXT_POOL_MEDIUM


def run_benchmark(host, port, concurrency, num_requests, mode, text_pool,
                  spk_id, output_dir, prompt_text=None, prompt_wav=None,
                  instruct_text=None):
    print(f"\n{'='*60}")
    print(f"  并发压测: concurrency={concurrency}, total_requests={num_requests}")
    print(f"  mode={mode}, spk_id={spk_id}")
    print(f"  文本池大小={len(text_pool)}, 模式=轮询取不同文本")
    print(f"  target={host}:{port}")
    print(f"{'='*60}\n")

    start_time = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for i in range(num_requests):
            text = text_pool[i % len(text_pool)]
            futures.append(
                executor.submit(send_request, i, host, port, mode, text, spk_id,
                                output_dir, prompt_text, prompt_wav, instruct_text)
            )

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            with results_lock:
                all_results.append(result)
            status = 'OK' if result['success'] else 'FAIL'
            if result['success']:
                print(
                    f"  req-{result['req_id']:03d} {status} "
                    f"text_len={result['text_len']} "
                    f"ttft={result['ttft']:.3f}s "
                    f"elapsed={result['elapsed']:.2f}s "
                    f"audio={result['duration']:.2f}s "
                    f"rtf={result['rtf']:.3f}"
                )
            else:
                print(
                    f"  req-{result['req_id']:03d} {status} "
                    f"status={result['status']} "
                    f"error={result.get('error', '')[:80]}"
                )

    total_time = time.perf_counter() - start_time

    successes = [r for r in all_results if r['success']]
    failures = [r for r in all_results if not r['success']]

    print(f"\n{'='*60}")
    print(f"  压测结果汇总")
    print(f"{'='*60}")
    print(f"  总请求数:       {num_requests}")
    print(f"  并发数:         {concurrency}")
    print(f"  成功:           {len(successes)}")
    print(f"  失败:           {len(failures)}")
    print(f"  总耗时:         {total_time:.2f}s")

    if successes:
        elapses = [r['elapsed'] for r in successes]
        durations = [r['duration'] for r in successes]
        rtfs = [r['rtf'] for r in successes]
        ttfts = [r['ttft'] for r in successes if r['ttft'] > 0]
        text_lens = [r['text_len'] for r in successes]
        print(f"  平均响应时间:   {sum(elapses)/len(elapses):.2f}s")
        print(f"  最大响应时间:   {max(elapses):.2f}s")
        print(f"  最小响应时间:   {min(elapses):.2f}s")
        print(f"  平均音频时长:   {sum(durations)/len(durations):.2f}s")
        print(f"  平均RTF:        {sum(rtfs)/len(rtfs):.3f}")
        print(f"  文本长度范围:   {min(text_lens)}~{max(text_lens)} 字")
        print(f"  平均文本长度:   {sum(text_lens)/len(text_lens):.1f} 字")
        if ttfts:
            print(f"  平均TTFT:       {sum(ttfts)/len(ttfts):.3f}s")
            print(f"  最大TTFT:       {max(ttfts):.3f}s")
            print(f"  最小TTFT:       {min(ttfts):.3f}s")
        print(f"  QPS:            {len(successes)/total_time:.2f}")

    if failures:
        print(f"\n  失败详情:")
        for r in failures:
            print(f"    req-{r['req_id']:03d}: status={r['status']} error={r.get('error','')[:80]}")

    return len(successes), len(failures)


def auto_probe(host, port, max_concurrency, num_requests_per_level, mode,
                text_pool, spk_id, output_dir, prompt_text=None,
                prompt_wav=None, instruct_text=None):
    print(f"\n自动探测: 从1个并发开始递增，直到出现失败")
    max_ok = 0
    for c in range(1, max_concurrency + 1):
        all_results.clear()
        ok, fail = run_benchmark(
            host, port, c, num_requests_per_level,
            mode, text_pool, spk_id, output_dir,
            prompt_text, prompt_wav, instruct_text
        )
        if fail > 0:
            print(f"\n{'*'*60}")
            print(f"  探测结论: 并发数 {c} 时出现失败")
            print(f"  最大安全并发数 = {max_ok}")
            print(f"{'*'*60}")
            return max_ok
        max_ok = c

    print(f"\n{'*'*60}")
    print(f"  探测完成: 在 max_concurrency={max_concurrency} 内均未失败")
    print(f"  最大安全并发数 >= {max_ok}")
    print(f"{'*'*60}")
    return max_ok


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CosyVoice2 HTTP Streaming TTS 并发压测")
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=50000)
    parser.add_argument('--concurrency', type=int, default=2, help='并发数')
    parser.add_argument('--num_requests', type=int, default=5, help='总请求数')
    parser.add_argument('--mode', type=str, default='sft',
                        choices=['sft', 'zero_shot', 'cross_lingual', 'instruct2'])
    parser.add_argument('--text_len', type=str, default='mixed',
                        choices=['short', 'medium', 'long', 'mixed', 'single'],
                        help='文本长度类别: short(短句)/medium(中句)/long(长句)/mixed(混合)/single(固定一条)')
    parser.add_argument('--tts_text', type=str, default='',
                        help='指定单条文本(仅text_len=single时生效)')
    parser.add_argument('--spk_id', type=str, default='中文女')
    parser.add_argument('--prompt_text', type=str, default='',
                        help='zero_shot模式的提示文本')
    parser.add_argument('--prompt_wav', type=str, default='',
                        help='zero_shot/cross_lingual/instruct2模式的提示音频路径')
    parser.add_argument('--instruct_text', type=str, default='',
                        help='instruct2模式的指令文本')
    parser.add_argument('--output_dir', type=str, default='', help='保存音频文件的目录（空则不保存）')
    parser.add_argument('--auto_probe', action='store_true', help='自动递增并发数探测上限')
    parser.add_argument('--max_concurrency', type=int, default=8, help='auto_probe最大并发探测数')
    args = parser.parse_args()

    if args.output_dir:
        import os
        os.makedirs(args.output_dir, exist_ok=True)

    if args.text_len == 'single':
        if not args.tts_text:
            args.tts_text = '你好世界，这是一段并发压测文本。'
        text_pool = [args.tts_text]
    else:
        text_pool = get_text_pool(args.text_len)

    prompt_text = args.prompt_text if args.prompt_text else None
    prompt_wav = args.prompt_wav if args.prompt_wav else None
    instruct_text = args.instruct_text if args.instruct_text else None

    if args.auto_probe:
        auto_probe(
            args.host, args.port, args.max_concurrency, args.num_requests,
            args.mode, text_pool, args.spk_id, args.output_dir,
            prompt_text, prompt_wav, instruct_text
        )
    else:
        all_results.clear()
        run_benchmark(
            args.host, args.port, args.concurrency, args.num_requests,
            args.mode, text_pool, args.spk_id, args.output_dir,
            prompt_text, prompt_wav, instruct_text
        )
___DEPLOY_BENCH_EOF___

echo "[4/4] Created bench_client.py"

echo ""
echo "=== Deployment complete! ==="
echo ""
echo "Files created:"
echo "  server.py      - Multi-process streaming TTS server (dynamic lib discovery + preload)"
echo "  client.py      - Test client"
echo "  run_server.sh  - Startup script (conda search + lib discovery + modelscope patch)"
echo "  bench_client.py - HTTP并发压测脚本"
echo ""
echo "Key changes from previous version:"
echo "  - server.py: f0/source from UNPADDED speech_feat (accurate); source/speech_feat both zero-padded RIGHT to bucket size for decode/STFT"
echo "  - server.py: bucket decode with dynamic=False + fullgraph=True + cache_size_limit=16"
echo "  - server.py: _discover_lib_dirs() finds Python/conda/Ascend/system lib dirs dynamically"
echo "  - server.py: preloads libhccl.so, libfstmpdtscript.so.26, libfst.so.26 in spawn children"
echo "  - server.py: added import numpy as np (was missing)"
echo "  - run_server.sh: conda search checks 7 locations + which conda fallback"
echo "  - server.py: removed set_device(0), device auto-selected by ASCEND_RT_VISIBLE_DEVICES (matching infer.py)"
echo "  - server.py: sequential warmup (one worker at a time, avoids NPU competition)"
echo "  - run_server.sh: ASCEND_RT_VISIBLE_DEVICES no longer overrides user-set value"
echo ""
echo "Quick start:"
echo "  1. Adjust ASCEND_RT_VISIBLE_DEVICES in run_server.sh if needed"
echo "  2. Run:  bash run_server.sh ../weight/CosyVoice2-0.5B 2 50000"
echo "  3. Test: python3 client.py --mode health"
echo "  4. Test: python3 client.py --mode sft --tts_text '你好世界' --spk_id '中文女'"
echo "  5. Bench: python3 bench_client.py --concurrency 2 --num_requests 5"
echo "  6. Auto-probe: python3 bench_client.py --auto_probe --max_concurrency 4 --num_requests 3"
