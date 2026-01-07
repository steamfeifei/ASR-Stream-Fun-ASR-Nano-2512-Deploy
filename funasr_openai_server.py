# -*- encoding: utf-8 -*-
"""
FunASR with OpenAI Realtime API Protocol
========================================
功能: 使用本地 FunASR 模型进行语音识别，但对外暴露兼容 OpenAI Realtime API 的 WebSocket 接口。
目的: 让支持 OpenAI Realtime API 的客户端可以直接连接本服务，使用本地 FunASR 模型进行识别。

协议说明 (OpenAI Realtime Beta):
  - 客户端发送:
    {
      "type": "input_audio_buffer.append",
      "audio": "base64_encoded_pcm16"
    }
  - 服务端返回:
    {
      "type": "response.audio_transcript.delta",
      "delta": "..."
    }
    {
      "type": "response.audio_transcript.done",
      "transcript": "..."
    }

依赖:
  pip install websockets numpy torch funasr modelscope
"""

import asyncio
import json
import websockets
import time
import logging
import argparse
import ssl
import os
import base64
import uuid
import numpy as np
import torch
import traceback
from concurrent.futures import ThreadPoolExecutor

# FunASR 依赖
from funasr import AutoModel
# 这是一个折衷方案，防止某些版本的 FunASR 报错
try:
    from funasr.models.fun_asr_nano.model import FunASRNano
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FunASR-OpenAI")

# 全局线程池
inference_executor = ThreadPoolExecutor(max_workers=10)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听 IP")
    parser.add_argument("--port", type=int, default=10095, help="服务端口")
    parser.add_argument("--asr_model", type=str, default="FunAudioLLM/Fun-ASR-Nano-2512", help="离线 ASR 模型")
    parser.add_argument("--asr_model_online", type=str, default="FunAudioLLM/Fun-ASR-Nano-2512", help="流式 ASR 模型")
    parser.add_argument("--vad_model", type=str, default="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", help="VAD 模型")
    parser.add_argument("--punc_model", type=str, default="iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727", help="标点模型")
    parser.add_argument("--ngpu", type=int, default=1, help="GPU 数量")
    parser.add_argument("--device", type=str, default="cuda", help="运行设备")
    parser.add_argument("--ncpu", type=int, default=4, help="CPU 核心数")
    parser.add_argument("--certfile", type=str, default="", help="SSL 证书")
    parser.add_argument("--keyfile", type=str, default="", help="SSL 密钥")
    parser.add_argument("--fp16", action="store_true", help="使用 fp16")
    return parser.parse_args()

args = get_args()

# --- 模型加载 (复用 FunASR 逻辑) ---
print("正在加载模型...", flush=True)

# 模型 ID (会自动去 modelscope cache 找，或者下载)
MODEL_ID = 'FunAudioLLM/Fun-ASR-Nano-2512'

# 指定本地缓存目录 (与 download_model.py 保持一致)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.join(CURRENT_DIR, 'models')

# 构建本地模型路径 (确保之前 download_model.py 已执行成功)
# 结构: models/FunAudioLLM/Fun-ASR-Nano-2512
local_model_path = os.path.join(MODELS_ROOT, MODEL_ID)

# 1. ASR 模型 (共用实例)
model_asr = AutoModel(
    model=local_model_path,
    ngpu=args.ngpu,
    ncpu=args.ncpu,
    device=args.device,
    disable_pbar=True,
    disable_log=True,
    fp16=args.fp16,
)
model_asr_streaming = model_asr

# 2. VAD 模型
model_vad = AutoModel(
    model=args.vad_model,
    ngpu=args.ngpu,
    ncpu=args.ncpu,
    device=args.device,
    disable_pbar=True,
    disable_log=True,
    fp16=args.fp16,
)

# 3. 标点模型
if args.punc_model:
    model_punc = AutoModel(
        model=args.punc_model,
        ngpu=args.ngpu,
        ncpu=args.ncpu,
        device=args.device,
        disable_pbar=True,
        disable_log=True,
        fp16=args.fp16,
    )
else:
    model_punc = None

print("模型加载完成。", flush=True)


# --- 辅助函数 ---

async def run_model_inference(model, input, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        inference_executor, 
        lambda: model.generate(input=input, **kwargs)
    )

def decode_audio_chunk(chunk_bytes):
    # PCM 16k 16bit -> Float32 Tensor
    data_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
    data_float32 = data_int16.astype(np.float32) / 32768.0
    return torch.from_numpy(data_float32)

# --- WebSocket 处理 ---

async def ws_serve(websocket, path=None):
    client_id = str(uuid.uuid4())[:8]
    logger.info(f"Client connected: {client_id}")
    
    # 状态初始化
    state = {
        "frames": [],           # 原始音频缓冲
        "frames_asr": [],      # 离线识别缓冲 (VAD片段)
        "frames_asr_online": [], # 在线流式缓冲
        
        "vad_pre_idx": 0,
        "speech_start": False,
        "is_speaking": False,   # 当前是否在说话
        "chunk_interval": 3,    # 流式推理间隔 (数值越小，实时上屏越快，但消耗算力更多)
        
        # 模型状态字典
        "status_dict_asr": {},
        "status_dict_asr_online": {"cache": {}, "is_final": False},
        "status_dict_vad": {"cache": {}, "is_final": False},
        "status_dict_punc": {"cache": {}},
        
        "response_id": str(uuid.uuid4()),
        "item_id": str(uuid.uuid4())
    }

    try:
        # 发送连接成功事件 (模拟 OpenAI)
        await websocket.send(json.dumps({
            "type": "session.created",
            "event_id": str(uuid.uuid4()),
            "session": {"id": client_id, "model": "funasr-local"}
        }))

        async for message in websocket:
            # OpenAI Protocol 主要是 JSON 文本帧
            if isinstance(message, str):
                try:
                    event = json.loads(message)
                except json.JSONDecodeError:
                    continue
                
                event_type = event.get("type")
                
                # 1. 处理音频数据: input_audio_buffer.append
                if event_type == "input_audio_buffer.append":
                    b64_audio = event.get("audio")
                    if b64_audio:
                        try:
                            audio_chunk = base64.b64decode(b64_audio)
                            await process_audio_chunk(websocket, state, audio_chunk)
                        except Exception as e:
                            logger.error(f"Audio decode error: {e}")

                # 2. 处理提交信号: input_audio_buffer.commit (强制断句)
                elif event_type == "input_audio_buffer.commit":
                    logger.info(f"Client {client_id} sent commit signal.")
                    # 处理剩余流式音频
                    if state["frames_asr_online"]:
                         audio_in = b"".join(state["frames_asr_online"])
                         text = await run_online_asr(websocket, state, audio_in)
                         if text:
                             await send_openai_delta(websocket, state, text)
                         state["frames_asr_online"] = []
                         
                    # 强制认为语音结束，触发离线识别
                    if state["is_speaking"] or len(state["frames_asr"]) > 0:
                        state["is_speaking"] = False
                        state["speech_start"] = False
                        await run_offline_asr(websocket, state)

                # 3. 处理配置更新: session.update (可选)
                elif event_type == "session.update":
                    # 可以解析 event["session"] 中的配置，如 vad 阈值等
                    # 这里暂时忽略，使用默认值
                    await websocket.send(json.dumps({
                        "type": "session.updated",
                        "event_id": str(uuid.uuid4()),
                        "session": event.get("session")
                    }))
            
            # 兼容：如果客户端直接发二进制流 (非 OpenAI 标准，但有些混合客户端可能这样做)
            elif isinstance(message, bytes):
                 await process_audio_chunk(websocket, state, message)

    except websockets.ConnectionClosed:
        logger.info(f"Client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"Error: {e}")
        traceback.print_exc()

async def process_audio_chunk(websocket, state, chunk):
    # 因为很多时候 VAD 需要一定的静音才判定结束，直接断开可能漏掉最后一句
    if state["speech_start"] and len(state["frames_asr"]) > 0:
        await run_offline_asr(websocket, state)

async def process_audio_chunk(websocket, state, chunk):
    """处理单个音频块：VAD -> 流式 ASR -> 离线 ASR"""
    
    state["frames"].append(chunk)
    state["frames_asr_online"].append(chunk)
    
    duration_ms = len(chunk) // 32
    state["vad_pre_idx"] += duration_ms

    # --- 1. 流式识别 (Online ASR) ---
    # 每隔一定间隔进行一次推理
    # 修正逻辑: 必须让 model_asr_streaming 看到连续的上下文，不能轻易清空状态，但音频可以分块送入
    if len(state["frames_asr_online"]) >= state["chunk_interval"]:
        audio_in = b"".join(state["frames_asr_online"])
        text = await run_online_asr(websocket, state, audio_in)
        if text:
            # 发送增量结果 (Delta)
            await send_openai_delta(websocket, state, text)
        state["frames_asr_online"] = [] # 清空当前缓冲，因为已送入模型且状态保存在 status_dict_asr_online 中

    # --- 2. VAD 检测 ---
    if state["speech_start"]:
        state["frames_asr"].append(chunk)
    
    # 修复：传递 chunk 而不是 audio_in (audio_in 可能为空)
    # 此处逻辑保持原样，因为 chunk 是本函数的参数
    speech_start_i, speech_end_i = await run_vad(websocket, state, chunk)
    
    # 检测到语音开始
    if speech_start_i != -1 and not state["speech_start"]:
        state["speech_start"] = True
        state["is_speaking"] = True
        # 回溯
        beg_bias = (state["vad_pre_idx"] - speech_start_i) // duration_ms
        state["frames_asr"] = state["frames"][-beg_bias:] if beg_bias > 0 else []
        # 发送语音开始事件
        await websocket.send(json.dumps({
             "type": "input_audio_buffer.speech_started",
             "audio_start_ms": speech_start_i,
             "item_id": state["item_id"]
        }))

    # 检测到语音结束
    if speech_end_i != -1:
        if state["speech_start"]:
            state["speech_start"] = False
            state["is_speaking"] = False
            # 发送语音结束事件
            await websocket.send(json.dumps({
                 "type": "input_audio_buffer.speech_stopped",
                 "audio_end_ms": speech_end_i,
                 "item_id": state["item_id"]
            }))
            # 触发离线识别
            await run_offline_asr(websocket, state)

async def run_online_asr(websocket, state, audio_in):
    """执行在线流式识别"""
    if len(audio_in) == 0: return None
    
    audio_tensor = decode_audio_chunk(audio_in)
    res_list = await run_model_inference(
        model_asr_streaming, 
        input=[audio_tensor], 
        **state["status_dict_asr_online"]
    )
    
    if res_list and len(res_list) > 0:
        text = res_list[0].get("text", "")
        return text
    return None

async def run_offline_asr(websocket, state):
    """执行离线高精度识别 (语音片段结束时)"""
    audio_data = b"".join(state["frames_asr"])
    if len(audio_data) == 0: return

    # 1. ASR
    audio_tensor = decode_audio_chunk(audio_data)
    res_list = await run_model_inference(
        model_asr, 
        input=[audio_tensor], 
        **state["status_dict_asr"]
    )
    text = res_list[0].get("text", "") if res_list else ""

    # 2. Punctuation
    if model_punc and text:
        punc_list = await run_model_inference(
            model_punc, 
            input=text, 
            **state["status_dict_punc"]
        )
        if punc_list:
            text = punc_list[0].get("text", text)

    # 3. 发送 Final 结果
    await send_openai_done(websocket, state, text)
    
    # 4. 重置状态
    state["frames_asr"] = []
    state["frames_asr_online"] = []
    state["status_dict_asr_online"]["cache"] = {}
    state["status_dict_punc"]["cache"] = {}
    # 更新 ID 为下一句话做准备
    state["response_id"] = str(uuid.uuid4())
    state["item_id"] = str(uuid.uuid4())

async def run_vad(websocket, state, audio_in):
    """执行 VAD 检测"""
    audio_tensor = decode_audio_chunk(audio_in)
    
    # 计算 chunk_size (如果还没有的话)
    if "chunk_size" not in state["status_dict_vad"] and "chunk_size" in state["status_dict_asr_online"]:
         # 尝试从 ASR 状态推断，或者使用默认值
         pass

    res_list = await run_model_inference(
        model_vad, 
        input=[audio_tensor], 
        **state["status_dict_vad"]
    )
    
    if not res_list: return -1, -1
    
    segments = res_list[0].get("value", [])
    if not segments: return -1, -1
    
    start, end = -1, -1
    if len(segments) > 0:
        if segments[0][0] != -1: start = segments[0][0]
        if segments[0][1] != -1: end = segments[0][1]
        
    return start, end

async def send_openai_delta(websocket, state, text):
    """发送增量文本 (流式)"""
    # 注意: OpenAI Delta 是增量字符，但 FunASR 往往返回当前句子的全量文本
    # 这里我们需要做一个简单的 diff 或者直接发送覆盖更新？
    # OpenAI 协议期望的是 delta。
    # 为了简化，我们假设 FunASR 每次返回的是当前 buffer 的识别结果。
    # 真正的 Delta 计算比较复杂，这里我们采用 "覆盖式" 的思路不太行，OpenAI 客户端通常是 append。
    # 更好的做法是记录上一次发送的长度。
    
    last_len = state.get("last_sent_len", 0)
    if len(text) > last_len:
        delta_text = text[last_len:]
        state["last_sent_len"] = len(text)
        
        await websocket.send(json.dumps({
            "type": "response.audio_transcript.delta",
            "response_id": state["response_id"],
            "item_id": state["item_id"],
            "output_index": 0,
            "content_index": 0,
            "delta": delta_text
        }, ensure_ascii=False))

async def send_openai_done(websocket, state, text):
    """发送最终结果"""
    # 发送 Transcript Done
    await websocket.send(json.dumps({
        "type": "response.audio_transcript.done",
        "response_id": state["response_id"],
        "item_id": state["item_id"],
        "output_index": 0,
        "content_index": 0,
        "transcript": text
    }, ensure_ascii=False))
    
    # 发送 Response Done (表示这一轮交互结束)
    await websocket.send(json.dumps({
        "type": "response.done",
        "response_id": state["response_id"],
        "status": "completed"
    }))
    
    # 重置发送计数
    state["last_sent_len"] = 0

async def main():
    if args.certfile:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(args.certfile, keyfile=args.keyfile)
    else:
        ssl_context = None

    server = await websockets.serve(
        ws_serve, 
        args.host, 
        args.port, 
        ssl=ssl_context, 
        ping_interval=None
    )
    logger.info(f"FunASR (OpenAI Protocol) Server started on ws://{args.host}:{args.port}")
    await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())

