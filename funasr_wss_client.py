# -*- encoding: utf-8 -*-
import os
import time
import websockets
"""
FunASR-Nano-2512 WebSocket Client
作者：凌封
来源：https://aibook.ren (AI全书)
"""
import ssl
import asyncio
import argparse
import json
import logging
from multiprocessing import Process

logging.basicConfig(level=logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--host", type=str, default="localhost", required=False, help="服务端 IP，例如 localhost 或 0.0.0.0"
)
parser.add_argument("--port", type=int, default=10095, required=False, help="服务端口")
parser.add_argument("--chunk_size", type=str, default="5, 10, 5", help="分块大小配置")
parser.add_argument("--encoder_chunk_look_back", type=int, default=4, help="编码器回溯步数")
parser.add_argument("--decoder_chunk_look_back", type=int, default=0, help="解码器回溯步数")
parser.add_argument("--chunk_interval", type=int, default=10, help="分块发送间隔")
parser.add_argument(
    "--hotword",
    type=str,
    default="",
    help="热词文件路径，每行一个热词 (例如: 阿里巴巴 20)",
)
parser.add_argument("--audio_in", type=str, default=None, help="输入音频文件路径")
parser.add_argument("--audio_fs", type=int, default=16000, help="音频采样率")
parser.add_argument("--thread_num", type=int, default=1, help="并发线程数")
parser.add_argument("--words_max_print", type=int, default=10000, help="最大打印通过数")
parser.add_argument("--ssl", type=int, default=0, help="是否使用 SSL 连接: 1 为是, 0 为否") # Default 0
parser.add_argument("--mode", type=str, default="2pass", help="模式: offline, online, 2pass")

args = parser.parse_args()
args.chunk_size = [int(x) for x in args.chunk_size.split(",")]
print(args)

async def ws_client(id, chunk_begin, chunk_size):
    if args.audio_in is None:
        chunk_begin = 0
        chunk_size = 1

    # URI 连接配置
    if args.ssl == 1:
        ssl_context = ssl.SSLContext()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        uri = "wss://{}:{}".format(args.host, args.port)
    else:
        uri = "ws://{}:{}".format(args.host, args.port)
        ssl_context = None

    print(f"正在连接到 {uri}...")
    
    async with websockets.connect(
        uri, subprotocols=["binary"], ping_interval=None, ssl=ssl_context
    ) as websocket:
        
        # 音频源处理
        wavs = []
        if args.audio_in is not None:
             if args.audio_in.endswith(".scp"):
                f_scp = open(args.audio_in)
                wavs = f_scp.readlines()
             else:
                wavs = [args.audio_in]
        
        # 如果没有指定音频输入，直接退出（此处不实现麦克风输入）
        if not wavs:
            print("未指定音频输入。请使用 --audio_in 参数")
            return

        for wav in wavs:
            wav_path = wav.strip()
            # 简单的 scp 文件解析
            if len(wav.split()) > 1:
                 wav_path = wav.split()[1]
            
            print(f"正在处理: {wav_path}")
            
            # 读取音频文件
            sample_rate = args.audio_fs
            if wav_path.endswith(".pcm"):
                 with open(wav_path, "rb") as f:
                    audio_bytes = f.read()
            elif wav_path.endswith(".wav"):
                import wave
                with wave.open(wav_path, "rb") as wav_file:
                    sample_rate = wav_file.getframerate()
                    audio_bytes = wav_file.readframes(wav_file.getnframes())
            else:
                # 暂不支持的格式
                continue

            # 发送初始配置消息
            message = json.dumps(
                {
                    "mode": args.mode,
                    "chunk_size": args.chunk_size,
                    "chunk_interval": args.chunk_interval,
                    "encoder_chunk_look_back": args.encoder_chunk_look_back,
                    "decoder_chunk_look_back": args.decoder_chunk_look_back,
                    "audio_fs": sample_rate,
                    "wav_name": "test",
                    "is_speaking": True,
                    "hotwords": "", # 简化热词处理
                    "itn": True,
                }
            )
            await websocket.send(message)

            # 模拟流式发送
            # 计算步长，例如 60ms (如果 interval 是 10，则每次发送一帧)
            # 官方逻辑:
            # stride = int(60 * args.chunk_size[1] / args.chunk_interval / 1000 * sample_rate * 2)
            # 默认 chunk_size[1] 是 10, interval 10 -> 60ms * 1 = 60ms
            
            stride = int(60 * args.chunk_size[1] / args.chunk_interval / 1000 * sample_rate * 2)
            chunk_num = (len(audio_bytes) - 1) // stride + 1
            
            for i in range(chunk_num):
                beg = i * stride
                data = audio_bytes[beg : beg + stride]
                await websocket.send(data)

                # 模拟实时延迟
                sleep_duration = 60 * args.chunk_size[1] / args.chunk_interval / 1000
                await asyncio.sleep(sleep_duration)
            
            # 发送结束信号
            is_speaking = False
            message = json.dumps({"is_speaking": is_speaking})
            await websocket.send(message)

            # 等待最终结果
            try:
                while True:
                    response = await websocket.recv()
                    msg = json.loads(response)
                    
                    text = msg.get("text", "")
                    mode = msg.get("mode", "")
                    is_final = msg.get("is_final", False)
                    
                    if mode == "2pass-online" or mode == "online":
                        print(f"\r[流式结果] {text}", end="")
                    elif mode == "2pass-offline" or mode == "offline":
                         print(f"\n[最终结果] {text}")
                    
                    if mode == "2pass-offline" or mode == "offline":
                        break
            except websockets.exceptions.ConnectionClosed:
                print("\n连接已被服务器关闭。")
                break
            
            # 文件间短暂暂停
            await asyncio.sleep(1)

def one_thread(id, chunk_begin, chunk_size):
    asyncio.run(ws_client(id, chunk_begin, chunk_size))

if __name__ == "__main__":
    p = Process(target=one_thread, args=(0, 0, 0))
    p.start()
    p.join()
