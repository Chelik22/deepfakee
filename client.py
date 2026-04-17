"""
Webcam -> RunPod WebSocket client.

Captures webcam via ffmpeg (MJPEG), streams JPEG frames to remote deepfake
server over WebSocket, displays processed frames back via OpenCV window.

Protocol (binary WS message, both directions):
    [8 bytes frame_id, big-endian uint64]
    [8 bytes timestamp_ms, big-endian uint64]
    [JPEG payload]

Usage:
    pip install websockets opencv-python numpy
    # ensure ffmpeg on PATH
    python client.py --server wss://<runpod-host>:<port>/ws \
        --device "video=HD Pro Webcam C920" --width 640 --height 480 --fps 30
"""

import argparse
import asyncio
import struct
import subprocess
import sys
import time
from collections import deque

import cv2
import numpy as np
import websockets

SOI = b"\xff\xd8"
EOI = b"\xff\xd9"
HEADER_FMT = ">QQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def build_ffmpeg_cmd(device: str, width: int, height: int, fps: int, platform: str) -> list[str]:
    if platform == "windows":
        input_args = ["-f", "dshow", "-video_size", f"{width}x{height}",
                      "-framerate", str(fps), "-i", device]
    elif platform == "linux":
        input_args = ["-f", "v4l2", "-video_size", f"{width}x{height}",
                      "-framerate", str(fps), "-i", device]
    elif platform == "mac":
        input_args = ["-f", "avfoundation", "-video_size", f"{width}x{height}",
                      "-framerate", str(fps), "-i", device]
    else:
        raise ValueError(f"unknown platform: {platform}")

    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        *input_args,
        "-f", "mjpeg", "-q:v", "5", "-",
    ]


class JpegReader:
    """Reads concatenated JPEGs from ffmpeg stdout stream."""

    def __init__(self, stream):
        self.stream = stream
        self.buf = bytearray()

    async def next_frame(self) -> bytes | None:
        loop = asyncio.get_event_loop()
        while True:
            start = self.buf.find(SOI)
            if start != -1:
                end = self.buf.find(EOI, start + 2)
                if end != -1:
                    frame = bytes(self.buf[start:end + 2])
                    del self.buf[:end + 2]
                    return frame
            chunk = await loop.run_in_executor(None, self.stream.read, 65536)
            if not chunk:
                return None
            self.buf.extend(chunk)


async def sender(ws, reader: JpegReader, state: dict):
    frame_id = 0
    while True:
        jpeg = await reader.next_frame()
        if jpeg is None:
            await ws.close()
            return
        frame_id += 1
        ts = int(time.time() * 1000)
        header = struct.pack(HEADER_FMT, frame_id, ts)
        try:
            await ws.send(header + jpeg)
        except websockets.ConnectionClosed:
            return
        state["sent"] = frame_id


async def receiver(ws, state: dict):
    latest_id = 0
    while True:
        try:
            msg = await ws.recv()
        except websockets.ConnectionClosed:
            return
        if len(msg) < HEADER_SIZE:
            continue
        frame_id, ts = struct.unpack(HEADER_FMT, msg[:HEADER_SIZE])
        if frame_id <= latest_id:
            continue
        latest_id = frame_id
        jpeg = msg[HEADER_SIZE:]
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        rtt = int(time.time() * 1000) - ts
        state["rtt_ms"] = rtt
        state["recv"] = frame_id
        cv2.putText(img, f"id={frame_id} rtt={rtt}ms", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("deepfake", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            await ws.close()
            return


async def run(args):
    cmd = build_ffmpeg_cmd(args.device, args.width, args.height, args.fps, args.platform)
    print("ffmpeg:", " ".join(cmd), file=sys.stderr)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    reader = JpegReader(proc.stdout)
    state = {"sent": 0, "recv": 0, "rtt_ms": 0}

    try:
        async with websockets.connect(args.server, max_size=2**24, ping_interval=20) as ws:
            print(f"connected {args.server}", file=sys.stderr)
            await asyncio.gather(sender(ws, reader, state), receiver(ws, state))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
        cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", required=True, help="wss://host:port/ws")
    p.add_argument("--device", required=True, help='e.g. "video=HD Pro Webcam C920"')
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--platform", choices=["windows", "linux", "mac"], default="windows")
    args = p.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
