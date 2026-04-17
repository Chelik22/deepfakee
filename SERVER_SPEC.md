# RunPod Deepfake Server — Spec for Implementer

Твоя задача: WebSocket-сервер, принимающий JPEG-кадры с вебки, пропускающий через
deepfake-модель, возвращающий обработанные JPEG обратно. Клиент уже написан —
менять протокол нельзя.

## Wire Protocol (строго, не меняй)

Один кадр = одно WebSocket binary-сообщение. Оба направления. Формат:

```
[8 bytes frame_id   — big-endian uint64]
[8 bytes timestamp_ms — big-endian uint64, UNIX epoch ms]
[JPEG bytes         — начинается с 0xFFD8, заканчивается 0xFFD9]
```

Python:
```python
import struct
HEADER = ">QQ"  # 16 bytes
frame_id, ts = struct.unpack(HEADER, msg[:16])
jpeg = msg[16:]
```

Ответ должен нести **тот же `frame_id`** что пришёл (клиент по нему дропает
устаревшие). `timestamp_ms` сохраняй оригинальный — клиент меряет RTT.

Endpoint: `/ws`. Схема: `ws://` или `wss://`. Один клиент = одно соединение.
Поддержи `ping_interval` (websockets шлёт ping каждые 20s).

## Стек

- Python 3.10+
- `websockets` (async, не Flask-SocketIO)
- `opencv-python` для decode/encode JPEG
- `torch` + твоя deepfake-модель (InsightFace / SimSwap / Roop / любая)
- CUDA на RunPod — держи модель на `cuda:0`, `torch.no_grad()`, `half()` если модель поддерживает

## Архитектура (обязательно)

Три асинхронных стадии + queue между ними. Нельзя блокировать event loop тяжёлым
inference — вынеси в `run_in_executor` (CPU) или отдельный GPU-поток.

```
ws.recv  ──►  inbound_queue (maxsize=2)  ──►  worker (GPU)  ──►  outbound_queue (maxsize=2)  ──►  ws.send
```

### Drop policy (критично для latency)

`maxsize=2` + `put_nowait` с try/except. При переполнении — **дропай старые**,
не новые. Иначе очередь растёт, RTT взрывается. Пример:

```python
try:
    inbound.put_nowait(item)
except asyncio.QueueFull:
    try: inbound.get_nowait()       # drop oldest
    except asyncio.QueueEmpty: pass
    inbound.put_nowait(item)
```

### Worker

```python
async def worker(inbound, outbound, model):
    loop = asyncio.get_event_loop()
    while True:
        frame_id, ts, jpeg = await inbound.get()
        out_jpeg = await loop.run_in_executor(None, process_frame, jpeg, model)
        await outbound.put((frame_id, ts, out_jpeg))
```

`process_frame`:
1. `cv2.imdecode` → BGR ndarray
2. prep → model forward (no_grad)
3. postprocess → BGR
4. `cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])` → bytes

### Handler

```python
async def handler(ws):
    inbound = asyncio.Queue(maxsize=2)
    outbound = asyncio.Queue(maxsize=2)
    tasks = [
        asyncio.create_task(reader(ws, inbound)),
        asyncio.create_task(worker(inbound, outbound, MODEL)),
        asyncio.create_task(sender(ws, outbound)),
    ]
    try:
        await asyncio.gather(*tasks)
    finally:
        for t in tasks: t.cancel()
```

`reader` парсит header, кладёт `(frame_id, ts, jpeg)` в inbound (с drop policy).
`sender` берёт из outbound, пакует `struct.pack(">QQ", fid, ts) + jpeg`, `ws.send`.

## Model Loading

Загружай **один раз** при старте процесса, не на каждое соединение. Глобальная
переменная `MODEL`, прогрей dummy-кадром чтобы CUDA-кэш/cudnn autotune выбрали
ядра до первого реального запроса.

## RunPod deployment

- Expose TCP port (обычно 8000). В RunPod Pod Settings → Expose HTTP Ports:
  поставь тот порт на котором запускаешь `websockets.serve`. Получишь публичный
  URL вида `https://<pod-id>-8000.proxy.runpod.net` — клиент цепляется через
  `wss://<pod-id>-8000.proxy.runpod.net/ws` (RunPod proxy даёт TLS бесплатно).
- Альтернативно — TCP port mapping, тогда `ws://<host>:<mapped_port>/ws`.
- GPU: RTX 4090 / A5000 хватит на 30fps 640x480 для большинства моделей.
- Dockerfile база: `nvidia/cuda:12.1.0-runtime-ubuntu22.04` + python3.10 +
  `pip install websockets opencv-python-headless torch --index-url https://download.pytorch.org/whl/cu121`.

## Sanity checks

1. Сервер стартует, логирует "model loaded" один раз.
2. `wscat -c wss://.../ws` подключается без ошибок.
3. Запусти клиента — окно `deepfake` открывается, RTT < 200ms на локальном тесте,
   frame_id монотонно растёт.
4. Отруби клиента (Ctrl+C) — сервер не падает, принимает следующего.

## NOT DO

- Не пересылай base64/JSON — overhead убьёт latency. Только binary.
- Не копи frame_id на сервере — echo'шь клиентский. Клиент дропает старые сам.
- Не делай HTTP polling / Server-Sent Events. Только WS.
- Не грузи модель per-request. Один раз на старте.
- Не пиши blocking `cv2.imshow` / GUI на сервере — headless.
