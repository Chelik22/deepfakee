"""Local echo WS server for pipeline smoke test. Returns each frame unchanged."""
import asyncio
import websockets


async def handler(ws):
    print("client connected")
    try:
        async for msg in ws:
            await ws.send(msg)
    except websockets.ConnectionClosed:
        pass
    print("client disconnected")


async def main():
    async with websockets.serve(handler, "127.0.0.1", 8000, max_size=2**24):
        print("echo server on ws://127.0.0.1:8000/ws")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
