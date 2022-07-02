import websockets
import json
import asyncio


uri = "ws://localhost:5511"


async def subscribe(*args):
    async with websockets.connect(uri) as websocket:
        # subcription
        submsg = f"sub . (`gendata; `{'`'.join(args)})"
        await websocket.send(submsg)

        # updates
        while True:
            json_string = await websocket.recv()
            data = json.loads(json_string)
            print(data)


asyncio.get_event_loop().run_until_complete(subscribe("A", "B"))
# asyncio.run(subscribe("A", "B"))