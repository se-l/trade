import asyncio
import websockets

uri = "ws://localhost:5511"


async def query(x):
    async with websockets.connect(uri) as websocket:
        params = f'ClientResponse `{x}'
        await websocket.send(params)

        return await websocket.recv()
        # print(resp)


result = asyncio.run(query("A"))
print(result)
# older versions use the following line
# asyncio.get_event_loop().run_until_complete(query("A"))