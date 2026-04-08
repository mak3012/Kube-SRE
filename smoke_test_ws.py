import asyncio
import json

import websockets


async def main() -> None:
    async with websockets.connect("ws://127.0.0.1:8001/ws") as ws:
        await ws.send(json.dumps({"op": "reset", "reset": {"scenario_id": "ghost_image"}}))
        r = await ws.recv()
        print("reset_ok" in r)

        await ws.send(
            json.dumps(
                {
                    "op": "step",
                    "step": {
                        "action": {
                            "kubectl": {
                                "type": "kubectl",
                                "verb": "get",
                                "namespace": "prod",
                                "kind": "pod",
                                "output": "json",
                            }
                        }
                    },
                }
            )
        )
        r = await ws.recv()
        print("step_ok" in r)


if __name__ == "__main__":
    asyncio.run(main())

