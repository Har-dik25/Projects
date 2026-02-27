from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, List
import json
import uuid

app = FastAPI()

connections: Dict[str, WebSocket] = {}
groups = {"group1": []}

# message history
message_history: List[dict] = []

@app.websocket("/ws/{username}")
async def chat_ws(websocket: WebSocket, username: str):
    await websocket.accept()
    connections[username] = websocket

    # send history
    for msg in message_history:
        await websocket.send_text(json.dumps(msg))

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            # CLEAR HISTORY
            if message["type"] == "clear":
                message_history.clear()
                for user in connections.values():
                    await user.send_text(json.dumps({
                        "type": "clear"
                    }))
                continue

            # DELETE MESSAGE
            if message["type"] == "delete":
                msg_id = message["id"]
                message_history[:] = [m for m in message_history if m["id"] != msg_id]

                for user in connections.values():
                    await user.send_text(json.dumps({
                        "type": "delete",
                        "id": msg_id
                    }))
                continue

            # NORMAL MESSAGE
            msg_data = {
                "id": str(uuid.uuid4()),
                "from": username,
                "msg": message["msg"],
                "type": "message"
            }

            message_history.append(msg_data)

            for ws in connections.values():
                await ws.send_text(json.dumps(msg_data))

    except WebSocketDisconnect:
        del connections[username]
