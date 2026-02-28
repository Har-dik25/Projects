# 💬 Chat Application — Real-Time WebSocket Messaging

A **real-time chat application** built with FastAPI WebSockets on the backend and vanilla HTML/CSS/JS on the frontend.

---

## ✨ Features

- **Real-Time Messaging** — Instant message delivery via WebSocket
- **Message History** — New users see full chat history on join
- **Delete Messages** — Remove individual messages (synced across all clients)
- **Clear Chat** — Admin-level clear of entire chat history
- **Group Chat Support** — Configurable group channels
- **Modern UI** — Dark-themed login screen + split-pane chat interface

---

## 🏗️ Project Structure

```
chat-application/
├── backend/
│   ├── main.py             # FastAPI WebSocket server
│   └── requirements.txt    # Python dependencies
└── frontend/
    ├── index.html          # Chat UI with login
    └── app.js              # WebSocket client logic
```

---

## ▶️ How to Run

### 1. Start the Backend
```bash
cd backend
pip install fastapi uvicorn
uvicorn main:app --reload --port 8000
```

### 2. Open the Frontend
Open `frontend/index.html` in your browser (or serve it with a simple HTTP server).

### 3. Chat!
1. Enter a username
2. Select Private or Group chat
3. Start messaging!

Open multiple browser tabs to simulate multiple users.

---

## 🔌 WebSocket Protocol

| Message Type | Direction | Description |
|---|---|---|
| `message` | Client → Server → All | Normal chat message |
| `delete` | Client → Server → All | Delete a specific message by ID |
| `clear` | Client → Server → All | Clear entire chat history |

---

## 🧠 Tech Stack
- **Backend:** FastAPI, WebSocket, Python
- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **Protocol:** WebSocket (full-duplex real-time communication)
