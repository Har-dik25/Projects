# 🤖 AI Agent Projects

A collection of Python-based AI agents demonstrating the **Think → Decide → Act** pattern — the foundation of modern agent architectures.

---

## 📂 Files

### 1. `agent.py` — Simple AI Agent
A minimal AI agent with a calculator tool. It detects math expressions in user input and routes them to the calculator, otherwise responds with a fallback message.

**Architecture:**
```
User Input → Think (regex detection) → Act (tool call or chat) → Output
```

### 2. `calc.py` — NLP Calculator Agent
A more advanced agent with **4 separate tools** (add, subtract, multiply, divide). The agent parses natural language to decide which tool to call and extracts numbers from the text.

**Supported Commands:**
- `"Add 5 and 3"` → `Result: 8`
- `"Multiply 10 times 4"` → `Result: 40`
- `"Divide 100 by 5"` → `Result: 20.0`

### 3. `ticket.py` — Ticket Management System
A CLI-based ticket support system using a **circular queue** data structure. Demonstrates FIFO (First In, First Out) processing of support tickets.

**Features:**
- Add, resolve, peek, and list tickets
- Queue-based priority processing
- Configurable capacity

---

## ▶️ How to Run

```bash
# Run the simple agent
python agent.py

# Run the NLP calculator agent
python calc.py

# Run the ticket system
python ticket.py
```

No external dependencies required — runs on pure Python.

---

## 🧠 Concepts Demonstrated
- Agent architecture (Think → Act loop)
- Tool-use pattern (function calling)
- Natural language processing with regex
- Queue data structure (FIFO)
- CLI interaction pattern
