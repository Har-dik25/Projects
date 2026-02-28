# 📚 Array + Stack — Data Structure Implementations

Hands-on implementations of **Array** and **Stack** data structures in Python, with real-world applications.

---

## 📂 Files

### 1. `array.py` — Custom Array Class
A fixed-capacity array implementation from scratch with core operations:
- **Traverse** — Print all elements
- **Search** — Linear search by value
- **Insert** — Insert at any index with element shifting
- **Delete** — Remove by index with element shifting

### 2. `stack.py` — Stack + Text Editor
A **Stack** data structure powering an **Undo/Redo Text Editor**:
- Type text → appends to current text
- Delete characters → removes from end
- **Undo** — reverts to previous state (stack pop)
- **Redo** — re-applies undone changes (second stack)

### 3. `cal/cal.py` — Scientific Calculator with History
A CLI calculator using **Array + Stack** together:
- **Array** — stores calculation history
- **Stack** — enables undo/redo of calculations
- Supports: `+`, `-`, `*`, `/`, `^` (power), `sqrt`, `log`

---

## ▶️ How to Run

```bash
# Run array demo
python array.py

# Run text editor with undo/redo
python stack.py

# Run scientific calculator
python cal/cal.py
```

No external dependencies required.

---

## 🧠 Concepts Demonstrated
- Array operations (traverse, search, insert, delete)
- Stack operations (push, pop, is_empty)
- Undo/Redo pattern using dual stacks
- Combined data structure usage in real applications
