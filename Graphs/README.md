# 🕸️ Graphs — Dynamic Relationship Graph Builder

An interactive **directed, weighted graph** builder that visualizes relationships between characters (or any entities) using NetworkX and Matplotlib.

---

## ✨ Features

- **Interactive Input** — Add relationships dynamically from the CLI
- **Directed Edges** — Specify "from" and "to" relationships
- **Weighted Edges** — Assign comfort weights from -10 to +10
- **Color Coding:**
  - 🟢 Green edges = positive relationships
  - 🔴 Red edges = negative relationships
- **Edge Thickness** — Proportional to relationship strength
- **Auto Layout** — Spring-force graph layout for clean visualization

---

## ▶️ How to Run

### Install Dependencies
```bash
pip install networkx matplotlib
```

### Run
```bash
python avg.py
```

Enter character relationships when prompted. Type `STOP` to finish and see the graph.

---

## 💡 Example

```
From character: Jon Snow
To character: Daenerys
Comfort weight (-10 to +10): 8

From character: Cersei
To character: Tyrion
Comfort weight (-10 to +10): -9
```

Generates a graph showing strong positive bond between Jon–Daenerys (thick green) and strong negative bond between Cersei–Tyrion (thick red).

---

## 🧠 Concepts Demonstrated
- Directed graph (DiGraph)
- Weighted edges
- Graph visualization and layout algorithms
- NetworkX library usage
