# 🌳 Treee — Tree Data Structure Implementations

A collection of **tree data structure** projects with interactive visualizations — featuring a Game of Thrones family tree and a binary decision tree guessing game.

---

## 📂 Files

### 1. `all_got.py` — Game of Thrones Family Tree
A complete family tree of the Great Houses of Westeros built using a custom `TreeNode` class.

**Tree Operations:**
- `get_parent(node)` — Find a character's parent
- `get_children(node)` — List all children
- `get_ancestors(node)` — Trace lineage to the root
- `get_descendants(node)` — Find all descendants (DFS)
- `print_tree(node)` — Display the full tree

**Houses Included:** Stark, Lannister, Targaryen, Baratheon, Tyrell, Arryn, Martell

### 2. `all_got_vis.py` — GoT Family Tree (Visualized)
Same family tree with **NetworkX + Matplotlib** visualization — renders the tree as a directed graph.

### 3. `got_tree.py` — Simplified GoT Tree
A minimal version of the family tree for quick demos.

### 4. `guess_number.py` — Number Guessing Game
A binary search-based number guessing game (0–10) using the terminal.

### 5. `vis.py` & `guess_vis.py` — Guessing Game with Tree Visualization
The guessing game with a **real-time binary decision tree** that grows as you play. Uses NetworkX to visualize the search path.

### 6. `v.py` — Binary Search Tree Visualization
Interactive BST visualization showing node coloring for visited, unvisited, and answer nodes.

---

## ▶️ How to Run

```bash
# Install dependencies
pip install networkx matplotlib

# Run GoT Family Tree (text-based)
python all_got.py

# Run GoT Family Tree (visual)
python all_got_vis.py

# Run Guessing Game with live tree visualization
python vis.py
```

---

## 🧠 Concepts Demonstrated
- Tree data structure (N-ary tree, binary tree)
- Tree traversal (DFS, level-order)
- Parent, children, ancestor, descendant queries
- Binary search algorithm
- Real-time graph visualization
