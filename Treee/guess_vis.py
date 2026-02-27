import networkx as nx
import matplotlib.pyplot as plt


# ---------- BUILD FULL BINARY TREE ----------
def build_tree(G, node, low, high):
    if low == high:
        leaf = f"Answer={low}"
        G.add_edge(node, leaf)
        return

    mid = (low + high) // 2

    left = f"> {mid} NO"
    right = f"> {mid} YES"

    G.add_edge(node, left)   # NO  -> left
    G.add_edge(node, right)  # YES -> right

    build_tree(G, left, low, mid)
    build_tree(G, right, mid + 1, high)


# ---------- USER SIMULATION ----------
def simulate():
    visited = set()
    low, high = 0, 10
    current = "Start"

    visited.add(current)

    print("Think of a number between 0 and 10.")
    print("Answer only yes or no.\n")

    while low < high:
        mid = (low + high) // 2
        ans = input(f"Is number > {mid}? (yes/no): ").lower()

        if ans == "yes":
            current = f"> {mid} YES"
            low = mid + 1
        else:
            current = f"> {mid} NO"
            high = mid

        visited.add(current)

    answer_node = f"Answer={low}"
    visited.add(answer_node)

    return visited, answer_node


# ---------- TREE LAYOUT (TOP-DOWN) ----------
def hierarchy_pos(G, root, width=1., vert_gap=0.15, vert_loc=1, xcenter=0.5):
    pos = {}

    def dfs(node, x, y, dx):
        pos[node] = (x, y)
        children = list(G.successors(node))
        if len(children) == 2:
            dfs(children[0], x - dx/2, y - vert_gap, dx/2)
            dfs(children[1], x + dx/2, y - vert_gap, dx/2)
        elif len(children) == 1:
            dfs(children[0], x, y - vert_gap, dx/2)

    dfs(root, xcenter, vert_loc, width)
    return pos


# ---------- VISUALIZATION ----------
def visualize(G, visited, answer_node):
    pos = hierarchy_pos(G, "Start")

    colors = []
    for node in G.nodes():
        if node == answer_node:
            colors.append("dodgerblue")
        elif node in visited:
            colors.append("lightgreen")
        else:
            colors.append("lightgray")

    plt.figure(figsize=(14, 10))
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=colors,
        node_size=2300,
        font_size=8,
        arrows=True
    )

    plt.title("Binary Decision Tree (Visited + Empty Nodes)")
    plt.show()


# ---------- MAIN ----------
G = nx.DiGraph()
G.add_node("Start")

build_tree(G, "Start", 0, 10)
visited_nodes, final_node = simulate()
visualize(G, visited_nodes, final_node)