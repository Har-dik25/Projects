import networkx as nx
import matplotlib.pyplot as plt


# ---------- TREE LAYOUT (ROBUST) ----------
def hierarchy_pos(G, root, width=1., vert_gap=0.18, vert_loc=1, xcenter=0.5):
    pos = {}

    def dfs(node, x, y, dx):
        pos[node] = (x, y)
        children = list(G.successors(node))

        if len(children) == 2:
            dfs(children[0], x - dx / 2, y - vert_gap, dx / 2)
            dfs(children[1], x + dx / 2, y - vert_gap, dx / 2)

        elif len(children) == 1:
            dfs(children[0], x, y - vert_gap, dx / 2)
        # 0 children → leaf node (position already assigned)

    dfs(root, xcenter, vert_loc, width)
    return pos


# ---------- DRAW TREE ----------
def draw_tree(G, visited, answer_node=None):
    plt.clf()
    pos = hierarchy_pos(G, "Start")

    colors = []
    for node in G.nodes():
        if node == answer_node:
            colors.append("dodgerblue")
        elif node in visited:
            colors.append("lightgreen")
        else:
            colors.append("lightgray")

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=colors,
        node_size=2300,
        font_size=8,
        arrows=True
    )

    plt.title("Dynamic Binary Decision Tree (0–10 Guess Game)")
    plt.pause(0.5)


# ---------- DYNAMIC TREE SIMULATION ----------
def dynamic_guess_tree():
    G = nx.DiGraph()
    G.add_node("Start")

    visited = set(["Start"])
    low, high = 0, 10
    current = "Start"

    plt.figure(figsize=(14, 10))

    print("Think of a number between 0 and 10.")
    print("Answer only yes or no.\n")

    while low < high:
        mid = (low + high) // 2

        yes_node = f"> {mid} YES"
        no_node = f"> {mid} NO"

        # Always create BOTH children
        if not G.has_edge(current, no_node):
            G.add_edge(current, no_node)
        if not G.has_edge(current, yes_node):
            G.add_edge(current, yes_node)

        draw_tree(G, visited)

        ans = input(f"Is number > {mid}? (yes/no): ").lower()

        if ans == "yes":
            visited.add(yes_node)
            current = yes_node
            low = mid + 1
        else:
            visited.add(no_node)
            current = no_node
            high = mid

    answer_node = f"Answer = {low}"
    G.add_edge(current, answer_node)
    visited.add(answer_node)

    draw_tree(G, visited, answer_node)
    plt.show()

    print(f"\n🎯 Your number is: {low}")


# ---------- MAIN ----------
dynamic_guess_tree()
