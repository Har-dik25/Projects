import networkx as nx
import matplotlib.pyplot as plt


# ---------- TREE NODE ----------
class TreeNode:
    def __init__(self, name):
        self.name = name
        self.children = []


    def add_child(self, child):
        self.children.append(child)


# ---------- BUILD GOT TREE ----------
westeros = TreeNode("Westeros")

houses = {
    "House Stark": ["Ned Stark", ["Robb Stark", "Sansa Stark", "Arya Stark", "Bran Stark"]],
    "House Lannister": ["Tywin Lannister", ["Jaime Lannister", "Cersei Lannister", "Tyrion Lannister"]],
    "House Targaryen": ["Aerys Targaryen", ["Rhaegar Targaryen", "Daenerys Targaryen"]],
    "House Baratheon": ["Robert Baratheon", ["Joffrey Baratheon"]],
    "House Tyrell": ["Mace Tyrell", ["Margaery Tyrell"]],
    "House Arryn": ["Jon Arryn", ["Robin Arryn"]],
    "House Martell": ["Doran Martell", ["Oberyn Martell"]]
}

tree_nodes = {"Westeros": westeros}

for house, data in houses.items():
    house_node = TreeNode(house)
    westeros.add_child(house_node)
    tree_nodes[house] = house_node

    head = TreeNode(data[0])
    house_node.add_child(head)
    tree_nodes[data[0]] = head

    for member in data[1]:
        member_node = TreeNode(member)
        head.add_child(member_node)
        tree_nodes[member] = member_node


# ---------- CONVERT TREE → GRAPH ----------
def build_graph(node, G):
    for child in node.children:
        G.add_edge(node.name, child.name)
        build_graph(child, G)


G = nx.DiGraph()
build_graph(westeros, G)


# ---------- HIERARCHICAL LAYOUT ----------
def hierarchy_pos(G, root, width=1., vert_gap=0.15, vert_loc=1, xcenter=0.5):
    pos = {}

    def dfs(node, x, y, dx):
        pos[node] = (x, y)
        children = list(G.successors(node))
        if children:
            step = dx / len(children)
            start = x - dx / 2 + step / 2
            for i, child in enumerate(children):
                dfs(child, start + i * step, y - vert_gap, step)

    dfs(root, xcenter, vert_loc, width)
    return pos


pos = hierarchy_pos(G, "Westeros")


# ---------- COLOR NODES ----------
colors = []
for node in G.nodes():
    if node == "Westeros":
        colors.append("dodgerblue")
    elif node.startswith("House"):
        colors.append("lightgreen")
    else:
        colors.append("lightgray")


# ---------- DRAW ----------
plt.figure(figsize=(18, 12))
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color=colors,
    node_size=2600,
    font_size=9,
    arrows=True
)

plt.title("Game of Thrones – Seven Great Houses Family Tree")
plt.show()
