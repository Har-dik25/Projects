import networkx as nx
import matplotlib.pyplot as plt

# Create directed graph
G = nx.DiGraph()

print("Enter relationships between characters.")
print("Type 'STOP' anytime to finish.\n")

while True:
    frm = input("From character: ")
    if frm.upper() == "STOP":
        break

    to = input("To character: ")
    if to.upper() == "STOP":
        break

    try:
        weight = int(input("Comfort weight (-10 to +10): "))
    except:
        print("Invalid weight. Try again.\n")
        continue

    # Add edge
    G.add_edge(frm, to, weight=weight)
    print("Added!\n")

# If no data entered
if len(G.edges()) == 0:
    print("No relationships entered.")
    exit()

# Layout
pos = nx.spring_layout(G, seed=42)

# Edge colors (green positive, red negative)
edge_colors = [
    "green" if G[u][v]["weight"] > 0 else "red"
    for u, v in G.edges()
]

# Edge thickness based on strength
edge_widths = [
    abs(G[u][v]["weight"]) / 2
    for u, v in G.edges()
]

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=3000)

# Draw edges
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, arrows=True)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

# Draw edge weights
labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

plt.title("Dynamic Relationship Graph\n(Directed + Weighted + Positive/Negative)")
plt.axis("off")
plt.show()