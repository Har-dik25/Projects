import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


class GuessTreeGUI:
    def __init__(self):
        self.G = nx.DiGraph()
        self.G.add_node("Start")

        self.low = 0
        self.high = 10
        self.current = "Start"
        self.visited = {"Start"}
        self.answer_node = None

        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        plt.subplots_adjust(bottom=0.2)

        # YES button
        ax_yes = plt.axes([0.55, 0.05, 0.15, 0.075])
        self.btn_yes = Button(ax_yes, "YES")
        self.btn_yes.on_clicked(self.yes_clicked)

        # NO button
        ax_no = plt.axes([0.30, 0.05, 0.15, 0.075])
        self.btn_no = Button(ax_no, "NO")
        self.btn_no.on_clicked(self.no_clicked)

        self.draw()

    # ---------- TREE LAYOUT ----------
    def hierarchy_pos(self, root, width=1., vert_gap=0.18, vert_loc=1, xcenter=0.5):
        pos = {}

        def dfs(node, x, y, dx):
            pos[node] = (x, y)
            children = list(self.G.successors(node))

            if len(children) == 2:
                dfs(children[0], x - dx / 2, y - vert_gap, dx / 2)
                dfs(children[1], x + dx / 2, y - vert_gap, dx / 2)
            elif len(children) == 1:
                dfs(children[0], x, y - vert_gap, dx / 2)

        dfs(root, xcenter, vert_loc, width)
        return pos

    # ---------- DRAW ----------
    def draw(self):
        self.ax.clear()

        pos = self.hierarchy_pos("Start")

        colors = []
        for node in self.G.nodes():
            if node == self.answer_node:
                colors.append("dodgerblue")
            elif node in self.visited:
                colors.append("lightgreen")
            else:
                colors.append("lightgray")

        nx.draw(
            self.G,
            pos,
            ax=self.ax,
            with_labels=True,
            node_color=colors,
            node_size=2400,
            font_size=8,
            arrows=True
        )

        if self.low < self.high:
            mid = (self.low + self.high) // 2
            self.ax.set_title(f"Is number > {mid} ?", fontsize=14)
        else:
            self.ax.set_title(f"🎯 Answer = {self.low}", fontsize=14)

        self.fig.canvas.draw_idle()

    # ---------- BUTTON ACTIONS ----------
    def yes_clicked(self, event):
        if self.low >= self.high:
            return

        mid = (self.low + self.high) // 2
        yes_node = f"> {mid} YES"
        no_node = f"> {mid} NO"

        self.G.add_edge(self.current, no_node)
        self.G.add_edge(self.current, yes_node)

        self.current = yes_node
        self.visited.add(yes_node)
        self.low = mid + 1

        if self.low == self.high:
            self.answer_node = f"Answer = {self.low}"
            self.G.add_edge(self.current, self.answer_node)
            self.visited.add(self.answer_node)

        self.draw()

    def no_clicked(self, event):
        if self.low >= self.high:
            return

        mid = (self.low + self.high) // 2
        yes_node = f"> {mid} YES"
        no_node = f"> {mid} NO"

        self.G.add_edge(self.current, no_node)
        self.G.add_edge(self.current, yes_node)

        self.current = no_node
        self.visited.add(no_node)
        self.high = mid

        if self.low == self.high:
            self.answer_node = f"Answer = {self.low}"
            self.G.add_edge(self.current, self.answer_node)
            self.visited.add(self.answer_node)

        self.draw()


# ---------- RUN ----------
print("🎯 Think of a number between 0 and 10.")
GuessTreeGUI()
plt.show()
