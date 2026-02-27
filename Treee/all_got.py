# ---------- TREE NODE ----------
class TreeNode:
    def __init__(self, name):
        self.name = name
        self.parent = None
        self.children = []

    def add_child(self, child):
        child.parent = self
        self.children.append(child)


# ---------- BUILD GOT FAMILY TREE ----------
westeros = TreeNode("Westeros")

# Houses
stark = TreeNode("House Stark")
lannister = TreeNode("House Lannister")
targaryen = TreeNode("House Targaryen")
baratheon = TreeNode("House Baratheon")
tyrell = TreeNode("House Tyrell")
arryn = TreeNode("House Arryn")
martell = TreeNode("House Martell")

for house in [stark, lannister, targaryen, baratheon, tyrell, arryn, martell]:
    westeros.add_child(house)

# -------- STARK --------
ned = TreeNode("Ned Stark")
stark.add_child(ned)

for child in ["Robb Stark", "Sansa Stark", "Arya Stark", "Bran Stark"]:
    ned.add_child(TreeNode(child))

# -------- LANNISTER --------
tywin = TreeNode("Tywin Lannister")
lannister.add_child(tywin)

for child in ["Jaime Lannister", "Cersei Lannister", "Tyrion Lannister"]:
    tywin.add_child(TreeNode(child))

# -------- TARGARYEN --------
aerys = TreeNode("Aerys Targaryen")
targaryen.add_child(aerys)

rhaegar = TreeNode("Rhaegar Targaryen")
daenerys = TreeNode("Daenerys Targaryen")

aerys.add_child(rhaegar)
aerys.add_child(daenerys)

rhaegar.add_child(TreeNode("Jon Snow"))

# -------- BARATHEON --------
robert = TreeNode("Robert Baratheon")
baratheon.add_child(robert)
robert.add_child(TreeNode("Joffrey Baratheon"))

# -------- TYRELL --------
mace = TreeNode("Mace Tyrell")
tyrell.add_child(mace)
mace.add_child(TreeNode("Margaery Tyrell"))

# -------- ARRYN --------
jon_arryn = TreeNode("Jon Arryn")
arryn.add_child(jon_arryn)
jon_arryn.add_child(TreeNode("Robin Arryn"))

# -------- MARTELL --------
doran = TreeNode("Doran Martell")
martell.add_child(doran)
doran.add_child(TreeNode("Oberyn Martell"))


# ---------- TREE OPERATIONS ----------
def get_parent(node):
    return node.parent.name if node.parent else None


def get_children(node):
    return [child.name for child in node.children]


def get_ancestors(node):
    ancestors = []
    while node.parent:
        node = node.parent
        ancestors.append(node.name)
    return ancestors


def get_descendants(node):
    result = []

    def dfs(curr):
        for child in curr.children:
            result.append(child.name)
            dfs(child)

    dfs(node)
    return result


def print_tree(node, level=0):
    print("  " * level + "- " + node.name)
    for child in node.children:
        print_tree(child, level + 1)


# ---------- DEMO ----------
print("\n🌳 GAME OF THRONES FAMILY TREE\n")
print_tree(westeros)

print("\nQueries:")
print("Parent of Arya Stark:", get_parent(ned.children[2]))
print("Children of Tywin Lannister:", get_children(tywin))
print("Ancestors of Jon Snow:", get_ancestors(rhaegar.children[0]))
print("Descendants of House Stark:", get_descendants(stark))
