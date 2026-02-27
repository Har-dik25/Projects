# ---------- TREE NODE ----------
class TreeNode:
    def __init__(self, name):
        self.name = name
        self.parent = None
        self.children = []

    def add_child(self, child):
        child.parent = self
        self.children.append(child)


# ---------- BUILD TREE (HARD-CODED HIERARCHY) ----------
# Root
aegon = TreeNode("Aegon")

# Level 1
rhaegar = TreeNode("Rhaegar")
viserys = TreeNode("Viserys")

# Level 2
jon = TreeNode("Jon")
rhaenys = TreeNode("Rhaenys")
daenerys = TreeNode("Daenerys")

# Build relationships
aegon.add_child(rhaegar)
aegon.add_child(viserys)

rhaegar.add_child(jon)
rhaegar.add_child(rhaenys)

viserys.add_child(daenerys)


# ---------- TREE OPERATIONS ----------
def get_parent(node):
    """Return parent of a node"""
    return node.parent.name if node.parent else None


def get_children(node):
    """Return children of a node"""
    return [child.name for child in node.children]


def get_ancestors(node):
    """Return all ancestors (preceding nodes)"""
    ancestors = []
    while node.parent:
        node = node.parent
        ancestors.append(node.name)
    return ancestors


def get_descendants(node):
    """Return all descendants (succeeding nodes)"""
    descendants = []

    def dfs(curr):
        for child in curr.children:
            descendants.append(child.name)
            dfs(child)

    dfs(node)
    return descendants


# ---------- DEMO QUERIES ----------
print("Parent of Jon:", get_parent(jon))
print("Children of Rhaegar:", get_children(rhaegar))
print("Ancestors of Jon:", get_ancestors(jon))
print("Descendants of Aegon:", get_descendants(aegon))