from collections import defaultdict, Counter, namedtuple
import matplotlib.pyplot as plt
import networkx as nx

transactions = [
    ['f','a','c','d','g','i','m','p'],   # TID 100
    ['a','b','c','f','i','m','o'],     # TID 200
    ['b','f','h','j','o'],             # TID 300
    ['b','c','k','s','p'],             # TID 400
    ['a','f','c','e','l','p','m','n']  # TID 500
]

tids = [100, 200, 300, 400, 500]
min_support = 3


item_counts = Counter()
for t in transactions:
    item_counts.update(set(t))

freq_items = {item: cnt for item, cnt in item_counts.items() if cnt >= min_support}

ordered_items = sorted(freq_items.keys(), key=lambda x: (-freq_items[x], x))

filtered_ordered_transactions = []
for t in transactions:
    filtered = [it for it in ordered_items if it in t]
    filtered_ordered_transactions.append(filtered)


class FPNode:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.link = None  
    def increment(self, count):
        self.count += count

class FPTree:
    def __init__(self):
        self.root = FPNode(None, 1, None)
        self.headers = {} 

    def add_transaction(self, transaction, count=1):
        current = self.root
        for item in transaction:
            if item in current.children:
                current.children[item].increment(count)
            else:
                new_node = FPNode(item, count, current)
                current.children[item] = new_node
                
                if item in self.headers:
                    last = self.headers[item]
                    while last.link is not None:
                        last = last.link
                    last.link = new_node
                else:
                    self.headers[item] = new_node
            current = current.children[item]

    def conditional_pattern_base(self, item):
        base_patterns = []
        node = self.headers.get(item, None)
        while node is not None:
            path = []
            parent = node.parent
            while parent and parent.item is not None:
                path.append(parent.item)
                parent = parent.parent
            path.reverse()
            if path:
                base_patterns.append((path, node.count))
            node = node.link
        return base_patterns

def build_fptree(transactions, min_support):
    freq = Counter()
    for t in transactions:
        freq.update(t)

    freq_items = {item for item, count in freq.items() if count >= min_support}
    filtered_txns = []
    for t in transactions:
        filtered = [item for item in t if item in freq_items]
        filtered.sort(key=lambda x: (-freq[x], x))
        if filtered:
            filtered_txns.append(filtered)

    tree = FPTree()
    for t in filtered_txns:
        tree.add_transaction(t)
    return tree, freq_items

def mine_tree(tree, prefix, min_support, freq_itemsets):
    items = sorted(tree.headers.keys(), key=lambda i: tree.headers[i].count if tree.headers[i] else 0)
    for item in items:
        new_prefix = prefix.copy()
        new_prefix.add(item)

        support = 0
        node = tree.headers[item]
        while node:
            support += node.count
            node = node.link

        if support >= min_support:
            freq_itemsets[tuple(sorted(new_prefix))] = support

            cond_patt_base = tree.conditional_pattern_base(item)
            cond_transactions = []
            for path, count in cond_patt_base:
                cond_transactions.extend([path] * count)

            if cond_transactions:
                cond_tree, cond_freq_items = build_fptree(cond_transactions, min_support)
                if cond_tree.root.children:
                    mine_tree(cond_tree, new_prefix, min_support, freq_itemsets)

main_tree, freq_items_set = build_fptree(filtered_ordered_transactions, min_support)

all_freq_itemsets = {}

mine_tree(main_tree, set(), min_support, all_freq_itemsets)

for item in freq_items_set:
    all_freq_itemsets[(item,)] = item_counts[item]

def print_table(headers, rows):
    col_widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    header_row = " | ".join(header.ljust(col_widths[i]) for i, header in enumerate(headers))
    separator = "-+-".join('-' * col_widths[i] for i in range(len(headers)))
    print(header_row)
    print(separator)
 
    for row in rows:
        print(" | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)))
    print()

freq_single_items = sorted(((item, item_counts[item]) for item in freq_items_set), key=lambda x: (-x[1], x[0]))
table1_rows = [(item, support) for item, support in freq_single_items]
print("Frequent Single Items (Support >= {})".format(min_support))
print_table(["Item", "Support"], table1_rows)

table2_rows = []
for tid, orig, filt in zip(tids, transactions, filtered_ordered_transactions):
    table2_rows.append((tid, ",".join(orig), ",".join(filt)))
print("Transactions Filtered to Frequent Items")
print_table(["TID", "Original", "Filtered & Ordered (freq only)"], table2_rows)

sorted_itemsets = sorted(all_freq_itemsets.items(), key=lambda x: (len(x[0]), -x[1], x[0]))
table3_rows = [(",".join(items), support) for items, support in sorted_itemsets]
print("All Frequent Itemsets (Support >= {})".format(min_support))
print_table(["Itemset", "Support"], table3_rows)

print("Summary:")
print(f"- Minimum support: {min_support}")
print(f"- Frequent single items (count): {len(freq_single_items)}")
print(f"- Frequent itemsets found (count): {len(sorted_itemsets)}")


# ----------- FP-Tree Visualization -----------

def build_graph(fpnode, graph, parent=None):
    """Recursively add nodes and edges to the graph from the FP-tree."""
    if fpnode.item is None:
        node_label = 'Root'
    else:
        node_label = f'{fpnode.item} ({fpnode.count})'
    graph.add_node(id(fpnode), label=node_label)
    if parent is not None:
        graph.add_edge(id(parent), id(fpnode))
    for child in fpnode.children.values():
        build_graph(child, graph, fpnode)

def draw_fp_tree(fp_tree):
    G = nx.DiGraph()
    build_graph(fp_tree.root, G)

    pos = hierarchy_pos(G, id(fp_tree.root))
    labels = nx.get_node_attributes(G, 'label')

    plt.figure(figsize=(12,8))
    nx.draw_networkx(G, pos, labels=labels, node_size=2500, node_color='lightblue',
                     font_size=10, font_weight='bold', arrowsize=20)
    plt.title("FP-Tree Visualization")
    plt.axis('off')
    plt.show()


def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    Position nodes in hierarchy layout (from https://stackoverflow.com/a/29597209)
    """
    if root is None:
        root = list(G.nodes)[0]

    def _hierarchy_pos(G, root, leftmost, width, vert_gap, vert_loc, pos, parent=None):
        children = list(G.successors(root))
        if not children:
            pos[root] = (leftmost[0], vert_loc)
            leftmost[0] += width
        else:
            start = leftmost[0]
            for child in children:
                _hierarchy_pos(G, child, leftmost, width, vert_gap, vert_loc - vert_gap, pos, root)
            middle = (start + leftmost[0] - width) / 2
            pos[root] = (middle, vert_loc)
        return pos

    return _hierarchy_pos(G, root, [0], width, vert_gap, vert_loc, {})

# Draw the FP-tree
draw_fp_tree(main_tree)
