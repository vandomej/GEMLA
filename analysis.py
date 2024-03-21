# Re-importing necessary libraries
import json
import matplotlib.pyplot as plt
import networkx as nx
import random

def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.successors(root))  # Use successors to get children for DiGraph
        if not isinstance(G, nx.DiGraph):
            if parent is not None:
                children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                    vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                    pos=pos, parent=root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

# Simplified JSON data for demonstration
with open('gemla/test.json', 'r') as file:
    simplified_json_data = json.load(file)

# Function to traverse the tree and create a graph
def traverse(node, graph, parent=None):
    if node is None:
        return
    
    node_id = node["val"]["id"]
    if "node" in node["val"] and node["val"]["node"]:
        scores = node["val"]["node"]["scores"]
        generations = node["val"]["node"]["generation"]
        population_size = node["val"]["node"]["population_size"]
        # Prepare to track the highest score across all generations and the corresponding individual
        overall_max_score = float('-inf')
        overall_max_score_individual = None
        overall_max_score_gen = None
        
        for gen, gen_scores in enumerate(scores):
            if gen_scores:  # Ensure the dictionary is not empty
                # Find the max score and the individual for this generation
                max_score_for_gen = max(gen_scores.values())
                individual_with_max_score_for_gen = max(gen_scores, key=gen_scores.get)

                # if max_score_for_gen > overall_max_score:
                overall_max_score = max_score_for_gen
                overall_max_score_individual = individual_with_max_score_for_gen
                overall_max_score_gen = gen

        label = f"{node_id}\nGenerations: {generations}, Population: {population_size}\nMax score: {overall_max_score:.6f} (Individual {overall_max_score_individual} in Gen {overall_max_score_gen})"
    else:
        label = node_id

    graph.add_node(node_id, label=label)
    if parent:
        graph.add_edge(parent, node_id)
    
    traverse(node.get("left"), graph, parent=node_id)
    traverse(node.get("right"), graph, parent=node_id)


# Create a directed graph
G = nx.DiGraph()

# Populate the graph
traverse(simplified_json_data[0], G)

# Find the root node (a node with no incoming edges)
root_candidates = [node for node, indeg in G.in_degree() if indeg == 0]

if root_candidates:
    root_node = root_candidates[0]  # Assuming there's only one root candidate
else:
    root_node = None  # This should ideally never happen in a properly structured tree

# Use the determined root node for hierarchy_pos
if root_node is not None:
    pos = hierarchy_pos(G, root=root_node)
    labels = nx.get_node_attributes(G, 'label')
    nx.draw(G, pos, labels=labels, with_labels=True, arrows=True)
    plt.show()
else:
    print("No root node found. Cannot draw the tree.")