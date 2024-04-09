# Re-importing necessary libraries
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# Simplified JSON data for demonstration
with open('gemla/round2.json', 'r') as file:
    simplified_json_data = json.load(file)

target_node_id = '0c1e64dc-6ddf-4dbb-bf6e-e8218b925194'

# Function to traverse the tree to find a node id
def traverse_left_nodes(node):
    if node is None:
        return []
    
    left_node = node.get("left")
    if left_node is None:
        return [node]
    
    return [node] + traverse_left_nodes(left_node)

# Function to traverse the tree to find a node id
def traverse_right_nodes(node):
    if node is None:
        return []
    
    right_node = node.get("right")
    left_node = node.get("left")

    if right_node is None and left_node is None:
        return []
    elif right_node and left_node:
        return [right_node] + traverse_right_nodes(left_node)
    
    return []


# Getting the left graph
left_nodes = traverse_left_nodes(simplified_json_data[0])
left_nodes.reverse()
# print(node)
# Print properties available on the first node
node = left_nodes[0]
# print(node["val"].keys())

scores = []
for node in left_nodes:
    # print(node)
    # print(f'Node ID: {node["val"]["id"]}')
    # print(f'Node scores length: {len(node["val"]["node"]["scores"])}')
    if node["val"]["node"]:
        node_scores = node["val"]["node"]["scores"]
        if node_scores:
            for score in node_scores:
                scores.append(score)

# print(scores)

scores_values = [list(score_set.values()) for score_set in scores]

# Set up the figure for plotting on the same graph
fig, ax = plt.subplots(figsize=(10, 6))

# Generate a boxplot for each set of scores on the same graph
boxplots = ax.boxplot(scores_values, vert=False, patch_artist=True, labels=[f'Set {i+1}' for i in range(len(scores_values))])

# Set figure name to node id
# fig.canvas.set_window_title('Main node line')

# Labeling
ax.set_xlabel(f'Scores - Main Line')
ax.set_ylabel('Score Sets')
ax.yaxis.grid(True)  # Add horizontal grid lines for clarity

# Set y-axis labels to be visible
ax.set_yticklabels([f'Set {i+1}' for i in range(len(scores_values))])

# Getting most recent right graph
right_nodes = traverse_right_nodes(simplified_json_data[0])
target_node_id = None
target_node = None
if target_node_id:
    for node in right_nodes:
        if node["val"]["id"] == target_node_id:
            target_node = node
            break
else:
    target_node = right_nodes[1]
scores = target_node["val"]["node"]["scores"]

scores_values = [list(score_set.values()) for score_set in scores]

# Set up the figure for plotting on the same graph
fig, ax = plt.subplots(figsize=(10, 6))

# Generate a boxplot for each set of scores on the same graph
boxplots = ax.boxplot(scores_values, vert=False, patch_artist=True, labels=[f'Set {i+1}' for i in range(len(scores_values))])


# Labeling
ax.set_xlabel(f'Scores: {target_node['val']['id']}')
ax.set_ylabel('Score Sets')
ax.yaxis.grid(True)  # Add horizontal grid lines for clarity

# Set y-axis labels to be visible
ax.set_yticklabels([f'Set {i+1}' for i in range(len(scores_values))])

# Find the highest scoring sets combining all scores and generations
scores = []
for node in left_nodes:
    if node["val"]["node"]:
        node_scores = node["val"]["node"]["scores"]
        translated_node_scores = []
        if node_scores:
            for i in range(len(node_scores)):
                for (individual, score) in node_scores[i].items():
                    translated_node_scores.append((node["val"]["id"], i, score))

            scores.append(translated_node_scores)

# Add scores from the right nodes
for node in right_nodes:
    if node["val"]["node"]:
        node_scores = node["val"]["node"]["scores"]
        translated_node_scores = []
        if node_scores:
            for i in range(len(node_scores)):
                for (individual, score) in node_scores[i].items():
                    translated_node_scores.append((node["val"]["id"], i, score))
            scores.append(translated_node_scores)

# Organize scores by individual and then by generation
individual_generation_scores = defaultdict(lambda: defaultdict(list))
for sublist in scores:
    for id, generation, score in sublist:
        individual_generation_scores[id][generation].append(score)

# Calculate Q3 for each individual's generation
individual_generation_q3 = {}
for id, generations in individual_generation_scores.items():
    for gen, scores in generations.items():
        individual_generation_q3[(id, gen)] = np.percentile(scores, 75)

# Sort by Q3 value, highest first, and select the top 20
top_20_individual_generations = sorted(individual_generation_q3, key=individual_generation_q3.get, reverse=True)[:40]

# Prepare scores for the top 20 for plotting
top_20_scores = [individual_generation_scores[id][gen] for id, gen in top_20_individual_generations]

# Adjust labels for clarity, indicating both the individual ID and generation
labels = [f'{id[:8]}... Gen {gen}' for id, gen in top_20_individual_generations]

# Generate box and whisker plots for the top 20 individual generations
fig, ax = plt.subplots(figsize=(12, 10))
ax.boxplot(top_20_scores, vert=False, patch_artist=True, labels=labels)
ax.set_xlabel('Scores')
ax.set_ylabel('Individual Generation')
ax.set_title('Top 20 Individual Generations by Q3 Value')

# Display the plot
plt.show()

