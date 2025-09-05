# Re-importing necessary libraries
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.ticker as ticker

# Simplified JSON data for demonstration
with open('gemla/round4.json', 'r') as file:
    simplified_json_data = json.load(file)

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

# Getting most recent right graph
right_nodes = traverse_right_nodes(simplified_json_data[0])

# Heatmaps
# Data structure to store mutation rates, generations, and scores
mutation_rate_data = defaultdict(lambda: defaultdict(list))

# Populate the dictionary with scores indexed by mutation rate and generation
for node in right_nodes:
    node_val = node["val"]["node"]
    if node_val:
        scores = node_val["scores"]
        minor_mutation_rate = node_val["minor_mutation_rate"]
        generation = node_val["generation"]
        # Ensure each score is associated with the correct generation
        for gen_index, score_list in enumerate(scores):
            for score in score_list.values():
                mutation_rate_data[minor_mutation_rate][gen_index].append(score)

# Prepare data for heatmap
max_generation = max(max(gens.keys()) for gens in mutation_rate_data.values())
heatmap_data = np.full((len(mutation_rate_data), max_generation + 1), np.nan)

# Populate the heatmap data with average scores
mutation_rates = sorted(mutation_rate_data.keys())
for i, mutation_rate in enumerate(mutation_rates):
    for generation in range(max_generation + 1):
        scores = mutation_rate_data[mutation_rate][generation]
        if scores:  # Check if there are scores for this generation
            heatmap_data[i, generation] = np.mean(scores)

# Creating a DataFrame for the heatmap
df_heatmap = pd.DataFrame(
    data=heatmap_data, 
    index=mutation_rates, 
    columns=range(max_generation + 1)
)

# Data structure to store major mutation rates, generations, and scores
major_mutation_rate_data = defaultdict(lambda: defaultdict(list))

# Populate the dictionary with scores indexed by major mutation rate and generation
# This is assuming the structure to retrieve major_mutation_rate is similar to minor_mutation_rate
for node in right_nodes:
    node_val = node["val"]["node"]
    if node_val:
        scores = node_val["scores"]
        major_mutation_rate = node_val["major_mutation_rate"]
        generation = node_val["generation"]
        for gen_index, score_list in enumerate(scores):
            for score in score_list.values():
                major_mutation_rate_data[major_mutation_rate][gen_index].append(score)

# Prepare the heatmap data for major_mutation_rate similar to minor_mutation_rate
major_heatmap_data = np.full((len(major_mutation_rate_data), max_generation + 1), np.nan)
major_mutation_rates = sorted(major_mutation_rate_data.keys())

for i, major_rate in enumerate(major_mutation_rates):
    for generation in range(max_generation + 1):
        scores = major_mutation_rate_data[major_rate][generation]
        if scores:  # Check if there are scores for this generation
            major_heatmap_data[i, generation] = np.mean(scores)

# Creating a DataFrame for the major mutation rate heatmap
df_major_heatmap = pd.DataFrame(
    data=major_heatmap_data,
    index=major_mutation_rates,
    columns=range(max_generation + 1)
)

# crossbreed_segments
# Data structure to store major mutation rates, generations, and scores
crossbreed_segments_data = defaultdict(lambda: defaultdict(list))

# Populate the dictionary with scores indexed by major mutation rate and generation
# This is assuming the structure to retrieve major_mutation_rate is similar to minor_mutation_rate
for node in right_nodes:
    node_val = node["val"]["node"]
    if node_val:
        scores = node_val["scores"]
        crossbreed_segments = node_val["crossbreed_segments"]
        generation = node_val["generation"]
        for gen_index, score_list in enumerate(scores):
            for score in score_list.values():
                crossbreed_segments_data[crossbreed_segments][gen_index].append(score)

# Prepare the heatmap data for crossbreed_segments similar to minor_mutation_rate
crossbreed_heatmap_data = np.full((len(crossbreed_segments_data), max_generation + 1), np.nan)
crossbreed_segments = sorted(crossbreed_segments_data.keys())

for i, crossbreed_segment in enumerate(crossbreed_segments):
    for generation in range(max_generation + 1):
        scores = crossbreed_segments_data[crossbreed_segment][generation]
        if scores:  # Check if there are scores for this generation
            crossbreed_heatmap_data[i, generation] = np.mean(scores)

# Creating a DataFrame for the major mutation rate heatmap
df_crossbreed_heatmap = pd.DataFrame(
    data=crossbreed_heatmap_data,
    index=crossbreed_segments,
    columns=range(max_generation + 1)
)

# mutation_weight_range
# Data structure to store major mutation rates, generations, and scores
mutation_weight_range_data = defaultdict(lambda: defaultdict(list))

# Populate the dictionary with scores indexed by major mutation rate and generation
# This is assuming the structure to retrieve major_mutation_rate is similar to minor_mutation_rate
for node in right_nodes:
    node_val = node["val"]["node"]
    if node_val:
        scores = node_val["scores"]
        mutation_weight_range = node_val["mutation_weight_range"]
        positive_extent = mutation_weight_range["end"]
        negative_extent = -mutation_weight_range["start"]
        mutation_weight_range = (positive_extent + negative_extent) / 2
        generation = node_val["generation"]
        for gen_index, score_list in enumerate(scores):
            for score in score_list.values():
                mutation_weight_range_data[mutation_weight_range][gen_index].append(score)

# Prepare the heatmap data for crossbreed_segments similar to minor_mutation_rate
mutation_weight_range_heatmap_data = np.full((len(mutation_weight_range_data), max_generation + 1), np.nan)
mutation_weight_ranges = sorted(mutation_weight_range_data.keys())

for i, mutation_weight_range in enumerate(mutation_weight_ranges):
    for generation in range(max_generation + 1):
        scores = mutation_weight_range_data[mutation_weight_range][generation]
        if scores:  # Check if there are scores for this generation
            mutation_weight_range_heatmap_data[i, generation] = np.mean(scores)

# Creating a DataFrame for the major mutation rate heatmap
df_mutation_weight_range_heatmap = pd.DataFrame(
    data=mutation_weight_range_heatmap_data,
    index=mutation_weight_ranges,
    columns=range(max_generation + 1)
)

# weight_initialization_range
# Data structure to store major mutation rates, generations, and scores
weight_initialization_range_data = defaultdict(lambda: defaultdict(list))

# Populate the dictionary with scores indexed by major mutation rate and generation
# This is assuming the structure to retrieve major_mutation_rate is similar to minor_mutation_rate
for node in right_nodes:
    node_val = node["val"]["node"]
    if node_val:
        scores = node_val["scores"]
        weight_initialization_range = node_val["weight_initialization_range"]
        positive_extent = weight_initialization_range["end"]
        negative_extent = -weight_initialization_range["start"]
        weight_initialization_range = (positive_extent + negative_extent) / 2
        generation = node_val["generation"]
        for gen_index, score_list in enumerate(scores):
            for score in score_list.values():
                weight_initialization_range_data[weight_initialization_range][gen_index].append(score)

# Prepare the heatmap data for crossbreed_segments similar to minor_mutation_rate
weight_initialization_range_heatmap_data = np.full((len(weight_initialization_range_data), max_generation + 1), np.nan)
weight_initialization_ranges = sorted(weight_initialization_range_data.keys())

for i, weight_initialization_range in enumerate(weight_initialization_ranges):
    for generation in range(max_generation + 1):
        scores = weight_initialization_range_data[weight_initialization_range][generation]
        if scores:  # Check if there are scores for this generation
            weight_initialization_range_heatmap_data[i, generation] = np.mean(scores)

# Creating a DataFrame for the major mutation rate heatmap
df_weight_initialization_range_heatmap = pd.DataFrame(
    data=weight_initialization_range_heatmap_data,
    index=weight_initialization_ranges,
    columns=range(max_generation + 1)
)

# weight_initialization_range_skew
# Data structure to store major mutation rates, generations, and scores
weight_initialization_range_skew_data = defaultdict(lambda: defaultdict(list))

# Populate the dictionary with scores indexed by major mutation rate and generation
# This is assuming the structure to retrieve major_mutation_rate is similar to minor_mutation_rate
for node in right_nodes:
    node_val = node["val"]["node"]
    if node_val:
        scores = node_val["scores"]
        weight_initialization_range = node_val["weight_initialization_range"]
        positive_extent = weight_initialization_range["end"]
        negative_extent = -weight_initialization_range["start"]
        weight_initialization_range_skew = (positive_extent - negative_extent) / 2
        generation = node_val["generation"]
        for gen_index, score_list in enumerate(scores):
            for score in score_list.values():
                weight_initialization_range_skew_data[weight_initialization_range_skew][gen_index].append(score)

# Prepare the heatmap data for crossbreed_segments similar to minor_mutation_rate
weight_initialization_range_skew_heatmap_data = np.full((len(weight_initialization_range_skew_data), max_generation + 1), np.nan)
weight_initialization_range_skews = sorted(weight_initialization_range_skew_data.keys())

for i, weight_initialization_range_skew in enumerate(weight_initialization_range_skews):
    for generation in range(max_generation + 1):
        scores = weight_initialization_range_skew_data[weight_initialization_range_skew][generation]
        if scores:  # Check if there are scores for this generation
            weight_initialization_range_skew_heatmap_data[i, generation] = np.mean(scores)

# Creating a DataFrame for the major mutation rate heatmap
df_weight_initialization_range_skew_heatmap = pd.DataFrame(
    data=weight_initialization_range_skew_heatmap_data,
    index=weight_initialization_range_skews,
    columns=range(max_generation + 1)
)

# Analyze number of neurons correlation to score
# We can get the number of neurons via node_val["nn_shapes"] which contains an array of maps
# Each map has a key for the individual id and a value which is an array of integers representing the number of neurons in each layer
# We can use the individual id to get the score from the scores array
# We then generate a density map of the number of neurons vs the score
neuron_number_score_data = defaultdict(lambda: defaultdict(list))

for node in right_nodes:
    node_val = node["val"]["node"]
    if node_val:
        scores = node_val["scores"]
        nn_shapes = node_val["nn_shapes"]
        # Both scores and nn_shapes are arrays where score is 1 less in length than nn_shapes (each index corresponds to a generation)
        for gen_index, score in enumerate(scores):
            for individual_id, nn_shape in nn_shapes[gen_index].items():
                neuron_number = sum(nn_shape)
                # check if score has a value for the individual id
                if individual_id not in score:
                    continue
                neuron_number_score_data[neuron_number][gen_index].append(score[individual_id])

# prepare the density map data
neuron_number_score_heatmap_data = np.full((len(neuron_number_score_data), max_generation + 1), np.nan)
neuron_numbers = sorted(neuron_number_score_data.keys())

for i, neuron_number in enumerate(neuron_numbers):
    for generation in range(max_generation + 1):
        scores = neuron_number_score_data[neuron_number][generation]
        if scores:  # Check if there are scores for this generation
            neuron_number_score_heatmap_data[i, generation] = np.mean(scores)

# Creating a DataFrame for the major mutation rate heatmap
df_neuron_number_score_heatmap = pd.DataFrame(
    data=neuron_number_score_heatmap_data,
    index=neuron_numbers,
    columns=range(max_generation + 1)
)

# Analyze number of layers correlation to score
nn_layers_score_data = defaultdict(lambda: defaultdict(list))

for node in right_nodes:
    node_val = node["val"]["node"]
    if node_val:
        scores = node_val["scores"]
        nn_shapes = node_val["nn_shapes"]
        # Both scores and nn_shapes are arrays where score is 1 less in length than nn_shapes (each index corresponds to a generation)
        for gen_index, score in enumerate(scores):
            for individual_id, nn_shape in nn_shapes[gen_index].items():
                layer_number = len(nn_shape)
                # check if score has a value for the individual id
                if individual_id not in score:
                    continue
                nn_layers_score_data[layer_number][gen_index].append(score[individual_id])

# prepare the density map data
nn_layers_score_heatmap_data = np.full((len(nn_layers_score_data), max_generation + 1), np.nan)
nn_layers = sorted(nn_layers_score_data.keys())

for i, nn_layer in enumerate(nn_layers):
    for generation in range(max_generation + 1):
        scores = nn_layers_score_data[nn_layer][generation]
        if scores:  # Check if there are scores for this generation
            nn_layers_score_heatmap_data[i, generation] = np.mean(scores)

# Creating a DataFrame for the major mutation rate heatmap
df_nn_layers_score_heatmap = pd.DataFrame(
    data=nn_layers_score_heatmap_data,
    index=nn_layers,
    columns=range(max_generation + 1)
)

# print("Format: ", custom_formatter(0.123498761234, 0))

# Creating subplots
fig, axs = plt.subplots(2, 2, figsize=(20, 14))  # Creates a 3x2 grid of subplots

# Plotting the minor mutation rate heatmap
sns.heatmap(df_heatmap.T, cmap='viridis', fmt=".4g", cbar_kws={'label': 'Mean Score'}, ax=axs[0, 0])
# axs[0, 0].set_title('Minor Mutation Rate')
axs[0, 0].set_xlabel('Minor Mutation Rate')
axs[0, 0].set_ylabel('Generation')
axs[0, 0].invert_yaxis()

# Plotting the major mutation rate heatmap
sns.heatmap(df_major_heatmap.T, cmap='viridis', fmt=".4g", cbar_kws={'label': 'Mean Score'}, ax=axs[0, 1])
# axs[0, 1].set_title('Major Mutation Rate')
axs[0, 1].set_xlabel('Major Mutation Rate')
axs[0, 1].invert_yaxis()

# Plotting the crossbreed_segments heatmap
sns.heatmap(df_crossbreed_heatmap.T, cmap='viridis', fmt=".4g", cbar_kws={'label': 'Mean Score'}, ax=axs[1, 0])
# axs[1, 0].set_title('Crossbreed Segments')
axs[1, 0].set_xlabel('Crossbreed Segments')
axs[1, 0].set_ylabel('Generation')
axs[1, 0].invert_yaxis()

# Plotting the mutation_weight_range heatmap
sns.heatmap(df_mutation_weight_range_heatmap.T, cmap='viridis', fmt=".4g", cbar_kws={'label': 'Mean Score'}, ax=axs[1, 1])
# axs[1, 1].set_title('Mutation Weight Range')
axs[1, 1].set_xlabel('Mutation Weight Range')
axs[1, 1].invert_yaxis()

fig3, axs3 = plt.subplots(1, 2, figsize=(20, 14))  # Creates a 3x2 grid of subplots

# Plotting the weight_initialization_range heatmap
sns.heatmap(df_weight_initialization_range_heatmap.T, cmap='viridis', fmt=".4g", cbar_kws={'label': 'Mean Score'}, ax=axs3[0])
# axs[2, 0].set_title('Weight Initialization Range')
axs3[0].set_xlabel('Weight Initialization Range')
axs3[0].set_ylabel('Generation')
axs3[0].invert_yaxis()

# Plotting the weight_initialization_range_skew heatmap
sns.heatmap(df_weight_initialization_range_skew_heatmap.T, cmap='viridis', fmt=".4g", cbar_kws={'label': 'Mean Score'}, ax=axs3[1])
# axs[2, 1].set_title('Weight Initialization Range Skew')
axs3[1].set_xlabel('Weight Initialization Range Skew')
axs3[1].set_ylabel('Generation')
axs3[1].invert_yaxis()

# Creating a new window for the scatter plots
fig2, axs2 = plt.subplots(2, 1, figsize=(20, 14))  # Creates a 2x1 grid of subplots

# Plotting the neuron number vs score heatmap
sns.heatmap(df_neuron_number_score_heatmap.T, cmap='viridis', fmt=".4g", cbar_kws={'label': 'Mean Score'}, ax=axs2[1])
# axs[3, 1].set_title('Neuron Number vs. Score')
axs2[1].set_xlabel('Neuron Number')
axs2[1].set_ylabel('Generation')
axs2[1].invert_yaxis()

# Plotting the number of layers vs score heatmap
sns.heatmap(df_nn_layers_score_heatmap.T, cmap='viridis', fmt=".4g", cbar_kws={'label': 'Mean Score'}, ax=axs2[0])
# axs[3, 1].set_title('Number of Layers vs. Score')
axs2[0].set_xlabel('Number of Layers')
axs2[0].set_ylabel('Generation')
axs2[0].invert_yaxis()

# Display the plot
plt.tight_layout()  # Adjusts the subplots to fit into the figure area.
plt.show()