import matplotlib.pyplot as plt
import networkx as nx
import subprocess
import tkinter as tk
from tkinter import filedialog

def select_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        initialdir="/",  # Set the initial directory to search for files
        title="Select file",
        filetypes=(("Net files", "*.net"), ("All files", "*.*"))
    )
    return file_path

def get_fann_data(network_file):
    # Adjust the path to the Rust executable as needed
    result = subprocess.run(['./extract_fann_data/target/debug/extract_fann_data.exe', network_file], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error:", result.stderr)
        return None, None

    layer_sizes = []
    connections = []
    parsing_connections = False

    for line in result.stdout.splitlines():
        if line.startswith("Layers:"):
            continue
        elif line.startswith("Connections:"):
            parsing_connections = True
            continue

        if parsing_connections:
            from_neuron, to_neuron, weight = map(float, line.split())
            connections.append((int(from_neuron), int(to_neuron), weight))
        else:
            layer_size, bias_count = map(int, line.split())
            layer_sizes.append((layer_size, bias_count))

    return layer_sizes, connections

def visualize_fann_network(network_file):
    # Get network data
    layer_sizes, connections = get_fann_data(network_file)
    if layer_sizes is None or connections is None:
        return  # Error handling in get_fann_data should provide error output

    # Create a directed graph
    G = nx.DiGraph()

    # Positions dictionary to hold the position of each neuron
    pos = {}
    node_count = 0
    x_spacing = 1.0
    y_spacing = 1.0
    
    # Calculate the maximum layer size for proper spacing
    max_layer_size = max(size for size, bias in layer_sizes)
    
    # Build nodes and position them layer by layer from left to right
    for layer_index, (layer_size, bias_count) in enumerate(layer_sizes):
        y_positions = list(range(-layer_size-bias_count+1, 1, 1))  # Center-align vertically
        y_positions = [y * (max_layer_size / (layer_size + bias_count)) * y_spacing for y in y_positions]  # Adjust spacing
        for neuron_index in range(layer_size + bias_count):  # Include bias neurons
            node_label = f"L{layer_index}N{neuron_index}"
            G.add_node(node_count, label=node_label)
            pos[node_count] = (layer_index * x_spacing, y_positions[neuron_index % len(y_positions)])
            node_count += 1

    # Add connections to the graph
    for from_neuron, to_neuron, weight in connections:
        G.add_edge(from_neuron, to_neuron, weight=weight)

    max_weight = max(abs(weight) for _, _, weight in connections)
    print(f"Max weight: {max_weight}")

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=200)
    nx.draw_networkx_labels(G, pos, font_size=7)

    # Custom function for edge properties
    def adjust_properties(weight):
        # if weight > 0:
        # print("Weight:", weight)
        color = 'green' if weight > 0 else 'red'
        alpha = min((abs(weight) / max_weight) ** 3, 1)
        # print(f"Color: {color}, Alpha: {alpha}")
        return color, alpha

    # Draw edges with custom properties
    for u, v, d in G.edges(data=True):
        color, alpha = adjust_properties(d['weight'])
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=color, alpha=alpha, width=1.5, arrows=False)

    # Show plot
    plt.title('FANN Network Visualization')
    plt.axis('off')  # Turn off the axis
    plt.show()

# Path to the FANN network file
fann_path = 'F:\\\\vandomej\\Projects\\dootcamp-AI-Simulation\\Simulations\\fighter_nn_4f2be613-ab26-4384-9a65-450e043984ea\\6\\4f2be613-ab26-4384-9a65-450e043984ea_fighter_nn_0.net'
# fann_path = "F:\\\\vandomej\\Projects\\dootcamp-AI-Simulation\\Simulations\\fighter_nn_fc294503-7b2a-40f8-be59-ccc486eb3f79\\0\\fc294503-7b2a-40f8-be59-ccc486eb3f79_fighter_nn_0.net"
# fann_path = 'F:\\\\vandomej\\Projects\\dootcamp-AI-Simulation\\Simulations\\fighter_nn_99c30a7f-40ab-4faf-b16a-b44703fdb6cd\\0\\99c30a7f-40ab-4faf-b16a-b44703fdb6cd_fighter_nn_0.net'
# Has a 4 layer network
# # Generation 1
# fann_path = "F:\\\\vandomej\\Projects\\dootcamp-AI-Simulation\\Simulations\\fighter_nn_16dfa1b4-03c7-45a6-84b4-22fe3c8e2d98\\1\\16dfa1b4-03c7-45a6-84b4-22fe3c8e2d98_fighter_nn_0.net"
# # Generation 5
# fann_path = "F:\\\\vandomej\\Projects\\dootcamp-AI-Simulation\\Simulations\\fighter_nn_16dfa1b4-03c7-45a6-84b4-22fe3c8e2d98\\5\\16dfa1b4-03c7-45a6-84b4-22fe3c8e2d98_fighter_nn_0.net"
# # Generation 10
# fann_path = "F:\\\\vandomej\\Projects\\dootcamp-AI-Simulation\\Simulations\\fighter_nn_16dfa1b4-03c7-45a6-84b4-22fe3c8e2d98\\10\\16dfa1b4-03c7-45a6-84b4-22fe3c8e2d98_fighter_nn_0.net"
# # Generation 20
# fann_path = "F:\\\\vandomej\\Projects\\dootcamp-AI-Simulation\\Simulations\\fighter_nn_16dfa1b4-03c7-45a6-84b4-22fe3c8e2d98\\20\\16dfa1b4-03c7-45a6-84b4-22fe3c8e2d98_fighter_nn_0.net"
# # Generation 32
# fann_path = "F:\\\\vandomej\\Projects\\dootcamp-AI-Simulation\\Simulations\\fighter_nn_16dfa1b4-03c7-45a6-84b4-22fe3c8e2d98\\32\\16dfa1b4-03c7-45a6-84b4-22fe3c8e2d98_fighter_nn_0.net"
fann_path = select_file()
visualize_fann_network(fann_path)