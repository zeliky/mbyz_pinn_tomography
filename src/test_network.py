from graph.network import GraphDataset
from dataset import TofDataset
from torch.utils.data import DataLoader
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from logger import log_message, log_image
import torch
import random


train_dataset = TofDataset(['train'])
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'



gd = GraphDataset(nx=8, ny=8, mnc=20)
for batch in train_loader:
    sources_positions = batch['x_s'].squeeze()
    receivers_positions = batch['x_r'].squeeze()
    tof = batch['raw_tof'].squeeze()

    gd.build(sources_positions, receivers_positions)
    #print('positions')
    #print(gd.positions)
    #print('edges')
    #print(gd.edges)

    print(f"positions: {gd.positions.shape}")

    selected_sources = random.choices(range(32), k=1)
    for i, data in gd.get_graph(tof,selected_sources, device):
        print(f"features: {data.x.shape}")

        graph_nx = to_networkx(data)
        fig= plt.figure(figsize=(8, 8))
        nx.draw(graph_nx, pos=data.pos.cpu().numpy(), node_size=10)
        log_message(f'Source #{i}')
        print(f"edges: {data.edge_index.shape}")
        log_image(fig)

    break




def plot_attention_graph(data, attn_weights, node_idx):
    """
    Visualizes message passing from a given node_idx in the graph.
    - data: PyG data object (graph)
    - attn_weights: Attention scores from GAT
    - node_idx: The node to trace message passing from
    """
    G = to_networkx(data, to_undirected=True)

    # Get edge weights (attention scores)
    edge_weights = {tuple(edge): weight.item() for edge, weight in zip(data.edge_index.T.tolist(), attn_weights)}

    # Plot graph
    fg = plt.figure(figsize=(8, 8))
    pos = {i: (data.pos[i, 0].item(), data.pos[i, 1].item()) for i in range(data.pos.shape[0])}
    nx.draw(G, pos, node_color="lightblue", edge_color="gray", alpha=0.3, node_size=30)

    # Highlight edges from the selected node
    for edge, weight in edge_weights.items():
        if node_idx in edge:
            nx.draw_networkx_edges(G, pos, edgelist=[edge], width=2, edge_color="red" if weight > 0.5 else "orange")

    log_image(fg)


def trace_message_flow(model, data, start_node, end_node):
    """
    Tracks how information flows from `start_node` to `end_node`.
    """
    # Initialize x with one-hot at start_node
    x = torch.zeros_like(data.x)
    x[start_node, 0] = 1.0  # Inject signal at start_node

    # Forward pass
    pred, attn1, attn2 = model(x, data.edge_index)

    # Check output at end_node
    print(f"Information received at node {end_node}: {pred[end_node, 0].item()}")

#trace_message_flow(model, data, start_node=0, end_node=50)