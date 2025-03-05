from graph.network import GraphDataset
from dataset import TofDataset
from torch.utils.data import DataLoader
import torch
import random

train_dataset = TofDataset(['train'])
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'



gd = GraphDataset(nx=3, ny=3)
for batch in train_loader:
    sources_positions = batch['x_s'].squeeze()
    receivers_positions = batch['x_r'].squeeze()
    tof = batch['raw_tof'].squeeze()

    gd.build(sources_positions, receivers_positions)
    #print('positions')
    #print(gd.positions)
    #print('edges')
    #print(gd.edges)
    exit()
    print(f"positions: {gd.positions.shape}")
    print(f"edges: {gd.edges.shape}")
    selected_sources = random.choices(range(32), k=5)
    for i, data in gd.get_graph(tof,selected_sources, device):
        print(f"features: {data.x.shape}")




