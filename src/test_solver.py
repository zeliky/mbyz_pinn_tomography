import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from graph.network import GraphDataset
from models.gat import DualHeadGATModel, SosEstimator
from dataset import TofDataset
from logger import log_message, log_image
from RL.env.acoustic_env import  AcousticEnv
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = TofDataset(['train'])
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
num_samples = 4
c_init = 1.2
grid_res = 2
x_range = (32, 96)
y_range = (32, 96)
mesh_node_k=20


val_loader = DataLoader(dataset, batch_size=1, shuffle=False)


def get_mesh_cmap(cmap):
    global  grid_res
    row_indices = np.arange(x_range[0], x_range[1], grid_res)  # Start from 32 to 96 with a step of 8
    col_indices = np.arange(y_range[0], y_range[1], grid_res)

    extracted_values = []
    for row in row_indices:
        for col in col_indices:
            extracted_values.append(cmap[row, col])  # Adjust indices for the sub-array poi_cmap
    return np.array(extracted_values)


for batch in val_loader:
    cmap = batch['sos'].squeeze()
    config = {
        'sources_positions': batch['x_s'].squeeze().float(),
        'receivers_positions' : batch['x_r'].squeeze().float(),
        'tof_matrix' : batch['raw_tof'].squeeze().float(),
        'selected_sources': [20,15,6]
    }
    gd = GraphDataset(c_init=c_init, x_range=x_range, y_range=y_range, nx=grid_res, ny=grid_res, mesh_node_k=mesh_node_k)
    env = AcousticEnv(config=config,graph_dataset=gd)
    env.reset()
    mesh_cmap = get_mesh_cmap(cmap)
    for src_id in config['selected_sources']:
        observation, reward, done, info = env.test_cmap(mesh_cmap,src_id)
    exit()





