import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from graph.network import GraphDataset
from models.gat import DualHeadGATModel, SosEstimator
from dataset import TofDataset
from logger import log_message, log_image
from RL.env.acoustic_env import  AcousticEnv
from settings import app_settings
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = TofDataset(['train'])
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
num_samples = 4
c_init = 0.8
grid_res = 1
x_range = (0, 127)
y_range = (0, 127)
mesh_node_k=9


val_loader = DataLoader(dataset, batch_size=1, shuffle=False)


def get_cmap(positions, cmap):
    """
    Extract speed of sound values from cmap that correspond to mesh node positions.
    
    Args:
        cmap: 128x128 matrix of speed of sound values
        
    Returns:
        numpy array of speed values in the same order as mesh nodes
    """
    # Create the same grid of positions as in GraphDataset._create_interior_mesh
    extracted_values = []
    for x, y in positions:
            # Convert float positions to integer indices for cmap
        x_idx = int(x)
        y_idx = int(y)
        extracted_values.append(cmap[y_idx, x_idx])
    return np.array(extracted_values)


for batch in val_loader:
    sos = batch['sos'].squeeze()
    config = {
        'c_init': c_init,
        'sources_positions': batch['x_s'].squeeze().float(),
        'receivers_positions' : batch['x_r'].squeeze().float(),
        'tof_matrix' : batch['raw_tof'].squeeze().float(),
        'selected_sources': [1,8,16,24],
        'full_mesh_resolution':(app_settings.anatomy_height, app_settings.anatomy_width),
    }
    gd = GraphDataset(c_init=c_init, x_range=x_range, y_range=y_range, nx=grid_res, ny=grid_res, mesh_node_k=mesh_node_k)
    env = AcousticEnv(config=config,graph_dataset=gd)
    env.reset()
    cmap = get_cmap(gd.positions, sos)
    c_map = torch.tensor(cmap, dtype=torch.float32, device=device)
    for src_id in config['selected_sources']:
        observation, reward, done, info = env.test_cmap(c_map, src_id)
    exit()





