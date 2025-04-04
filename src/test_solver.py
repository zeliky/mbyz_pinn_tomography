import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from graph.network import GraphDataset
from models.gat import DualHeadGATModel, SosEstimator
from dataset import TofDataset
from logger import log_message, log_image
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = TofDataset(['train'])
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
num_samples = 4
c_init = 0.15
grid_res = 8
x_range = (32, 96)
y_range = (32, 96)
fmm_iterations = 20


estimator = SosEstimator(num_nodes=grid_res * grid_res + 64, init_value=c_init, fmm_iterations=fmm_iterations)
gd = GraphDataset(c_init=c_init, x_range=x_range, y_range=y_range, nx=grid_res, ny=grid_res)


def visualize_tof(self, num_samples=5):
    val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False)
    self.model.eval()
    with torch.no_grad():
        count = 0
        for batch in val_loader:
            tofs_true, tofs_pred = self.training_step_handler.eval_tof(batch)

            for tof_pred, tof_true in zip(tofs_pred, tofs_true):
                fig, axs = plt.subplots(1, 2, figsize=(8, 4))

                axs[0].imshow(tof_true.squeeze(0), cmap='jet')
                axs[0].set_title('TOF True')
                axs[0].axis('off')

                axs[1].imshow(tof_pred.squeeze(0), cmap='jet')
                axs[1].set_title('Predicted TOF')
                axs[1].axis('off')

                # plt.tight_layout()
                # plt.show()
                log_image(fig)
                log_message(' ')
                count += 1
                if count >= num_samples:
                    return


def visualize_sos(anatomy, tof, c_pred):
    # tof_np = tof[i].cpu().numpy()
    tof_np = tof.cpu()
    anatomy_np = anatomy.numpy().squeeze(0).squeeze(0)


    # Plot anatomy and c_pred side by side
    fig, axs = plt.subplots(1, 3, figsize=(8, 4))

    axs[0].imshow(tof_np, cmap='jet')
    axs[0].set_title('TOF')
    axs[0].axis('off')

    axs[1].imshow(anatomy_np, cmap='jet')
    axs[1].set_title('Original Anatomy')
    axs[1].axis('off')

    axs[2].imshow(c_pred, cmap='jet')
    axs[2].set_title('Predicted SoS (c_pred)')
    axs[2].axis('off')

    # plt.tight_layout()
    # plt.show()
    log_image(fig)
    log_message(' ')


count = 0
for batch in data_loader:
    sources_positions = batch['x_s'].squeeze()
    receivers_positions = batch['x_r'].squeeze()
    tof = batch['raw_tof'].squeeze().float().to(device)
    anatomy = batch['anatomy'].cpu()
    s_count, _ = sources_positions.shape
    r_count, _ = receivers_positions.shape
    transmitters_indices = torch.arange(0, s_count, device=device)
    receiver_indices = torch.arange(s_count, s_count + r_count, device=device)
    if not gd.initialized:
        gd.build(sources_positions, receivers_positions)

    selected_sources = random.sample(range(32), k=32)
    estimator.reset(device)
    c_pred = None
    for i, data in gd.get_graph(tof, selected_sources, device):
        T, c = estimator.estimate(data.x, data.edge_index, data.pos, transmitters_indices, receiver_indices)

    c_pred = estimator.get_sos(128, 128)
    visualize_sos(anatomy, tof, c_pred)
    count += 1
    if count >= num_samples:
        break
