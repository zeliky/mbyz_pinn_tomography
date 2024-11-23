import torch
import torch.optim as optim
from torch.utils.data import DataLoader,random_split
from model import UNet
from dataset import TofDataset
from physics import TotalLoss ,compute_eikonal_loss

from report_dataset_info import report_dataset_info
from visualization import visualize_tof_image, visualize_anatomy_image, visualize_sources_and_receivers

def train_model(dataset:TofDataset, num_epochs=100, batch_size=1, learning_rate=1e-4,p=None):
    p.print("[train.py] Initializing training...")
    p.print(' ')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    
    #train_data, val_data = random_split(dataset, [500, 298])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = TotalLoss(lambda_data=1.0, lambda_physics=0.1)

    p.print("[train.py] Starting training loop...")
    p.print(' ')
    loss =None
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in dataloader:

            coords = batch['coords'].view(-1, 2).to(device).requires_grad_(True)  # Flatten to (N, 2)
            sos_values = batch['sos_values'].view(-1).to(device)  # Flatten to (N,)
            source_positions = batch['source_positions'].to(device)
            source_values = batch['source_values'].to(device)
            receiver_positions = batch['receiver_positions'].to(device)
            receiver_values = batch['receiver_values'].to(device)
            print(receiver_values.shape)
            pinn_output = model(receiver_values)

            # Forward pass
            sos_pred = model(receiver_values)

            # Compute loss
            loss = criterion(sos_pred, ssos_values, tof_images)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if loss is not None:
            p.print(f"[train.py] Epoch {epoch + 1}/{num_epochs}, Total Loss: {loss.item():.6f}")

    p.print("[train.py] Training complete.")
    p.print(' ')
