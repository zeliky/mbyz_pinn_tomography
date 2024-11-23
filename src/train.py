import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import EikonalPINN
from dataset import TofDataset
from physics import compute_eikonal_loss

from report_dataset_info import report_dataset_info
from visualization import visualize_tof_image, visualize_anatomy_image, visualize_sources_and_receivers

def train_model(num_epochs=100, batch_size=1, learning_rate=1e-4, p=None):
    p.print("[train.py] Initializing training...")
    p.print(' ')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EikonalPINN().to(device)
    dataset = TofDataset(['train'], p)
    
    report_dataset_info(dataset, p)
    # Visualize the first ToF image
    visualize_tof_image(dataset, idx=0, p=p)
    visualize_anatomy_image(dataset, idx=0, p=p)
    visualize_sources_and_receivers(dataset, idx=0, p=p)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    p.print("[train.py] Starting training loop...")
    p.print(' ')
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

            pinn_output = model(coords)
            source_output = model(source_positions)
            receiver_output = model(receiver_positions)

            physics_loss = compute_eikonal_loss(coords, pinn_output, sos_values)
            source_loss = torch.mean((source_output - source_values) ** 2)
            receiver_loss = torch.mean((receiver_output - receiver_values) ** 2)

            total_batch_loss = physics_loss + source_loss + receiver_loss

            if torch.isnan(total_batch_loss):
                # p.print("[train.py] NaN loss detected; skipping batch.")
                continue

            optimizer.zero_grad()
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += total_batch_loss.item()

        p.print(f"[train.py] Epoch {epoch + 1}/{num_epochs}, Total Loss: {total_loss:.6f}")

    p.print("[train.py] Training complete.")
    p.print(' ')
