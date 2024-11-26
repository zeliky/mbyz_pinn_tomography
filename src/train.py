import torch
import torch.optim as optim
from torch.utils.data import DataLoader,random_split
from model import PINNModel
from dataset import TofDataset
from physics import PINNLoss
from logger import log_message
from report_dataset_info import report_dataset_info
from visualization import visualize_tof_image, visualize_anatomy_image, visualize_sources_and_receivers


def train_model(dataset, num_epochs=100, batch_size=16, learning_rate=1e-4, physics_loss_weight=1.0, L_x=0.1, L_y=0.1):
    log_message("[train.py] Initializing training...")
    log_message(' ')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PINNModel().to(device)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define the custom loss function
    loss_fn = PINNLoss(physics_loss_weight, L_x, L_y)

    log_message("[train.py] Starting training loop...")
    log_message(' ')

    for epoch in range(num_epochs):
        model.train()
        total_loss_epoch = 0.0
        total_data_loss_epoch = 0.0
        total_physics_loss_epoch = 0.0
        num_batches = 0

        for batch in data_loader:
            # Move data to device
            tof_input = batch['tof_input'].to(device)  # Shape: [batch_size, 1, 32, 32]
            tof_data = batch['tof_data']
            x_r = tof_data['x_r'].to(device)  # Shape: [batch_size, 2]
            x_s = tof_data['x_s'].to(device)  # Shape: [batch_size, 2]
            observed_tof = tof_data['x_o'].to(device)  # Shape: [batch_size]

            batch_size = tof_input.shape[0]
            num_points = 1024  # Number of points for physics loss

            # Ensure x_s has the correct shape
            if x_s.dim() == 1:
                x_s = x_s.unsqueeze(0)  # Shape: [1, 2]
            x_s = x_s.expand(batch_size, -1)  # Shape: [batch_size, 2]

            x_coords = torch.rand((batch_size, num_points, 2), device=device)
            x_coords[:, :, 0] *= L_x  # Scale to physical units
            x_coords[:, :, 1] *= L_y  # Scale to physical units
            x_coords.requires_grad = True

            x_s_grid = x_s.unsqueeze(1).repeat(1, num_points, 1)  # Shape: [batch_size, num_points, 2]

            optimizer.zero_grad()

            # Compute the loss
            total_loss, data_loss_value, physics_loss_value = loss_fn(
                model, tof_input, x_r, x_s, observed_tof, x_coords, x_s_grid
            )

            # Backpropagation and optimization
            total_loss.backward()
            optimizer.step()

            # Accumulate losses
            total_loss_epoch += total_loss.item()
            total_data_loss_epoch += data_loss_value
            total_physics_loss_epoch += physics_loss_value
            num_batches += 1

        # Calculate average losses
        avg_total_loss = total_loss_epoch / num_batches
        avg_data_loss = total_data_loss_epoch / num_batches
        avg_physics_loss = total_physics_loss_epoch / num_batches

        # Print epoch statistics
        log_message(f'Epoch [{epoch + 1}/{num_epochs}], Total Loss: {avg_total_loss:.6f}, '
                    f'Data Loss: {avg_data_loss:.6f}, Physics Loss: {avg_physics_loss:.6f}')

    return model