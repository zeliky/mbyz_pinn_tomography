import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from logger import log_message, log_image
from physics import compute_eikonal_loss
from settings import app_settings
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt


class PINNTrainer:
    def __init__(self, model,train_dataset,val_dataset,  **kwargs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.epochs = kwargs.get('epochs', 10)
        self.batch_size = kwargs.get('batch_size', 1)
        self.lr = kwargs.get('lr',  1e-3)
        self.data_weight = kwargs.get('data_weight',  1e-4)
        self.pde_weight = kwargs.get('pde_weight',  1.0)
        self.bc_weight = kwargs.get('bc_weight',  1e-4)

    def train_model(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        model = self.model.to(self.device)

        for epoch in range(self.epochs):
            model.train()
            running_loss = 0.0
            with tqdm(total=train_loader.__len__()) as pbar:
                pbar.set_description("Training" )
                optimizer.zero_grad()

                for batch in train_loader:
                    input_tof_tensor = batch['tof_inputs'].to(self.device)
                    observed_tof = batch['tof_inputs'].clone().to(self.device)

                    boundary_indices = batch['boundary_indices'].to(self.device)
                    sos_map = batch['anatomy'].to(self.device) # only one anatomy image
                    predicted_tof = model(input_tof_tensor)

                    predicted_boundary = predicted_tof.masked_select(boundary_indices)
                    observed_boundary = observed_tof.masked_select(boundary_indices)
                    boundary_loss = F.mse_loss(predicted_boundary, observed_boundary)

                    eikonal_loss = compute_eikonal_loss(predicted_tof.squeeze(), sos_map.squeeze())
                    #print(f"pde_loss:{eikonal_loss}  bc_loss:{boundary_loss}")

                    #total_loss =  self.pde_weight * eikonal_loss + self.bc_weight * boundary_loss
                    total_loss =  self.pde_weight * eikonal_loss


                    # Backward pass and optimization
                    total_loss.backward()

                    # Gradient Clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    pbar.update(1)
            print(f"Epoch {epoch}, Loss: {total_loss.item()}")


            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                with tqdm(total=val_loader.__len__()) as pbar:
                    pbar.set_description("Validation ")
                    for batch in val_loader:
                        input_tof_tensor = batch['tof_inputs'].to(self.device)
                        observed_tof = batch['tof_inputs'].copy().to(self.device)
                        known_indices = batch['known_indices'].to(self.device)
                        boundary_indices = batch['boundary_indices'].to(self.device)
                        sos_map = batch['anatomy'].to(self.device)
                        predicted_tof = model(input_tof_tensor)
                        data_loss = F.mse_loss(predicted_tof[:, known_indices], observed_tof[:, known_indices])
                        boundary_loss = F.mse_loss(predicted_tof[:, boundary_indices],
                                                   input_tof_tensor[:, boundary_indices])
                        eikonal_loss = compute_eikonal_loss(predicted_tof, sos_map)
                        val_loss = alpha * data_loss + self.pde_weight * eikonal_loss + self.bc_weight * boundary_loss
                        pbar.update(1)

            log_message(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
            self.save_state(model, optimizer, val_loss, epoch)

    def get_domain_coords(self):
        x_min, x_max = 0.0, 1.0
        y_min, y_max = 0.0, 1.0
        num_collocation = 50 * 50
        coords = torch.rand(num_collocation, 2, device=self.device)
        coords[:, 0] = coords[:, 0] * (x_max - x_min) + x_min
        coords[:, 1] = coords[:, 1] * (y_max - y_min) + y_min
        coords.requires_grad_(True)
        return coords

    def load_checkpoint(self, checkpoint_path):
        checkpoint_path = f"{app_settings.output_folder}/{checkpoint_path}"
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])

    def save_state(self, model, optimizer,loss, epoch):
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M_%S_%f")
        save_path = f"{app_settings.output_folder}/pinn_tof-sos_model.{formatted_datetime}-{epoch}.pth"
        torch.save({
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
        }, save_path)
        log_message(f"Model saved to {save_path}")

    def visualize_predictions(self,  num_samples=5):
        val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False)
        self.model.eval()
        with torch.no_grad():
            count = 0
            for batch in val_loader:
                inputs = batch['inputs'].to(self.device)
                anatomy = batch['anatomy'].cpu().squeeze(0).squeeze(0)  # [1,H,W] -> [H,W]

                t, c = self.model(inputs)
                c_pred= c.cpu().squeeze(0).squeeze(0)  # [1,H,W] -> [H,W]
                # Convert to numpy
                anatomy_np = anatomy.numpy()
                c_pred_np = c_pred.numpy()

                # Plot anatomy and c_pred side by side
                fig, axs = plt.subplots(1, 2, figsize=(8, 4))
                axs[0].imshow(anatomy_np, cmap='gray')
                axs[0].set_title('Original Anatomy')
                axs[0].axis('off')

                axs[1].imshow(c_pred_np, cmap='jet')
                axs[1].set_title('Predicted SoS (c_pred)')
                axs[1].axis('off')

                plt.tight_layout()
                plt.show()
                log_image(fig)
                log_message(' ')
                count += 1
                if count >= num_samples:
                    break


def _bilinear_interpolate(tensor, coords):
    # tensor: [B,C,H,W]
    # coords: [num_pairs,2] in pixel coordinates (x,y)
    B, C, H, W = tensor.shape

    # Normalize coordinates to [-1,1]
    coords = coords.squeeze(0)
    x = coords[:, 0]
    y = coords[:, 1]

    norm_x = 2.0 * (x / (W - 1)) - 1.0
    norm_y = 2.0 * (y / (H - 1)) - 1.0

    # Create a grid of shape [1, num_pairs, 1, 2]
    grid = torch.stack([norm_x, norm_y], dim=1).unsqueeze(0).unsqueeze(2)  # [1, num_pairs, 1, 2]

    grid = grid.to(tensor.device)

    # Expand to match batch size: [B, num_pairs, 1, 2]
    grid = grid.expand(B, -1, -1, -1)

    # Now we can sample
    sampled = F.grid_sample(tensor, grid, align_corners=True)  # [B,C,num_pairs,1]

    # Reshape to [B, num_pairs, C]
    sampled = sampled.squeeze(-1).transpose(1, 2)  # [B, num_pairs, C]

    return sampled


def _to_pixel_coordinates(src, W, H):
    norm_pixels = src.clone()
    norm_pixels[:, 0] = norm_pixels[:, 0] * (W - 1)
    norm_pixels[:, 1] = norm_pixels[:, 1] * (H - 1)

    return norm_pixels