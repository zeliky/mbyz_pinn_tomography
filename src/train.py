import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from logger import log_message, log_image
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
        self.pde_weight = kwargs.get('pde_weight',  1.0)
        self.bc_weight = kwargs.get('bc_weight',  1.0)

    def train_model(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        mse_criterion = nn.MSELoss()

        model = self.model.to(self.device)

        #coords= self.get_domain_coords()

        for epoch in range(self.epochs):
            model.train()
            running_loss = 0.0
            with tqdm(total=train_loader.__len__()) as pbar:
                pbar.set_description("Training " )
                for batch in train_loader:
                    # batch contains: 'anatomy', 'tof', 'x_s', 'x_r', 'x_o'
                    inputs = batch['inputs'].to(self.device)  # Ground truth SoS
                    tof = batch['tof'].to(self.device)          # Measured TOF field
                    anatomy = batch['anatomy'].to(self.device)  # Original anatomy SOS
                    x_s = batch['x_s'].to(self.device)          # Source positions [num_pairs,2]
                    x_r = batch['x_r'].to(self.device)          # Receiver positions [num_pairs,2]
                    x_o = batch['x_o'].to(self.device)          # Measured travel times for each pair
                    source_map = inputs[:, 1:2, :, :]
                    receiver_map = inputs[:, 2:3, :, :]

                    t_pred, c_pred = model(inputs)

                    #  compute physics loss tof/sos
                    pde_residual = model.compute_pde_residual(t_pred, c_pred)
                    pde_loss = torch.mean(pde_residual**2)

                    # loss between predicted SoS and anatomy sos
                    #pred_image = c_pred.squeeze(0).squeeze(0)
                    #org_image = anatomy.squeeze(0).squeeze(0)
                    mse_loss = mse_criterion(c_pred, anatomy)

                    # Boundary conditions loss:
                    B, _, H, W = inputs.shape
                    T_s = t_pred[source_map == 1]
                    #T_r = t_pred[receiver_map == 1]

                    # T_s should be ~0
                    bc_loss_s = mse_criterion(T_s, torch.zeros_like(T_s))

                    # T_r should match x_o
                    #print(f"T_s:{T_s.shape} ,T_r:{T_r.shape}, x_o:{x_o.shape}")
                    #x_o = x_o.unsqueeze(-1) # [B, num_pairs, 1]

                    #bc_loss_r = mse_criterion(T_r, x_o)
                    #bc_loss = bc_loss_s + bc_loss_r
                    bc_loss = bc_loss_s

                    #print(f"s:{bc_loss_s}, r:{bc_loss_r}")
                    # Total loss
                    loss = mse_loss + self.pde_weight * pde_loss + self.bc_weight * bc_loss

                    #print(f"mse_loss:{mse_loss} , pde_loss:{pde_loss}, bc_loss:{bc_loss}")
                    optimizer.zero_grad( )
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    pbar.update(1)

            val_loss = 0.0
            """
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                with tqdm(total=val_loader.__len__()) as pbar:
                    pbar.set_description("Validation ")
                    for batch in val_loader:
                        inputs = batch['inputs'].to(self.device)
                        anatomy = batch['anatomy'].to(self.device)
                        x_s = batch['x_s'].to(self.device)
                        x_r = batch['x_r'].to(self.device)
                        x_o = batch['x_o'].to(self.device)

                        t_pred, c_pred = self.model(inputs)

                        pde_residual = self.model.compute_pde_residual(t_pred, c_pred)
                        pde_loss = torch.mean(pde_residual**2)

                        pred_image = c_pred.squeeze(0).squeeze(0)
                        org_image = anatomy.squeeze(0).squeeze(0)
                        mse_loss = mse_criterion(pred_image, org_image)

                        T_s = _bilinear_interpolate(t_pred, _to_pixel_coordinates(x_s,H,W))
                        T_r = _bilinear_interpolate(t_pred, _to_pixel_coordinates(x_r,H,W))
                        bc_loss_s = mse_criterion(T_s, torch.zeros_like(T_s))
                        x_o = x_o.unsqueeze(-1)
                        bc_loss_r = mse_criterion(T_r, x_o)
                        bc_loss = bc_loss_s + bc_loss_r

                        loss = mse_loss + self.pde_weight * pde_loss + self.bc_weight * bc_loss
                        val_loss += loss.item()
                        pbar.update(1)
            """
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