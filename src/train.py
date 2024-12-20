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
        criterion = nn.MSELoss()
        num_epochs = self.epochs
        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0
            with tqdm(total=train_loader.__len__()) as pbar:
                pbar.set_description("Training" )
                optimizer.zero_grad()

                for batch in train_loader:
                    tof_tensor  = batch['tof'].to(self.device)
                    sos_tensor = batch['anatomy'].to(self.device)

                    optimizer.zero_grad()
                    sos_pred = model(tof_tensor)
                    loss = criterion(sos_pred, sos_tensor)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    pbar.update(1)
                log_message(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.6f}')

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                with tqdm(total=val_loader.__len__()) as pbar:
                    pbar.set_description("Validation ")
                    for batch in val_loader:
                        tof_tensor = batch['tof'].to(self.device)
                        sos_tensor = batch['anatomy'].to(self.device)
                        sos_pred = model(tof_tensor)
                        loss = criterion(sos_pred, sos_tensor)
                        val_loss += loss.item()
                        pbar.update(1)

                    log_message(f'Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss / len(val_loader):.6f}')
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
        val_loader = DataLoader(self.val_dataset, batch_size=10, shuffle=False)
        self.model.eval()
        with torch.no_grad():
            count = 0
            for batch in val_loader:
                tof = batch['tof'].to(self.device)
                anatomy = batch['anatomy'].cpu()
                c_pred = self.model(tof)
                for i in range(c_pred.size(0)):
                    tof_np = tof[i].cpu().numpy()
                    anatomy_np = anatomy[i].numpy()
                    c_pred_np = c_pred[i].cpu().numpy()


                    # Plot anatomy and c_pred side by side
                    fig, axs = plt.subplots(1, 3, figsize=(8, 4))

                    axs[0].imshow(tof_np.squeeze(0), cmap='gray')
                    axs[0].set_title('TOF')
                    axs[0].axis('off')


                    axs[1].imshow(anatomy_np.squeeze(0), cmap='gray')
                    axs[1].set_title('Original Anatomy')
                    axs[1].axis('off')

                    axs[2].imshow(c_pred_np.squeeze(0), cmap='jet')
                    axs[2].set_title('Predicted SoS (c_pred)')
                    axs[2].axis('off')

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