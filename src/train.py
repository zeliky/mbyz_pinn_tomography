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
    def __init__(self, model, training_step_handler, train_dataset,val_dataset,  **kwargs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.training_step_handler = training_step_handler
        self.training_step_handler.init(model=model, device=self.device)

        self.epochs = kwargs.get('epochs', 10)
        self.batch_size = kwargs.get('batch_size', 1)
        self.lr = kwargs.get('lr',  1e-3)
        self.data_weight = kwargs.get('data_weight',  1e-4)
        self.pde_weight = kwargs.get('pde_weight',  1.0)
        self.bc_weight = kwargs.get('bc_weight',  1e-4)
        self.scheduler_step_size= 4

    def train_model(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        num_epochs = self.epochs
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler_step_size, gamma=0.1)
        for epoch in range(self.epochs):
            self.training_step_handler.set_train_mode()
            epoch_loss = 0
            with tqdm(total=train_loader.__len__()) as pbar:
                pbar.set_description("Training")
                optimizer.zero_grad()

                for batch in train_loader:
                    optimizer.zero_grad()
                    loss = self.training_step_handler.perform_step(batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    pbar.update(1)
                log_message(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.6f}')

            # Validation
            self.training_step_handler.set_eval_mode()
            val_loss = 0.0
            with torch.no_grad():
                with tqdm(total=val_loader.__len__()) as pbar:
                    pbar.set_description("Validation ")
                    for batch in val_loader:
                        loss = self.training_step_handler.perform_step(batch)
                        val_loss += loss.item()
                        pbar.update(1)

                    log_message(f'Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss / len(val_loader):.6f}')
            self.save_state(self.model, optimizer, val_loss, epoch)
            scheduler.step()




    def load_checkpoint(self, checkpoint_path):
        checkpoint_path = f"{app_settings.output_folder}/{checkpoint_path}"
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])

    def save_state(self, model, optimizer,loss, epoch):
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M_%S_%f")
        save_path = f"{app_settings.output_folder}/pinn_tof-predictor_model.{formatted_datetime}-{epoch}.pth"
        torch.save({
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
        }, save_path)
        log_message(f"Model saved to {save_path}")

    def check_tof(self, grid_h, grid_w):
        val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False)
        self.model.eval()
        with torch.no_grad():
            for batch in val_loader:
                input = batch['tof'].float().to(self.device)
                positions_mask = batch['positions_mask'].float().to(self.device)
                x_s = batch['x_s'].float()
                x_r = batch['x_r'].float()

                C_pred, T_pred = self.model(input, positions_mask)
                print(C_pred[0].tolist())
                exit()
                known_tof = input.squeeze()
                T_pred = T_pred.squeeze()
                src_loc = (x_s.squeeze() ).int()
                rec_loc = (x_r.squeeze() ).int()

                for s_idx, s in enumerate(src_loc):
                    p_tof = T_pred[s_idx, s[0], s[1]]
                    print(f"seloc:({s_idx}, {s[0]},{s[1]}) k_tof: {0}  p_tof: {p_tof}")
                    for r_idx, r in enumerate(rec_loc):
                        k_tof = known_tof[s_idx, r_idx]
                        p_tof= T_pred[s_idx, r[0],r[1]]
                        print(f"   reloc:({s_idx}, {r[0]},{r[1]}) k_tof: {k_tof}  p_tof: {p_tof}")


    def visualize_predictions(self,  num_samples=5):
        val_loader = DataLoader(self.val_dataset, batch_size=2, shuffle=False)
        self.model.eval()
        with torch.no_grad():
            count = 0
            for batch in val_loader:
                tof = batch['tof'].float().to(self.device)
                anatomy = batch['anatomy'].cpu()
                #positions_mask = batch['positions_mask'].float().to(self.device)
                #c_pred, t_pred = self.model(tof)
                #c_pred, t_pred = self.model(tof, positions_mask)
                c_pred, t_pred = self.model(tof)
                for i in range(c_pred.size(0)):
                    tof_np = tof[i].cpu().numpy()
                    anatomy_np = anatomy[i].numpy()
                    c_pred_np = c_pred[i].cpu().numpy()


                    # Plot anatomy and c_pred side by side
                    fig, axs = plt.subplots(1, 3, figsize=(8, 4))

                    axs[0].imshow(tof_np.squeeze(0), cmap='jet')
                    axs[0].set_title('TOF')
                    axs[0].axis('off')


                    axs[1].imshow(anatomy_np.squeeze(0), cmap='jet')
                    axs[1].set_title('Original Anatomy')
                    axs[1].axis('off')

                    #axs[2].imshow(c_pred_np.squeeze(0), cmap='jet')
                    axs[2].imshow(c_pred_np, cmap='jet')
                    axs[2].set_title('Predicted SoS (c_pred)')
                    axs[2].axis('off')

                    #plt.tight_layout()
                    #plt.show()
                    log_image(fig)
                    log_message(' ')
                    count += 1
                    if count >= num_samples:
                        return




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