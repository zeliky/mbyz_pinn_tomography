import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from logger import log_image, log_message
from dataset import TofDataset
from settings import  app_settings
# Set random seed for reproducibility
torch.manual_seed(0)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define domain boundaries
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0

# Define neural network for c(x, y)
class SpeedNN(nn.Module):
    def __init__(self):
        super(SpeedNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
            nn.Softplus()  # Ensure positive speed
        )
    def forward(self, x):
        return self.net(x)

# Define neural network for u(x, y, x_s, y_s)
class TravelTimeNN(nn.Module):
    def __init__(self):
        super(TravelTimeNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 50),  # Input: (x_r, y_r, x_s, y_s)
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )
    def forward(self, x_r, x_s):
        # x_r: (batch_size, 2), x_s: (batch_size, 2)
        input = torch.cat([x_r, x_s], dim=1)  # Shape: (batch_size, 4)
        return self.net(input)

# Initialize models
model_c = SpeedNN().to(device)
model_u = TravelTimeNN().to(device)

# Define optimizer
optimizer = optim.Adam(list(model_u.parameters()) + list(model_c.parameters()), lr=1e-3)



# Assuming you have arrays x_s_all, x_r_all, x_o_all containing all your data
# For demonstration, let's create dummy data
num_pairs = 1000  # Total number of source-receiver pairs



# Create dataset and DataLoader
dataset = TofDataset(['train'])
batch_size = 32*32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Generate collocation points once outside the training loop
num_collocation = 50*50
xy_collocation = torch.rand(num_collocation, 2, device=device)
xy_collocation[:, 0] = xy_collocation[:, 0] * (x_max - x_min) + x_min
xy_collocation[:, 1] = xy_collocation[:, 1] * (y_max - y_min) + y_min
xy_collocation.requires_grad_(True)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    for batch in data_loader:
        x_s = batch['x_s'].to(device)  # Shape: [batch_size, 2]
        x_r = batch['x_r'].to(device)  # Shape: [batch_size, 2]
        observed_tof = batch['x_o'].to(device).unsqueeze(1)  # Shape: [batch_size, 1]
        optimizer.zero_grad()

        # Data loss: Predict times of flight and compare with observed
        u_pred = model_u(x_r, x_s)  # Predicted times of flight, Shape: [batch_size, 1]
        data_loss = nn.MSELoss()(u_pred, observed_tof)

        # PDE loss: Enforce Eikonal equation at collocation points
        # Sample a subset of sources from the batch for PDE loss
        pde_batch_size = min(10, batch_size)
        indices = torch.randperm(batch_size)[:pde_batch_size]
        total_pde_loss = 0

        for idx in indices:
            x_s_i = x_s[idx].unsqueeze(0).repeat(num_collocation, 1)
            u_i_pred = model_u(xy_collocation, x_s_i)
            c_pred = model_c(xy_collocation)

            grads = torch.autograd.grad(u_i_pred, xy_collocation, torch.ones_like(u_i_pred), create_graph=True, retain_graph=True)[0]
            u_x = grads[:, 0].unsqueeze(1)
            u_y = grads[:, 1].unsqueeze(1)
            grad_magnitude = torch.sqrt(u_x ** 2 + u_y ** 2 + 1e-8)

            residual = grad_magnitude - 1 / c_pred
            pde_loss = torch.mean(residual ** 2)
            total_pde_loss += pde_loss

        total_pde_loss = total_pde_loss / pde_batch_size  # Average over the sampled sources

        # Boundary condition: u(x_s, x_s) = 0
        u_source_pred = model_u(x_s, x_s)
        boundary_loss = nn.MSELoss()(u_source_pred, torch.zeros_like(u_source_pred))

        # Total loss
        loss = data_loss + total_pde_loss + boundary_loss

        loss.backward()
        optimizer.step()

        # Optionally print progress every few epochs
    #if epoch % 5 == 0:
    print(f'Epoch {epoch}, Loss: {loss.item():.6f}, Data Loss: {data_loss.item():.6f}, PDE Loss: {total_pde_loss.item():.6f}, Boundary Loss: {boundary_loss.item():.6f}')


current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M_%S_%f")
save_path = f"{app_settings.output_folder()}/pinn_tof-sos_model.{formatted_datetime}.pth"

# Save the state dictionaries
torch.save({
    'model_c_state_dict': model_c.state_dict(),
    'model_u_state_dict': model_u.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),  # Optional, if you plan to resume training
    'epoch': epoch,
    'loss': loss.item(),
}, save_path)

print(f"Models saved to {save_path}")




model_c.eval()
dataset = TofDataset(['validation'])


# After training, you can evaluate c(x, y) and visualize as before
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0

# Create grid coordinates
grid_size = 318
x_vis = torch.linspace(x_min, x_max, grid_size, device=device)
y_vis = torch.linspace(y_min, y_max, grid_size, device=device)
X_vis, Y_vis = torch.meshgrid(x_vis, y_vis, indexing='ij')

# Flatten the grid for model input
XY_vis = torch.hstack((X_vis.reshape(-1, 1), Y_vis.reshape(-1, 1)))  # Shape: [grid_size^2, 2]


with torch.no_grad():
    # Predict c(x, y) over the grid
    c_pred_vis = model_c(XY_vis).detach().cpu().numpy().reshape(grid_size, grid_size)

if isinstance(sos_true, torch.Tensor):
    sos_true = sos_true.numpy()


# Visualization
import matplotlib.pyplot as plt

# Plot the predicted sos map
fig1 = plt.figure(figsize=(8, 6))
plt.imshow(c_pred_vis, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='viridis')
plt.title('Predicted Speed of Sound c(x, y)')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='c(x, y)')
log_image(fig1)

# Plot the ground truth sos map
fig1 = plt.figure(figsize=(8, 6))
plt.imshow(sos_true, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='viridis')
plt.title('Ground Truth Speed of Sound')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='c(x, y)')
log_image(fig2)


# Compute and plot the difference
sos_difference = np.abs(c_pred_vis - sos_true)
fig3 = plt.figure(figsize=(8, 6))
plt.imshow(sos_difference, extent=[0, 1, 0, 1], origin='lower', cmap='hot')
plt.title('Absolute Difference Between Predicted and True SOS')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='|Predicted - True|')
log_image(fig3)

# Compute error metrics
mse = np.mean((c_pred_vis - sos_true) ** 2)
mae = np.mean(np.abs(c_pred_vis - sos_true))
max_pixel = np.max(sos_true)
psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr} dB")

