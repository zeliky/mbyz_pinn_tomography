import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from logger import log_image, log_message

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

# Define the DataLoader
# Assuming you have a dataset that provides batches with keys: 'x_r', 'x_s', 'x_o'

# Placeholder for DataLoader
# Replace 'YourDataset' with your actual dataset class
# data_loader = DataLoader(YourDataset, batch_size=batch_size, shuffle=True)

# For illustration, let's create a dummy DataLoader
class TOFDataset(torch.utils.data.Dataset):
    def __init__(self, x_s_all, x_r_all, x_o_all):
        self.x_s_all = x_s_all
        self.x_r_all = x_r_all
        self.x_o_all = x_o_all

    def __len__(self):
        return len(self.x_o_all)

    def __getitem__(self, idx):
        return {
            'x_s': self.x_s_all[idx],
            'x_r': self.x_r_all[idx],
            'x_o': self.x_o_all[idx]
        }

# Assuming you have arrays x_s_all, x_r_all, x_o_all containing all your data
# For demonstration, let's create dummy data
num_pairs = 1000  # Total number of source-receiver pairs
x_s_all = torch.rand(num_pairs, 2) * (x_max - x_min) + x_min  # Sources
x_r_all = torch.rand(num_pairs, 2) * (y_max - y_min) + y_min  # Receivers
# Simulate observed times of flight (using a true_c function)
def true_c_func(x, y):
    return 1.0 + 0.5 * x + 0.3 * y
def generate_tof(x_s_all, x_r_all, true_c_func):
    times = []
    for s, r in zip(x_s_all, x_r_all):
        x_s, y_s = s
        x_r, y_r = r
        distance = torch.sqrt((x_r - x_s)**2 + (y_r - y_s)**2)
        avg_speed = true_c_func((x_s + x_r)/2, (y_s + y_r)/2)
        time = distance / avg_speed
        times.append(time)
    return torch.tensor(times)

x_o_all = generate_tof(x_s_all, x_r_all, true_c_func)

# Create dataset and DataLoader
dataset = TOFDataset(x_s_all, x_r_all, x_o_all)
batch_size = 64
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Generate collocation points once outside the training loop
num_collocation = 10000
xy_collocation = torch.rand(num_collocation, 2, device=device)
xy_collocation[:, 0] = xy_collocation[:, 0] * (x_max - x_min) + x_min
xy_collocation[:, 1] = xy_collocation[:, 1] * (y_max - y_min) + y_min
xy_collocation.requires_grad_(True)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    for batch in data_loader:
        x_s = batch['x_s'].to(device)  # Shape: [batch_size, 2]
        x_r = batch['x_r'].to(device)  # Shape: [batch_size, 2]
        observed_tof = batch['x_o'].to(device).unsqueeze(1)  # Shape: [batch_size, 1]
        b_size,_= batch['x_s'].shape
        optimizer.zero_grad()

        # Data loss: Predict times of flight and compare with observed
        u_pred = model_u(x_r, x_s)  # Predicted times of flight, Shape: [batch_size, 1]
        data_loss = nn.MSELoss()(u_pred, observed_tof)

        # PDE loss: Enforce Eikonal equation at collocation points
        # For PDE loss, we can sample a subset of sources from the batch
        pde_batch_size = min(10, batch_size)  # Limit number of sources for PDE loss
        indices = torch.randperm(b_size)[:pde_batch_size]
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
    if epoch % 5 == 0:
        log_message(f'Epoch {epoch}, Loss: {loss.item():.6f}, Data Loss: {data_loss.item():.6f}, PDE Loss: {total_pde_loss.item():.6f}, Boundary Loss: {boundary_loss.item():.6f}')

# After training, you can evaluate c(x, y) and visualize as before
x_vis = torch.linspace(x_min, x_max, 128, device=device)
y_vis = torch.linspace(y_min, y_max, 128, device=device)
X_vis, Y_vis = torch.meshgrid(x_vis, y_vis, indexing='ij')
XY_vis = torch.hstack((X_vis.reshape(-1, 1), Y_vis.reshape(-1, 1)))
c_pred_vis = model_c(XY_vis).detach().cpu().numpy().reshape(128, 128)

# True c(x, y) for comparison (unknown in practice)
c_true_vis = true_c_func(X_vis.cpu(), Y_vis.cpu()).numpy().reshape(128, 128)




# Visualization
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 5))

# Predicted c(x, y)
plt.subplot(1, 2, 1)
plt.imshow(c_pred_vis, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='viridis')
plt.title('Predicted Speed c(x, y)')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='c(x, y)')

# True c(x, y)
plt.subplot(1, 2, 2)
plt.imshow(c_true_vis, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='viridis')
plt.title('True Speed c(x, y) (Unknown in Practice)')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='c(x, y)')

plt.tight_layout()
log_image(fig)
