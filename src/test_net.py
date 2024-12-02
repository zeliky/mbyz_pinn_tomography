import torch
import torch.nn as nn
import torch.optim as optim

# Set random seed for reproducibility
torch.manual_seed(0)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define domain boundaries
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0

# Define source locations
source_points = torch.tensor([
    [0.2, 0.2],
    [0.8, 0.2],
    [0.5, 0.8]
], device=device)

num_sources = source_points.shape[0]

# Define receiver locations
receiver_points = torch.tensor([
    [0.1, 0.5],
    [0.9, 0.5],
    [0.5, 0.1],
    [0.5, 0.9]
], device=device)

num_receivers = receiver_points.shape[0]


# True speed of sound function (unknown in practice)
def true_c_func(x, y):
    # Example: Speed varies with x and y
    return 1.0 + 0.5 * x + 0.3 * y


# Measured times of flight from each source to each receiver
def generate_measured_times(source_points, receiver_points, true_c_func):
    times = []
    for s in source_points:
        s_times = []
        for r in receiver_points:
            # Compute true travel time (simplified)
            x_s, y_s = s
            x_r, y_r = r
            # For simplicity, assume straight-line path and average speed
            distance = torch.sqrt((x_r - x_s) ** 2 + (y_r - y_s) ** 2)
            avg_speed = true_c_func((x_s + x_r) / 2, (y_s + y_r) / 2)
            time = distance / avg_speed
            s_times.append(time)
        times.append(s_times)
    return torch.tensor(times, device=device)  # Shape: (num_sources, num_receivers)


# Generate measured times
measured_times = generate_measured_times(source_points, receiver_points, true_c_func)

# Create collocation points in the domain for PDE residual computation
num_points = 10000
xy_collocation = torch.rand(num_points, 2, device=device)
xy_collocation[:, 0] = xy_collocation[:, 0] * (x_max - x_min) + x_min
xy_collocation[:, 1] = xy_collocation[:, 1] * (y_max - y_min) + y_min
xy_collocation.requires_grad_(True)


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


# Define neural network for u_i(x, y)
class TravelTimeNN(nn.Module):
    def __init__(self):
        super(TravelTimeNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 50),  # Input: (x, y, x_s, y_s)
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, x, s):
        # x: (batch_size, 2), s: (batch_size, 2)
        input = torch.cat([x, s], dim=1)
        return self.net(input)


# Initialize models
model_c = SpeedNN().to(device)
model_u = TravelTimeNN().to(device)

# Define optimizer
optimizer = optim.Adam(list(model_u.parameters()) + list(model_c.parameters()), lr=1e-3)

# Training loop
num_epochs = 5000
for epoch in range(num_epochs):
    optimizer.zero_grad()

    total_pde_loss = 0
    total_data_loss = 0
    total_bc_loss = 0

    # Compute PDE loss for each source
    for i in range(num_sources):
        s_i = source_points[i].unsqueeze(0).repeat(num_points, 1)
        u_i_pred = model_u(xy_collocation, s_i)
        c_pred = model_c(xy_collocation)

        grads = torch.autograd.grad(u_i_pred, xy_collocation, torch.ones_like(u_i_pred), create_graph=True)[0]
        u_x = grads[:, 0].unsqueeze(1)
        u_y = grads[:, 1].unsqueeze(1)
        grad_magnitude = torch.sqrt(u_x ** 2 + u_y ** 2 + 1e-8)

        residual = grad_magnitude - 1 / c_pred
        pde_loss = torch.mean(residual ** 2)
        total_pde_loss += pde_loss

    # Compute data loss at receiver points
    for i in range(num_sources):
        s_i = source_points[i].unsqueeze(0).repeat(num_receivers, 1)
        r_j = receiver_points

        u_i_rj_pred = model_u(r_j, s_i)
        t_ij = measured_times[i].unsqueeze(1)
        data_loss = nn.MSELoss()(u_i_rj_pred, t_ij)
        total_data_loss += data_loss

    # Compute boundary condition loss at source points
    for i in range(num_sources):
        s_i = source_points[i].unsqueeze(0)
        u_i_si_pred = model_u(s_i, s_i)
        bc_loss = nn.MSELoss()(u_i_si_pred, torch.zeros_like(u_i_si_pred))
        total_bc_loss += bc_loss

    # Total loss
    loss = total_pde_loss + total_data_loss + total_bc_loss

    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f'Epoch {epoch}, Total Loss: {loss.item():.6f}, PDE Loss: {total_pde_loss.item():.6f}, Data Loss: {total_data_loss.item():.6f}, BC Loss: {total_bc_loss.item():.6f}')

        # Evaluate c(x, y) on a grid for visualization
        x_vis = torch.linspace(x_min, x_max, 128, device=device)
        y_vis = torch.linspace(y_min, y_max, 128, device=device)
        X_vis, Y_vis = torch.meshgrid(x_vis, y_vis, indexing='ij')
        XY_vis = torch.hstack((X_vis.reshape(-1, 1), Y_vis.reshape(-1, 1)))
        c_pred_vis = model_c(XY_vis).detach().cpu().numpy().reshape(128, 128)

        # True c(x, y) for comparison (unknown in practice)
        c_true_vis = true_c_func(X_vis.cpu(), Y_vis.cpu()).numpy().reshape(128, 128)

        # Visualization
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 5))

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
        plt.show()
