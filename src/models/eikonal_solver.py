import torch
import torch.nn as nn
import torch.nn.functional as F
import heapq
import math
import  numpy as np

class EikonalSolverMultiLayer(nn.Module):
    def __init__(self, num_layers, speed_of_sound, domain_size, grid_resolution, kernel_size=3):
        super().__init__()
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.speed_of_sound = speed_of_sound
        self.domain_size = domain_size
        self.grid_resolution =  grid_resolution

        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            for _ in range(num_layers)
        ])

    def reset_weights(self):
        # Initialize weights for slowness (1/c)
        self.delta_s = self.domain_size / self.grid_resolution
        #self.delta_s = 1
        for conv in self.convs:
            conv.weight.data.fill_(self.delta_s / self.speed_of_sound)

    def forward(self, T_init, sos_pred):
        T = T_init.clone()

        for conv in self.convs:
            # Compute tentative updates from neighbors
            #slowness = 1.0 / sos_pred
            slowness = self.delta_s / sos_pred.clamp(min=1e-8)  # Add batch and channel dims

            T_neighbors = conv(T) * slowness
            # Update travel time by propagating the minimum travel time
            T = torch.min(T, T_neighbors)

        return T





class DifferentiableSolver(torch.nn.Module):
    def __init__(self,  dy=1, dx=1):
        super(DifferentiableSolver, self).__init__()
        self.dy = dy
        self.dx = dx

    def forward(self, sos, sources, receivers):
        """
        Compute Time of Flight (ToF) between a source and a list of receivers, considering c(x, y).

        :param sos: Speed of sound tensor of shape (batch_size, 1, grid_size, grid_size).
        :param sources: Tuple (x, y) indicating the source coordinates.
        :param receivers: List of tuples [(x1, y1), (x2, y2), ...] indicating receiver coordinates.
        :return: Tuple (arrival_times, ToF).
                 - arrival_times: Tensor of shape (batch_size, grid_x, grid_y) with the time of arrival for all points.
                 - ToF: Tensor of shape (batch_size, len(receivers)) with the time of flight to each receiver.
        """
        batch_size, _, grid_x, grid_y = sos.shape
        device = sos.device

        sos = sos.cpu()


        # Output ToF for each receiver in each batch
        tof = np.full((batch_size, grid_x, grid_y), np.inf)
        for b in range(batch_size):
            for sid, source_idx in enumerate(sources):
                arrival_times = np.full_like(sos[b].squeeze(), np.inf)
                arrival_times[source_idx] = 0
                arrival_times = fast_marching(sos[b].squeeze(), arrival_times, self.dx, self.dy)
                update_tof_matrix(sid, source_idx, receivers, arrival_times, tof[b])
                print(tof[b][sid].tolist())
        return tof




# Fast Marching Method
# This function solves the Eikonal equation using the Fast Marching Method (FMM).
def fast_marching(speed, T, dx, dy):
    """
    Compute the arrival times T using the Fast Marching Method.

    Args:
        speed: 2D numpy array of sound speed values.
        T: 2D numpy array of initial arrival times (infinity for all except source).
        dx, dy: Grid spacings in the x and y directions.

    Returns:
        T: 2D numpy array of computed arrival times.
    """
    nx, ny = T.shape  # Dimensions of the grid
    heap = []  # Priority queue to process grid points

    # Initialize the heap with known arrival times (source and its neighbors)
    for i in range(nx):
        for j in range(ny):
            if T[i, j] < np.inf:
                heapq.heappush(heap, (T[i, j], i, j))

    visited = np.zeros_like(T, dtype=bool)  # Track visited grid points
    debug_counter = 0  # Debug counter to limit log outputs

    # Process the heap until all reachable points are processed
    while heap:
        t, i, j = heapq.heappop(heap)  # Get the point with the smallest arrival time
        if visited[i, j]:
            continue  # Skip if the point has already been processed

        visited[i, j] = True  # Mark the point as visited

        # Iterate over neighbors (4-connectivity: up, down, left, right)
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj

            # Check if the neighbor is within bounds and not yet visited
            if 0 <= ni < nx and 0 <= nj < ny and not visited[ni, nj]:
                s = speed[ni, nj]  # Sound speed at the neighbor point
                if s <= 0:
                    continue  # Skip points with zero or negative speed
                dt = 1 / s  # Time increment based on speed

                # Compute the minimum arrival time using neighboring points
                t_x_candidates = []
                if 0 <= ni - 1 < nx:
                    t_x_candidates.append(T[ni - 1, nj])
                if 0 <= ni + 1 < nx:
                    t_x_candidates.append(T[ni + 1, nj])

                t_y_candidates = []
                if 0 <= nj - 1 < ny:
                    t_y_candidates.append(T[ni, nj - 1])
                if 0 <= nj + 1 < ny:
                    t_y_candidates.append(T[ni, nj + 1])

                t_x = min(t_x_candidates) if t_x_candidates else np.inf
                t_y = min(t_y_candidates) if t_y_candidates else np.inf

                # Solve the quadratic equation for arrival time
                if t_x != np.inf and t_y != np.inf:
                    a = 2
                    b = -2 * (t_x + t_y)
                    c = t_x**2 + t_y**2 - dt**2
                    discriminant = b**2 - 4 * a * c

                    if discriminant < 0:
                        tentative_t = np.inf
                    else:
                        tentative_t = (-b + np.sqrt(discriminant)) / (2 * a)
                elif t_x != np.inf:
                    tentative_t = t_x + dt
                elif t_y != np.inf:
                    tentative_t = t_y + dt
                else:
                    tentative_t = np.inf

                # Update arrival time if a smaller time is found
                if tentative_t < T[ni, nj]:
                    if debug_counter < 2000000:  # Limit debug outputs
                        debug_counter += 1
                    T[ni, nj] = tentative_t
                    heapq.heappush(heap, (tentative_t, ni, nj))

    if debug_counter > 0:
        print(f"[eikonal.py] updated T elements {debug_counter} times")

    return T


# Calculate Raypaths
# This function computes raypaths from the emitter to multiple receivers based on arrival times.
def calculate_raypaths(emitter_idx, receiver_indices, T):
    """
    Calculate the raypaths from the emitter to multiple receivers.

    Args:
        emitter_idx: Index of the emitter in the grid (tuple: (i, j)).
        receiver_indices: List of indices for the receivers in the grid [(i1, j1), (i2, j2), ...].
        T: 2D numpy array of computed arrival times.

    Returns:
        raypaths: List of raypaths, where each raypath is a list of (i, j) coordinates.
    """
    raypaths = []  # Store all computed raypaths
    for receiver_idx in receiver_indices:
        raypath = [receiver_idx]  # Initialize the raypath with the receiver
        current_idx = receiver_idx

        # Trace back the path from receiver to emitter
        while current_idx != emitter_idx:
            i, j = current_idx
            # Get neighboring points
            neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
            neighbors = [(ni, nj) for ni, nj in neighbors if 0 <= ni < T.shape[0] and 0 <= nj < T.shape[1]]

            # Find the neighbor with the smallest arrival time
            next_idx = min(neighbors, key=lambda idx: T[idx])
            raypath.append(next_idx)  # Add the next point to the raypath
            current_idx = next_idx  # Move to the next point

        raypaths.append(raypath[::-1])  # Reverse the path to go from emitter to receiver
    return raypaths

def update_tof_matrix(sid, emitter_idx, receiver_indices, T, tof):


    time = 0
    for rid, receiver_idx in  enumerate(receiver_indices):
        current_idx = receiver_idx
        # Trace back the path from receiver to emitter
        while current_idx != emitter_idx:
            i, j = current_idx
            # Get neighboring points
            neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
            neighbors = [(ni, nj) for ni, nj in neighbors if 0 <= ni < T.shape[0] and 0 <= nj < T.shape[1]]
            # Find the neighbor with the smallest arrival time
            next_idx = min(neighbors, key=lambda idx: T[idx])
            time += T[next_idx]
            current_idx = next_idx  # Move to the next point

        tof[sid, rid] = time
    return tof
