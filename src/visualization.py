import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def visualize_tof_image(dataset, idx=0, p=None):
    """
    Visualizes an example ToF image from the dataset.

    Args:
        dataset (TofDataset): The dataset object containing ToF images.
        idx (int): The index of the ToF image to visualize (default: 0).

    Returns:
        None
    """
    if idx < 0 or idx >= len(dataset):
        p.print(f"Index {idx} is out of range. Dataset size: {len(dataset)}")
        return

    tof_data = dataset.tof_images[idx]
    if tof_data.ndim != 2 or tof_data.shape != (dataset.grid_size, dataset.grid_size):
        p.print(f"Unexpected shape for ToF data: {tof_data.shape}. Expected ({dataset.grid_size}, {dataset.grid_size})")
        return

    fontsize = 16
    labelsize = 16
    nbins = 5
    
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(tof_data, cmap='viridis', origin='upper')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = plt.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=labelsize)
    cb.set_label('ToF (Time of Flight)', fontsize=fontsize)
    ax.set_title(f"ToF Image Example at Index {idx}", fontsize=fontsize)
    ax.set_xlabel("X-axis", fontsize=fontsize)
    ax.set_ylabel("Y-axis", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    ax.locator_params(axis='x', nbins=nbins)
    ax.locator_params(axis='y', nbins=nbins)
    plt.tight_layout()
    p.print('[visualization.py]: visualize_tof_image')
    plt.show()
    p.show(fig)


def visualize_anatomy_image(dataset, idx=0, p=None):
    """
    Visualizes an example anatomy image from the dataset.

    Args:
        dataset (TofDataset): The dataset object containing anatomy images.
        idx (int): The index of the anatomy image to visualize (default: 0).

    Returns:
        None
    """
    if idx < 0 or idx >= len(dataset):
        print(f"Index {idx} is out of range. Dataset size: {len(dataset)}")
        return

    anatomy_data = dataset.anatomy_images[idx]
    if anatomy_data.ndim != 2 or anatomy_data.shape != (dataset.grid_size, dataset.grid_size):
        print(f"Unexpected shape for Anatomy data: {anatomy_data.shape}. Expected ({dataset.grid_size}, {dataset.grid_size})")
        return

    fontsize = 16
    labelsize = 16
    nbins = 5
    
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(anatomy_data, cmap='gray', origin='upper')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = plt.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=labelsize)
    cb.set_label('Pixel Intensity', fontsize=fontsize)
    ax.set_title(f"Anatomy Image Example at Index {idx}", fontsize=fontsize)
    ax.set_xlabel("X-axis", fontsize=fontsize)
    ax.set_ylabel("Y-axis", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    ax.locator_params(axis='x', nbins=nbins)
    ax.locator_params(axis='y', nbins=nbins)
    plt.tight_layout()
    p.print('[visualization.py]: visualize_anatomy_image')
    plt.show()
    p.show(fig)


def visualize_sources_and_receivers(dataset, idx=0, p=None):
    """
    Visualizes the source and receiver positions on top of the SoS values grid in grayscale.

    Args:
        dataset (TofDataset): The dataset object containing ToF data.
        idx (int): The index of the data to visualize (default: 0).

    Returns:
        None
    """
    if idx < 0 or idx >= len(dataset):
        print(f"Index {idx} is out of range. Dataset size: {len(dataset)}")
        return

    tof_data = dataset.tof_data[idx]
    source_positions = tof_data['source_positions']
    receiver_positions = tof_data['receiver_positions']
    sos_values = dataset.sos_data[idx]
    grid_size = dataset.grid_size
    sos_matrix = np.reshape(sos_values, (grid_size, grid_size))

    fontsize = 16
    labelsize = 16
    nbins = 5
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(sos_matrix, cmap='gray', origin='upper', extent=[0, dataset.anatomy_width, 0, dataset.anatomy_height])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = plt.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=labelsize)
    cb.set_label('Speed of Sound (SoS)', fontsize=fontsize)
    ax.set_title(f"SoS Values with Sources and Receivers at Index {idx}", fontsize=fontsize)
    ax.set_xlabel("X-axis", fontsize=fontsize)
    ax.set_ylabel("Y-axis", fontsize=fontsize)
    ax.scatter(source_positions[:, 0], source_positions[:, 1], c='green', label='Sources', s=50)
    ax.scatter(receiver_positions[:, 0], receiver_positions[:, 1], c='red', label='Receivers', s=50)
    ax.legend(loc='upper right', fontsize=fontsize)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    ax.locator_params(axis='x', nbins=nbins)
    ax.locator_params(axis='y', nbins=nbins)
    plt.tight_layout()
    p.print('[visualization.py]: visualize_sources_and_receivers')
    plt.show()
    p.show(fig)

# import matplotlib.pyplot as plt
# import numpy as np

# def visualize_tof_image(dataset, idx=0, p=None):
#     """
#     Visualizes an example ToF image from the dataset.

#     Args:
#         dataset (TofDataset): The dataset object containing ToF images.
#         idx (int): The index of the ToF image to visualize (default: 0).

#     Returns:
#         None
#     """
#     if idx < 0 or idx >= len(dataset):
#         p.print(f"Index {idx} is out of range. Dataset size: {len(dataset)}")
#         return

#     tof_data = dataset.tof_images[idx]
#     if tof_data.ndim != 2 or tof_data.shape != (dataset.grid_size, dataset.grid_size):
#         p.print(f"Unexpected shape for ToF data: {tof_data.shape}. Expected ({dataset.grid_size}, {dataset.grid_size})")
#         return

#     fontsize = 16
#     labelsize = 16
#     nbins = 5
    
#     fig = plt.figure(figsize=(6, 6))
#     plt.imshow(tof_data, cmap='viridis', origin='upper')
#     cb = plt.colorbar()
#     cb.ax.tick_params(labelsize=labelsize)
#     cb.set_label('ToF (Time of Flight)', fontsize=fontsize)
#     plt.title(f"ToF Image Example at Index {idx}", fontsize=fontsize)
#     plt.xlabel("X-axis", fontsize=fontsize)
#     plt.ylabel("Y-axis", fontsize=fontsize)
#     plt.tick_params(axis='both', which='major', labelsize=labelsize)
#     plt.locator_params(axis='x', nbins=nbins)  # Reduce number of x-axis ticks
#     plt.locator_params(axis='y', nbins=nbins)  # Reduce number of y-axis ticks
#     plt.tight_layout()  # Adjust layout to fit labels and elements
#     p.print('[visualization.py]: visualize_tof_image')
#     plt.show()
#     p.show(fig)


# def visualize_anatomy_image(dataset, idx=0, p=None):
#     """
#     Visualizes an example anatomy image from the dataset.

#     Args:
#         dataset (TofDataset): The dataset object containing anatomy images.
#         idx (int): The index of the anatomy image to visualize (default: 0).

#     Returns:
#         None
#     """
#     if idx < 0 or idx >= len(dataset):
#         print(f"Index {idx} is out of range. Dataset size: {len(dataset)}")
#         return

#     anatomy_data = dataset.anatomy_images[idx]
#     if anatomy_data.ndim != 2 or anatomy_data.shape != (dataset.grid_size, dataset.grid_size):
#         print(f"Unexpected shape for Anatomy data: {anatomy_data.shape}. Expected ({dataset.grid_size}, {dataset.grid_size})")
#         return

#     fontsize = 16
#     labelsize = 16
#     nbins = 5
    
#     fig = plt.figure(figsize=(6, 6))
#     plt.imshow(anatomy_data, cmap='gray', origin='upper')
#     cb = plt.colorbar()
#     cb.ax.tick_params(labelsize=labelsize)
#     cb.set_label('Pixel Intensity', fontsize=fontsize)
#     plt.title(f"Anatomy Image Example at Index {idx}", fontsize=fontsize)
#     plt.xlabel("X-axis", fontsize=fontsize)
#     plt.ylabel("Y-axis", fontsize=fontsize)
#     plt.tick_params(axis='both', which='major', labelsize=labelsize)
#     plt.locator_params(axis='x', nbins=nbins)
#     plt.locator_params(axis='y', nbins=nbins)
#     plt.tight_layout()
#     p.print('[visualization.py]: visualize_anatomy_image')
#     plt.show()
#     p.show(fig)


# def visualize_sources_and_receivers(dataset, idx=0, p=None):
#     """
#     Visualizes the source and receiver positions on top of the SoS values grid in grayscale.

#     Args:
#         dataset (TofDataset): The dataset object containing ToF data.
#         idx (int): The index of the data to visualize (default: 0).

#     Returns:
#         None
#     """
#     if idx < 0 or idx >= len(dataset):
#         print(f"Index {idx} is out of range. Dataset size: {len(dataset)}")
#         return

#     tof_data = dataset.tof_data[idx]
#     source_positions = tof_data['source_positions']
#     receiver_positions = tof_data['receiver_positions']
#     sos_values = dataset.sos_data[idx]
#     grid_size = dataset.grid_size
#     sos_matrix = np.reshape(sos_values, (grid_size, grid_size))

#     fontsize = 16
#     labelsize = 16
#     nbins = 5
    
#     fig = plt.figure(figsize=(8, 8))
#     plt.imshow(sos_matrix, cmap='gray', origin='upper', extent=[0, dataset.anatomy_width, 0, dataset.anatomy_height])
#     cb = plt.colorbar()
#     cb.ax.tick_params(labelsize=labelsize)
#     cb.set_label('Speed of Sound (SoS)', fontsize=fontsize)
#     plt.title(f"SoS Values with Sources and Receivers at Index {idx}", fontsize=fontsize)
#     plt.xlabel("X-axis", fontsize=fontsize)
#     plt.ylabel("Y-axis", fontsize=fontsize)
#     plt.scatter(source_positions[:, 0], source_positions[:, 1], c='green', label='Sources', s=50)
#     plt.scatter(receiver_positions[:, 0], receiver_positions[:, 1], c='red', label='Receivers', s=50)
#     plt.legend(loc='upper right', fontsize=fontsize)
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.tick_params(axis='both', which='major', labelsize=labelsize)
#     plt.locator_params(axis='x', nbins=nbins)
#     plt.locator_params(axis='y', nbins=nbins)
#     plt.tight_layout()
#     p.print('[visualization.py]: visualize_sources_and_receivers')
#     plt.show()
#     p.show(fig)


# import matplotlib.pyplot as plt
# import numpy as np

# def visualize_tof_image(dataset, idx=0, p=None):
#     """
#     Visualizes an example ToF image from the dataset.

#     Args:
#         dataset (TofDataset): The dataset object containing ToF images.
#         idx (int): The index of the ToF image to visualize (default: 0).

#     Returns:
#         None
#     """
#     if idx < 0 or idx >= len(dataset):
#         p.print(f"Index {idx} is out of range. Dataset size: {len(dataset)}")
#         return

#     tof_data = dataset.tof_images[idx]
#     if tof_data.ndim != 2 or tof_data.shape != (dataset.grid_size, dataset.grid_size):
#         p.print(f"Unexpected shape for ToF data: {tof_data.shape}. Expected ({dataset.grid_size}, {dataset.grid_size})")
#         return

#     fontsize = 20
#     labelsize = 20
    
#     fig = plt.figure(figsize=(6, 6))
#     plt.imshow(tof_data, cmap='viridis', origin='upper')
#     cb = plt.colorbar()
#     cb.ax.tick_params(labelsize=labelsize)
#     cb.set_label('ToF (Time of Flight)', fontsize=fontsize)
#     plt.title(f"ToF Image Example at Index {idx}", fontsize=fontsize)
#     plt.xlabel("X-axis", fontsize=fontsize)
#     plt.ylabel("Y-axis", fontsize=fontsize)
#     plt.tick_params(axis='both', which='major', labelsize=labelsize)
#     p.print('[visualization.py]: visualize_tof_image')
#     plt.show()
#     p.show(fig)


# def visualize_anatomy_image(dataset, idx=0, p=None):
#     """
#     Visualizes an example anatomy image from the dataset.

#     Args:
#         dataset (TofDataset): The dataset object containing anatomy images.
#         idx (int): The index of the anatomy image to visualize (default: 0).

#     Returns:
#         None
#     """
#     if idx < 0 or idx >= len(dataset):
#         print(f"Index {idx} is out of range. Dataset size: {len(dataset)}")
#         return

#     anatomy_data = dataset.anatomy_images[idx]
#     if anatomy_data.ndim != 2 or anatomy_data.shape != (dataset.grid_size, dataset.grid_size):
#         print(f"Unexpected shape for Anatomy data: {anatomy_data.shape}. Expected ({dataset.grid_size}, {dataset.grid_size})")
#         return

#     fontsize = 20
#     labelsize = 20
    
#     fig = plt.figure(figsize=(6, 6))
#     plt.imshow(anatomy_data, cmap='gray', origin='upper')
#     cb = plt.colorbar()
#     cb.ax.tick_params(labelsize=labelsize)
#     cb.set_label('Pixel Intensity', fontsize=fontsize)
#     plt.title(f"Anatomy Image Example at Index {idx}", fontsize=14)
#     plt.xlabel("X-axis", fontsize=fontsize)
#     plt.ylabel("Y-axis", fontsize=fontsize)
#     plt.tick_params(axis='both', which='major', labelsize=labelsize)
#     p.print('[visualization.py]: visualize_anatomy_image')
#     plt.show()
#     p.show(fig)


# def visualize_sources_and_receivers(dataset, idx=0, p=None):
#     """
#     Visualizes the source and receiver positions on top of the SoS values grid in grayscale.

#     Args:
#         dataset (TofDataset): The dataset object containing ToF data.
#         idx (int): The index of the data to visualize (default: 0).

#     Returns:
#         None
#     """
#     if idx < 0 or idx >= len(dataset):
#         print(f"Index {idx} is out of range. Dataset size: {len(dataset)}")
#         return

#     tof_data = dataset.tof_data[idx]
#     source_positions = tof_data['source_positions']
#     receiver_positions = tof_data['receiver_positions']
#     sos_values = dataset.sos_data[idx]
#     grid_size = dataset.grid_size
#     sos_matrix = np.reshape(sos_values, (grid_size, grid_size))


#     fontsize = 20
#     labelsize = 20
    
#     fig = plt.figure(figsize=(8, 8))
#     plt.imshow(sos_matrix, cmap='gray', origin='upper', extent=[0, dataset.anatomy_width, 0, dataset.anatomy_height])
#     cb = plt.colorbar()
#     cb.ax.tick_params(labelsize=labelsize)
#     cb.set_label('Speed of Sound (SoS)', fontsize=fontsize)
#     plt.title(f"SoS Values with Sources and Receivers at Index {idx}", fontsize=14)
#     plt.xlabel("X-axis", fontsize=fontsize)
#     plt.ylabel("Y-axis", fontsize=fontsize)
#     plt.scatter(source_positions[:, 0], source_positions[:, 1], c='green', label='Sources', s=50)
#     plt.scatter(receiver_positions[:, 0], receiver_positions[:, 1], c='red', label='Receivers', s=50)
#     plt.legend(loc='upper right', fontsize=fontsize)
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.tick_params(axis='both', which='major', labelsize=labelsize)
#     p.print('[visualization.py]: visualize_sources_and_receivers')
#     plt.show()
#     p.show(fig)
