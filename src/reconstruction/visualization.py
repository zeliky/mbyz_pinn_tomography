import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from logger import log_message, log_image

from scipy.ndimage import rotate

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_results(speed, T, dx, dy, obstacles, emitters, receivers, selected_emitter, selected_receivers, raypaths):
    """
    Plot the speed map in grayscale, arrival time contours in color, obstacles, emitters, receivers, and raypaths.

    Args:
        speed: 2D numpy array of sound speed values.
        T: 2D numpy array of arrival times.
        dx, dy: Grid spacings.
        obstacles: List of tuples (cx, cy, rx, ry, speed) for ellipses.
        emitters: List of emitter coordinates (x, y).
        receivers: List of receiver coordinates (x, y).
        selected_emitter: Tuple of selected emitter coordinates (x, y).
        selected_receivers: List of selected receiver coordinates (x, y).
        raypaths: List of raypaths, where each raypath is a list of (i, j) tuples.
    """
    extent = (0, speed.shape[1] * dx, 0, speed.shape[0] * dy)
    fig, ax = plt.subplots(figsize=(8, 8))
    
    cmap = plt.get_cmap("gray")  # or plt.get_cmap("viridis")
    
    # Create an axes divider to allocate space for colorbars
    divider = make_axes_locatable(ax)

    # Plot the speed map in grayscale
    speed_im = ax.imshow(speed, extent=extent, origin='lower', cmap='gray', aspect='equal')
    cax_speed = divider.append_axes("right", size="5%", pad=0.1)
    cbar1 = fig.colorbar(speed_im, cax=cax_speed)
    cbar1.set_label('Speed of Sound')

    # # Plot obstacles
    # if obstacles is not None:
    #     for cx, cy, rx, ry, speed_val in obstacles:
    #         ellipse = Ellipse((cx, cy), 2 * rx, 2 * ry, edgecolor='black', facecolor='gray', lw=2, alpha=1)
    #         ax.add_patch(ellipse)
    
    # Plot obstacles
    if obstacles is not None:
        for cx, cy, rx, ry, speed_val in obstacles:
            face_color = cmap(speed_val / np.max(speed))
            ellipse = Ellipse((cx, cy), 2 * rx, 2 * ry, edgecolor=face_color, facecolor=face_color, lw=2, alpha=0.5)
            ax.add_patch(ellipse)

    # Plot all emitters and receivers
    for emitter in emitters:
        ax.plot(emitter[0], emitter[1], 'go', markersize=5, alpha=0.6)
    for receiver in receivers:
        ax.plot(receiver[0], receiver[1], 'ro', markersize=5, alpha=0.6)

    # Highlight selected emitter
    ax.plot(selected_emitter[0], selected_emitter[1], 'go', markersize=10, label='Selected Emitter')

    # Highlight selected receivers
    for idx, receiver in enumerate(selected_receivers):
        if idx == 0:
            ax.plot(receiver[0], receiver[1], 'ro', markersize=10, label='Selected Receivers')
        else:
            ax.plot(receiver[0], receiver[1], 'ro', markersize=10)

    # Plot raypaths
    for idx, raypath in enumerate(raypaths):
        raypath_coords = [(j * dx, i * dy) for i, j in raypath]
        raypath_x, raypath_y = zip(*raypath_coords)
        if idx == 0:
            ax.plot(raypath_x, raypath_y, 'r-', alpha=0.8, label='Raypaths')
        else:
            ax.plot(raypath_x, raypath_y, 'r-', alpha=0.8)

    # Normalize T for better visualization and plot it in color with partial transparency
    finite_T = T[np.isfinite(T) & (T > 1e-3)]   
    if finite_T.size > 0:
        max_t = np.max(finite_T)
        if max_t > 0:
            levels = np.linspace(0, max_t, 50)
            ax.contour(T, levels=levels, extent=extent, origin='lower', colors='white', alpha=0.75, linewidths=0.5)
            tof_im = ax.imshow(T, extent=extent, origin='lower', cmap='jet', aspect='equal', alpha=0.2)  # Adjust alpha for transparency
            cax_tof = divider.append_axes("right", size="5%", pad=0.6)
            cbar2 = fig.colorbar(tof_im, cax=cax_tof)
            cbar2.set_label('Time of Flight')

    ax.set_title('Eikonal - Fast Marching Method Results')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    #plt.show()
    log_image(fig)
    
def plot_tof(TOF, title=None):
    # Plot the TOF image

    fig, ax = plt.subplots(figsize=(8, 8))
    # Ensure the y-axis is inverted
    ax.invert_yaxis()
    # ax.imshow(TOF, aspect='auto', origin='lower', cmap='viridis')
    # fig.colorbar(label='Time of Flight')
    cmap = plt.colormaps["viridis"]
    #im = ax.imshow(TOF, origin='lower', cmap=cmap, aspect='equal')
    im = ax.imshow(TOF, cmap=cmap, aspect='equal')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label='Time of Flight (TOF)')
    
    ax.set_xlabel('Emitters')
    ax.set_ylabel('Receivers')
    src = f'{title} Time of Flight (TOF) Image'
    ax.set_title(src)
    #plt.show()
    log_image(fig)
    plt.close()
    
# def plot_sos(speed, dx, dy, mystr):
       
#     # Plot the SOS image
    
#     extent = (0, speed.shape[1] * dx, 0, speed.shape[0] * dy)

#     fig, ax = plt.subplots(figsize=(8, 8))
#     # Reverse the y-axis
#     ax.invert_yaxis()
#     cmap = plt.colormaps["viridis"]
    
#     #im = ax.imshow(speed, extent=extent, origin='lower', cmap=cmap, aspect='equal')
#     im = ax.imshow(speed, extent=extent, cmap=cmap, aspect='equal')
#     cbar = fig.colorbar(im, ax=ax, shrink=0.8, label='Speed of Sound')
    
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_title(f'{mystr} Speed of Sound (SOS) Image')
#     #plt.show()
#     log_image(fig)
#     plt.close()

def plot_sos(speed, dx, dy, mystr):
    
    # Swapped the Y-range in extent:
    # Before: (0, width, 0, height)
    # After: (0, width, height, 0) → This inverts the Y-axis properly.
    
    # Plot the SOS image
    extent = (0, speed.shape[1] * dx, speed.shape[0] * dy, 0)  # Swap Y limits

    fig, ax = plt.subplots(figsize=(8, 8))
    
    cmap = plt.colormaps["viridis"]
    
    # Plot the image with corrected extent
    im = ax.imshow(speed, extent=extent, cmap=cmap, aspect='equal')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label='Speed of Sound')
    
    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'{mystr} Speed of Sound (SOS) Image')

    # Log and close
    log_image(fig)
    plt.close()
