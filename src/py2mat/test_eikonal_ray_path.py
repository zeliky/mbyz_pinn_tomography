import numpy as np
from scipy.io import loadmat
from eikonal_traveltime import eikonal_traveltime
from my_eikonal_ray_path import my_eikonal_ray_path
from logger import log_message, log_image

def prepare_input(D):
    x = D['x'].squeeze()
    y = D['y'].squeeze()
    z = D['z']
    s = D['V']  # Speed of sound (already used correctly)

    xs_sources = D['xs_sources'].squeeze()
    ys_sources = D['ys_sources'].squeeze()
    xs_receivers = D['xs_receivers'].squeeze()
    ys_receivers = D['ys_receivers'].squeeze()

    isrc = 0 #110
    ir = 0 #3

    source = np.array([xs_sources[isrc], ys_sources[isrc]])
    receiver = np.array([xs_receivers[ir], ys_receivers[ir]])

    log_message(f"Receiver: index (ir) = {ir}, coordinates = ({receiver[0]:.4f}, {receiver[1]:.4f})")
    log_message(f"Source: index (isrc) = {isrc}, coordinates =   ({source[0]:.4f}, {source[1]:.4f})")

    # Call with receiver->source convention (as in MATLAB)
    R = np.array([[receiver[0], receiver[1], 1.0]])
    S = np.array([[source[0], source[1], 1.0]])
    tmap, _ = eikonal_traveltime(x, y, z, s, R, S)

    # Do not change the order of source and receiver here
    raylength, raypath = my_eikonal_ray_path(x, y, s, receiver, source, tmap, doPlot=True)

    # if raypath.shape[0] > 0:
    #     log_message(f"Ray path start: ({raypath[0,0]:.4f}, {raypath[0,1]:.4f})")
    #     log_message(f"Ray path end:   ({raypath[-1,0]:.4f}, {raypath[-1,1]:.4f})")
    # else:
    #     log_message("Ray path is empty.")

    # log_message(f"Ray length: {raylength:.2f}")

def load_data():
    mat_path = '../TimeOfFlightData/Time_Of_Flight_data2.mat'
    D = loadmat(mat_path)
    D_in = {key: D[key] for key in D if not key.startswith('__')}
    prepare_input(D_in)

if __name__ == "__main__":
    load_data()

# import numpy as np
# from scipy.io import loadmat
# from eikonal_traveltime import eikonal_traveltime
# from my_eikonal_ray_path import my_eikonal_ray_path
# from visualization import plot_sos
# from logger import log_message, log_image

# def prepare_input(D):
#     x = D['x'].squeeze()
#     y = D['y'].squeeze()
#     z = D['z']
#     s = 1.0 / z  # Slowness
#     s = D['V']
#     xs_sources = D['xs_sources'].squeeze()
#     ys_sources = D['ys_sources'].squeeze()
#     xs_receivers = D['xs_receivers'].squeeze()
#     ys_receivers = D['ys_receivers'].squeeze()

#     isrc = 110
#     ir = 3

#     source = np.array([xs_sources[isrc], ys_sources[isrc]])
#     receiver = np.array([xs_receivers[ir], ys_receivers[ir]])

#     log_message(f"Receiver: index (ir) = {ir}, coordinates = ({receiver[0]:.4f}, {receiver[1]:.4f})")
#     log_message(f"Source: index (isrc) = {isrc}, coordinates =   ({source[0]:.4f}, {source[1]:.4f})")

#     # Call with R, S order to match MATLAB's behavior
#     #tmap, _ = eikonal_traveltime(x, y, z, s, receiver, source)
#     R = np.array([[receiver[0], receiver[1], 1.0]])
#     S = np.array([[source[0], source[1], 1.0]])
#     tmap, _ = eikonal_traveltime(x, y, z, s, R, S)

#     #raylength, raypath = my_eikonal_ray_path(x, y, s, source, receiver, tmap, doPlot=True)
#     raylength, raypath = my_eikonal_ray_path(x, y, s, receiver, source, tmap, doPlot=True)

#     log_message(f"Ray path start: ({raypath[0,0]:.4f}, {raypath[0,1]:.4f})")
#     log_message(f"Ray path end:   ({raypath[-1,0]:.4f}, {raypath[-1,1]:.4f})")
#     log_message(f"Ray length: {raylength:.2f}")

#     # Additional plot (optional, since log_image is already handled inside my_eikonal_ray_path)
#     # fig = plot_sos(x, y, s)
#     # plt.plot(raypath[:, 0], raypath[:, 1], color='cyan')
#     # plt.plot(source[0], source[1], 'go')
#     # plt.plot(receiver[0], receiver[1], 'ro')
#     # plt.title("Ray Path Overlay")
#     # log_image(fig)

# def load_data():
#     mat_path = '../TimeOfFlightData/Time_Of_Flight_data2.mat'
#     D = loadmat(mat_path)
#     D_in = {key: D[key] for key in D if not key.startswith('__')}
#     prepare_input(D_in)

# if __name__ == "__main__":
#     load_data()

# import numpy as np
# import scipy.io
# import matplotlib.pyplot as plt
# from my_eikonal_ray_path import my_eikonal_ray_path
# from eikonal_traveltime import eikonal_traveltime
# from logger import log_message, log_image

# def prepare_input(D):
#     x = D['x'].flatten()
#     y = D['y'].flatten()
#     z = D['z'].flatten()
#     s = D['V']

#     # Flatten source/receiver coordinates in case they are shaped (1, N)
#     xs_sources = D['xs_sources'].flatten()
#     ys_sources = D['ys_sources'].flatten()
#     xs_receivers = D['xs_receivers'].flatten()
#     ys_receivers = D['ys_receivers'].flatten()

#     isrc = 110
#     ir = 3

#     source = np.array([xs_sources[isrc], ys_sources[isrc]])
#     receiver = np.array([xs_receivers[ir], ys_receivers[ir]])

#     log_message(f"Receiver: index (ir) = {ir}, coordinates = ({receiver[0]:.4f}, {receiver[1]:.4f})")
#     log_message(f"Source: index (isrc) = {isrc}, coordinates =   ({source[0]:.4f}, {source[1]:.4f})")

#     R = np.array([[receiver[0], receiver[1], 1.0]])
#     S = np.array([[source[0], source[1], 1.0]])

#     tmap, _ = eikonal_traveltime(x, y, z, s, R, S)
#     #tmap, _ = eikonal_traveltime(x, y, z, s, S, R)

#     raylength, raypath = my_eikonal_ray_path(x, y, s, source, receiver, tmap, doPlot=False)

#     log_message(f"Ray path start: ({raypath[0,0]:.4f}, {raypath[0,1]:.4f})")
#     log_message(f"Ray path end:   ({raypath[-1,0]:.4f}, {raypath[-1,1]:.4f})")
#     log_message(f"Ray length: {raylength:.2f}")

#     # Visualization
#     fig, ax = plt.subplots(figsize=(8, 6))
#     im = ax.imshow(s, extent=[x.min(), x.max(), y.min(), y.max()],
#                    origin='lower', aspect='equal', cmap='viridis')
#     plt.colorbar(im, ax=ax, label='Slowness')

#     # Overlay ray path
#     ax.plot(raypath[:, 0], raypath[:, 1], 'w-', linewidth=2, label='Ray path')

#     # Overlay selected source and receiver
#     ax.plot(source[0], source[1], 'go', markersize=8, label='Source')
#     ax.plot(receiver[0], receiver[1], 'ro', markersize=8, label='Receiver')

#     # Overlay all sources and receivers
#     ax.plot(xs_sources, ys_sources, 'g.', markersize=4, label='All Sources')
#     ax.plot(xs_receivers, ys_receivers, 'r.', markersize=4, label='All Receivers')

#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_title('Eikonal Ray Path over Slowness Map')
#     ax.legend()

#     # Log the figure
#     log_image(fig)

# def load_data():
#     mat = scipy.io.loadmat("../TimeOfFlightData/Time_Of_Flight_data2.mat")
#     D = {k: v for k, v in mat.items() if not k.startswith("__")}
#     prepare_input(D)

# if __name__ == "__main__":
#     load_data()


# import numpy as np
# import scipy.io
# from my_eikonal_ray_path import my_eikonal_ray_path
# from eikonal_traveltime import eikonal_traveltime
# from logger import log_message, log_image

# #from visualization import plot_sos  # Make sure this is imported
# import matplotlib.pyplot as plt

# def prepare_input(D_in):
#     D = D_in
#     x, y, z = D['x'].squeeze(), D['y'].squeeze(), D['z'].squeeze()
#     dx = x[1]-x[0]
#     dy = y[1]-y[0]
    
#     s = D['V']
#     xs_receivers, ys_receivers = D['xs_receivers'].squeeze(), D['ys_receivers'].squeeze()
#     xs_sources, ys_sources = D['xs_sources'].squeeze(), D['ys_sources'].squeeze()
    
#     ir = 3
#     isrc = 110
#     number_of_sources = len(xs_sources)

#     S = np.column_stack((xs_sources, ys_sources))
#     R1 = np.column_stack((
#         np.full((number_of_sources,), xs_receivers[ir]),
#         np.full((number_of_sources,), ys_receivers[ir]),
#         np.ones(number_of_sources)
#     ))
    
#     tmap, _ = eikonal_traveltime(x, y, z, s, R1, S)

#     receiver = np.array([xs_receivers[ir], ys_receivers[ir]])
#     source = np.array([xs_sources[isrc], ys_sources[isrc]])

#     raylength, raypath = my_eikonal_ray_path(x, y, s, receiver, source, tmap, doPlot=True)

#     # Plot SoS with sensors and ray path
#     mystr = "test"
    
#     speed = s
    
#     # Plot the SOS image
#     extent = (0, speed.shape[1] * dx, speed.shape[0] * dy, 0)  # Swap Y limits

#     fig, ax = plt.subplots(figsize=(8, 8))
    
#     cmap = plt.colormaps["viridis"]
    
#     # Plot the image with corrected extent
#     im = ax.imshow(speed, extent=extent, cmap=cmap, aspect='equal')
    
#     # Add colorbar
#     cbar = fig.colorbar(im, ax=ax, shrink=0.8, label='Speed of Sound')
    
#     ax.plot(xs_sources, ys_sources, 'go', label='Sources', markersize=4)
#     ax.plot(xs_receivers, ys_receivers, 'ro', label='Receivers', markersize=4)
#     ax.plot(raypath[:, 0], raypath[:, 1], 'w-', linewidth=2, label='Ray Path')
#     ax.plot(source[0], source[1], 'go', markersize=8)    # Mark source
#     ax.plot(receiver[0], receiver[1], 'ro', markersize=8)  # Mark receiver
#     ax.legend()
#     # Labels and title
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_title(f'{mystr} Speed of Sound (SOS) Image')
    
#     # plt.show()
#     log_image(fig)
    
#     log_message(f"Receiver: index (ir) = {ir}, coordinates = ({receiver[0]:.4f}, {receiver[1]:.4f})")
#     log_message(f"Source: index (isrc) = {isrc}, coordinates =   ({source[0]:.4f}, {source[1]:.4f})")
#     log_message(f"Ray path start: ({raypath[0, 0]:.4f}, {raypath[0, 1]:.4f})")
#     log_message(f"Ray path end:   ({raypath[-1, 0]:.4f}, {raypath[-1, 1]:.4f})")

# def load_data():
#     D_in = scipy.io.loadmat('../TimeOfFlightData/Time_Of_Flight_data2.mat')
#     D_in['use_input_s0'] = False
#     prepare_input(D_in)

# if __name__ == "__main__":
#     load_data()