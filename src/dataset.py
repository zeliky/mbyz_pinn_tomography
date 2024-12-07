import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.io import loadmat
from PIL import Image
import cv2
import pickle
from logger import log_message
from settings import  app_settings

class TofDataset(Dataset):
    def __init__(self, modes, **kwargs):
        super().__init__()
        self.modes = modes
        self.file_index = {}

        self.sources_amount = kwargs.get('sources_amount', 32)
        self.receivers_amount = kwargs.get('receivers_amount', 32)
        self.anatomy_width = kwargs.get('anatomy_width', 128.0)
        self.anatomy_height = kwargs.get('anatomy_height', 128.0)

        self.grid_size = kwargs.get('grid_size', 128.0)

        self.min_sos = kwargs.get('min_sos', 0.140)   # speed of sound in anatomy
        self.max_sos = kwargs.get('min_sos', 0.145)   # speed of sound in background

        self.min_tof = kwargs.get('min_tof', 0)
        self.max_tof = kwargs.get('max_tof', 100)

        self.modes_path = {
            'train': app_settings.train_path,
            'validation': app_settings.validation_path,
            'test': app_settings.test_path
        }
        self.tof_path = app_settings.tof_path

        self.x_s_list = []
        self.x_r_list = []
        self.x_o_list = []

        self._build_files_index(self.modes)

    @staticmethod
    def load_dataset(source_file):
        with open(source_file, 'rb') as file:
            return pickle.load(file)

    def __len__(self):
        return len(self.files_index)

    def __getitem__(self, idx):
        entry = self.files_index[idx]

        # Use unified dimensions for all images (anatomy dimensions)
        anatomy_dimensions = (int(self.anatomy_height), int(self.anatomy_width))

        # Prepare images
        tof_data = self._prepare_image(entry['tof'], anatomy_dimensions)         # [1,H,W]
        anatomy_data = self._prepare_image(entry['anatomy'], anatomy_dimensions) # [1,H,W]

        # Load mat data
        mat_data = self._prepare_mat_data(entry['mat'])
        x_s = mat_data['x_s']  # [num_pairs, 2]
        x_r = mat_data['x_r']  # [num_pairs, 2]
        x_o = mat_data['x_o']  # [num_pairs]

        # Create source and receiver maps
        # Dimensions
        H, W = anatomy_dimensions
        source_map = torch.zeros((1, H, W), dtype=torch.float32)
        receiver_map = torch.zeros((1, H, W), dtype=torch.float32)


        #  place all pairs (x_s[i], x_r[i]):
        # Coordinates are normalized [0,1], convert to pixel indices:
        # pixel_x = round(x * (W-1))
        # pixel_y = round(y * (H-1))

        # Place sources
        x_s_coords = x_s.detach().cpu().numpy()
        x_r_coords = x_r.detach().cpu().numpy()

        for (sx, sy) in x_s_coords:
            px = min(W-1, max(0, int(round(sx * (W-1)))))
            py = min(H-1, max(0, int(round(sy * (H-1)))))
            source_map[0, py, px] = 1.0

        # Place receivers
        for (rx, ry) in x_r_coords:
            px = min(W-1, max(0, int(round(rx * (W-1)))))
            py = min(H-1, max(0, int(round(ry * (H-1)))))
            receiver_map[0, py, px] = 1.0

        # Combine all channels: [anatomy, tof, source_map, receiver_map]
        # anatomy_data: [1,H,W]
        # tof_data: [1,H,W]
        # source_map: [1,H,W]
        # receiver_map: [1,H,W]
        inputs = torch.cat([tof_data, source_map, receiver_map], dim=0)  # [3,H,W]

        return {
            'inputs': inputs,
            'tof': tof_data,
            'anatomy': anatomy_data,
            'x_s': x_s,
            'x_r': x_r,
            'x_o': x_o
        }

    def _prepare_mat_data(self, path):
        #log_message(f'loading mat file {path}')
        mat_data = loadmat(path)

        xs_sources = np.array(mat_data['xs_sources'] / self.anatomy_width).flatten()
        ys_sources = np.array(mat_data['ys_sources'] / self.anatomy_height).flatten()
        source_positions = np.column_stack([xs_sources, ys_sources])  # [num_sources, 2]

        xs_receivers = np.array(mat_data['xs_receivers'] / self.anatomy_width).flatten()
        ys_receivers = np.array(mat_data['ys_receivers'] / self.anatomy_height).flatten()
        receiver_positions = np.column_stack([xs_receivers, ys_receivers])  # [num_receivers, 2]

        tof_to_receivers = np.array(mat_data['t_obs'] / self.max_tof)  # [num_sources, num_receivers]

        source_indices, receiver_indices = np.meshgrid(
            np.arange(self.sources_amount),
            np.arange(self.receivers_amount),
            indexing='ij'
        )
        source_indices = source_indices.flatten()
        receiver_indices = receiver_indices.flatten()

        x_s_pairs = source_positions[source_indices]     # [num_pairs, 2]
        x_r_pairs = receiver_positions[receiver_indices] # [num_pairs, 2]
        x_o_pairs = tof_to_receivers[source_indices, receiver_indices] # [num_pairs]

        x_s = torch.tensor(x_s_pairs, dtype=torch.float32)
        x_r = torch.tensor(x_r_pairs, dtype=torch.float32)
        x_o = torch.tensor(x_o_pairs, dtype=torch.float32)

        return {
            'x_s': x_s,
            'x_r': x_r,
            'x_o': x_o
        }

    def _prepare_image(self, path, dimensions):
        transform = transforms.Compose([
            transforms.Resize((int(dimensions[0]), int(dimensions[1]))),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
        img = Image.open(path)
        image_tensor = transform(img)  # [1,H,W], normalized to [0,1]
        return image_tensor

    def _build_files_index(self, modes):
        patterns = {
            'anatomy': re.compile(r'anatomy(.*)_(\d+)\.png'),
            'tof': re.compile(r'tof(.*)_(\d+)\.png')
        }
        for mode in modes:
            base_path = self.modes_path[mode]
            for file_name in os.listdir(base_path):
                for key, pattern in patterns.items():
                    match = pattern.match(file_name)
                    if match:
                        tumor_id = match.group(2)
                        if tumor_id not in self.file_index:
                            self.file_index[tumor_id] = {'anatomy': None, 'tof': None, 'mat': None}
                        self.file_index[tumor_id][key] = os.path.join(base_path, file_name)
            self._build_mat_files_index()

        index = {}
        for i,v in enumerate(self.file_index.values()):
            index[i] = v
        self.files_index = index

    def _build_mat_files_index(self):
        pattern = re.compile(r'ToF(.*)_(\d+)\.mat')
        for file_name in os.listdir(self.tof_path):
            match = pattern.match(file_name)
            if match:
                tumor_id = match.group(2)
                if tumor_id in self.file_index:
                    self.file_index[tumor_id]['mat'] = os.path.join(self.tof_path, file_name)




