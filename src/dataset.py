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
        self.max_sos = kwargs.get('min_sos', 0.145)    # speed of sound in water (background)

        self.min_tof = kwargs.get('min_tof', 0)  # min observed tof ???
        self.max_tof = kwargs.get('max_tof', 100)  # max observed tof ???

        self.modes_path = {
            'train': app_settings.train_path,
            'validation': app_settings.validation_path,
            'test': app_settings.test_path
        }
        self.tof_path = app_settings.tof_path

        self._build_files_index(self.modes)

    @staticmethod
    def load_dataset(source_file):
        with open(source_file, 'rb') as file:
            return pickle.load(file)
    def __len__(self):
        return len(self.file_index)

    def __getitem__(self, idx):
        paths = self.file_index[idx]
        tof_data = self._prepare_mat_data(paths['mat'])
        tof_input = self._prepare_image(paths['tof'],(self.sources_amount, self.receivers_amount), self.min_tof, self.max_tof)
        sos_image = self._prepare_image(paths['anatomy'], (int(self.anatomy_width), int(self.anatomy_height)), self.min_sos, self.max_sos)

        return {
            'tof_data': tof_data,
            'tof_input': tof_input,
            'sos_image':  sos_image
        }

    def _prepare_mat_data(self, path):
        mat_data = loadmat(path)

        xs_sources = np.array(mat_data['xs_sources']).flatten()
        ys_sources = np.array(mat_data['ys_sources']).flatten()
        source_positions = np.column_stack([xs_sources, ys_sources])

        xs_receivers = np.array(mat_data['xs_receivers']).flatten()
        ys_receivers = np.array(mat_data['ys_receivers']).flatten()
        receiver_positions = np.column_stack([xs_receivers, ys_receivers])

        tof_to_receivers = np.array(mat_data['t_obs'])

        source_indices, receiver_indices = np.meshgrid(np.arange(self.sources_amount), np.arange(self.receivers_amount), indexing='ij')
        source_indices = source_indices.flatten()
        receiver_indices = receiver_indices.flatten()

        source_coords = source_positions[source_indices]  # Shape: [num_pairs, 2]
        receiver_coords = receiver_positions[receiver_indices]  # Shape: [num_pairs, 2]
        observed_tof = tof_to_receivers[source_indices, receiver_indices]  # Shape: [num_pairs]

        x_s = torch.tensor(source_coords, dtype=torch.float32)  # Shape: [num_pairs, 2]
        x_r = torch.tensor(receiver_coords, dtype=torch.float32)  # Shape: [num_pairs, 2]
        x_o = torch.tensor(observed_tof, dtype=torch.float32) # Shape: [num_pairs]

        # return  torch.cat([x_r, x_s], dim=1)
        return {
            'x_s': x_s,
            'x_r': x_r,
            'x_o': x_o
        }

    def _prepare_image(self, path, dimensions, min_value, max_value):
        transform = transforms.Compose([
            transforms.Resize(dimensions ),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        img  = Image.open(path)
        image_tensor = transform(img)
        physical_units = image_tensor * (max_value - min_value) + min_value
        return  physical_units

    def _build_files_index(self, modes):
        """
        Build a file index mapping tumor IDs to their anatomy, ToF, and ToF data files.
        """
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
        self.file_index = index


    def _build_mat_files_index(self):
        pattern = re.compile(r'ToF(.*)_(\d+)\.mat')
        for file_name in os.listdir(self.tof_path):
            match = pattern.match(file_name)
            if match:
                tumor_id = match.group(2)
                if tumor_id in self.file_index:
                    self.file_index[tumor_id]['mat'] = os.path.join(self.tof_path, file_name)



