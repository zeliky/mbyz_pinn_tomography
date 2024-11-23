import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from PIL import Image
import cv2

class TofDataset(Dataset):
    def __init__(self, modes, p, anatomy_width=128.0, anatomy_height=128.0, grid_size=128):
        super().__init__()
        self.p = p
        self.modes = modes
        self.file_index = {}
        self.tof_images = []
        self.anatomy_images = []
        self.tof_data = []
        self.sos_data = []
        self.anatomy_width = anatomy_width
        self.anatomy_height = anatomy_height
        self.min_sos = 0.145  # speed of sound in water (background)
        self.max_sos = 0.14   # speed of sound in anatomy
        self.grid_size = grid_size
        self.modes_path = {
            'train': '../inputData/ForLearning/',
            'validation': '../inputData/ForValidation/',
            'test': '../inputData/ForTest/'
        }
        self.tof_path = '../inputData/TimeOfFlightData/'
        self.load()
        

    def __len__(self):
        return len(self.tof_data)

    def __getitem__(self, idx):
        item = self.tof_data[idx]
        sos = self.sos_data[idx]
        return {
            'coords': torch.tensor(item['coords'], dtype=torch.float32),
            'source_positions': torch.tensor(item['source_positions'], dtype=torch.float32),
            'source_values': torch.tensor(item['source_values'], dtype=torch.float32),
            'receiver_positions': torch.tensor(item['receiver_positions'], dtype=torch.float32),
            'receiver_values': torch.tensor(item['receiver_values'], dtype=torch.float32),
            'sos_values': torch.tensor(sos, dtype=torch.float32)
        }

    def load(self):
        """
        Load all data for the specified modes.
        """
        self.build_files_index(self.modes)
        for mode in self.modes:
            self._load_data(mode)

    def build_files_index(self, modes):
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
                            self.file_index[tumor_id] = {'anatomy': None, 'tof': None, 'tof_data': None}
                        self.file_index[tumor_id][key] = os.path.join(base_path, file_name)
            self._add_tof_data()

    def _add_tof_data(self):
        """
        Add ToF data files to the file index.
        """
        pattern = re.compile(r'ToF(.*)_(\d+)\.mat')
        for file_name in os.listdir(self.tof_path):
            match = pattern.match(file_name)
            if match:
                tumor_id = match.group(2)
                if tumor_id in self.file_index:
                    self.file_index[tumor_id]['tof_data'] = os.path.join(self.tof_path, file_name)

    def _load_data(self, mode):
        """
        Load anatomy, ToF, and speed-of-sound data for the specified mode.
        """
        self.p.print(f"[dataset.py] Loading data for mode: {mode}...")
        for tumor_id, paths in self.file_index.items():
            if not all(paths.values()):
                continue
            anatomy_img = self._prepare_image_data(paths['anatomy'])
            tof_img = self._prepare_image_data(paths['tof'])  # Fix applied here
            sos_data = self._prepare_sos_data(paths['anatomy'])
            mat_data = loadmat(paths['tof_data'])
            tof_data = self._prepare_tof_data(mat_data)
            self.tof_images.append(tof_img)
            self.anatomy_images.append(anatomy_img)
            self.sos_data.append(sos_data)
            self.tof_data.append(tof_data)
        self.p.print("[dataset.py] Data loading complete.")

    def _prepare_image_data(self, path):
        """
        Read an image, convert it to grayscale, resize it, and return as a NumPy array.
        """
        # self.p.print(f"[dataset.py] Preparing image data for path: {path}")
        # Ensure the image is loaded as grayscale
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image file not found at path: {path}")
        
        # Resize the image to match grid_size
        image_resized = cv2.resize(image, (self.grid_size, self.grid_size), interpolation=cv2.INTER_AREA)
        return image_resized

    def _prepare_sos_data(self, path):
        """
        Prepare speed-of-sound (SoS) data from an anatomy image.
        """
        image = Image.open(path).resize((self.grid_size, self.grid_size)).convert('L')
        intensity_array = np.array(image, dtype=np.float32) / 255.0
        sos_data = (self.min_sos + intensity_array * (self.max_sos - self.min_sos))
        return sos_data.flatten()  # Ensure sos_values is a 1D array

    def _prepare_tof_data(self, mat_data):
        """
        Prepare ToF data from the MATLAB file.
        """
        xs_sources = np.array(mat_data['xs_sources']).flatten()
        ys_sources = np.array(mat_data['ys_sources']).flatten()
        xs_receivers = np.array(mat_data['xs_receivers']).flatten()
        ys_receivers = np.array(mat_data['ys_receivers']).flatten()
        coords = np.column_stack([np.linspace(0, self.anatomy_width, self.grid_size).repeat(self.grid_size),
                                  np.tile(np.linspace(0, self.anatomy_height, self.grid_size), self.grid_size)])
        return {
            'coords': coords,
            'source_positions': np.column_stack([xs_sources, ys_sources]),
            'source_values': np.zeros(len(xs_sources)),
            'receiver_positions': np.column_stack([xs_receivers, ys_receivers]),
            'receiver_values': np.array(mat_data['t_obs']).flatten()
        }
