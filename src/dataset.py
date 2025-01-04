import os
import re
import math
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
import math
class TofDataset(Dataset):
    def __init__(self, modes, **kwargs):
        super().__init__()
        self.modes = modes
        self.file_index = {}

        self.sources_amount = kwargs.get('sources_amount', 32)
        self.receivers_amount = kwargs.get('receivers_amount', 32)
        self.anatomy_width = kwargs.get('anatomy_width', 128.0)
        self.anatomy_height = kwargs.get('anatomy_height', 128.0)


        self.min_sos = kwargs.get('min_sos', 1400)   # speed of sound in anatomy
        self.max_sos = kwargs.get('min_sos', 1450)   # speed of sound in background

        self.min_tof = kwargs.get('min_tof', 0)
        self.max_tof = kwargs.get('max_tof', 100)
        self.tof_scale_factor = kwargs.get('tof_scale_factor', 1e-4)
        self.sos_scale_factor = kwargs.get('tof_scale_factor', 1e-3)

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
        self.empty_tof = self._prepare_image(app_settings.no_anatomy_img, (int(self.anatomy_height), int(self.anatomy_width)) )

    @staticmethod
    def load_dataset(source_file):
        with open(source_file, 'rb') as file:
            return pickle.load(file)

    def __len__(self):
        return len(self.files_index)

    def __getitem__(self, idx):
        entry = self.files_index[idx]
        anatomy_dimensions = (int(self.anatomy_height), int(self.anatomy_width))
        tof_dimensions = (int(self.sources_amount), int(self.receivers_amount))

        #prepare tof values
        mat_data = self._prepare_mat_data( entry['mat'])

        # load images
        tof_img = self.empty_tof - self._prepare_image(entry['tof'], anatomy_dimensions)
        #tof_img = self._prepare_image(entry['tof'], tof_dimensions)
        anatomy_img = self._prepare_image(entry['anatomy'], anatomy_dimensions)
        anatomy_img_sml = self._prepare_image(entry['anatomy'], tof_dimensions)
        #anatomy_img = self._prepare_image(entry['anatomy'], tof_dimensions)

        #use real tof data instead of the data encoded in the image since resize and image manipulations change the real TOF values
        #tof_img = np.expand_dims(mat_data['raw_tof'], axis=0)
        raw_tof = mat_data['raw_tof']
        return {
            'anatomy': anatomy_img,
            'anatomy_sml': anatomy_img_sml,
            'tof': tof_img,
            'raw_tof': raw_tof,
            'x_s': mat_data['x_s'],
            'x_r': mat_data['x_r'],
            'positions_mask': mat_data['positions_mask'],
        }


    def _prepare_mat_data(self, path):
        #log_message(f'loading mat file {path}')
        mat_data = loadmat(path)

        xs_sources = np.array(np.round(mat_data['xs_sources'])  ).astype(int).flatten()
        ys_sources = np.array(np.round(mat_data['ys_sources']) ).astype(int).flatten()
        source_positions = np.column_stack([xs_sources, ys_sources])  # [num_sources, 2]

        xs_receivers = np.array(np.round(mat_data['xs_receivers']) ).astype(int).flatten()
        ys_receivers = np.array(np.round(mat_data['ys_receivers']) ).astype(int).flatten()
        receiver_positions = np.column_stack([xs_receivers, ys_receivers])  # [num_receivers, 2]

        tof_to_receivers = np.array((mat_data['t_obs'] / self.max_tof))  # [num_sources, num_receivers]
        num_sources = self.sources_amount
        num_receivers = self.receivers_amount

        tof_mask = np.zeros([int(self.anatomy_width),int(self.anatomy_height)])
        for j in range(num_receivers):
            x, y = receiver_positions[j]
            tof_mask[x-1, y-1] = 100+j
        for j in range(num_sources):
            x, y = source_positions[j]
            tof_mask[x-1, y-1] = j

        return {
           'x_s': source_positions,
           'x_r': receiver_positions,
           'positions_mask': tof_mask,
           'raw_tof' : tof_to_receivers
        }

    def _prepare_image(self, path, dimensions):
        transform = transforms.Compose([
            transforms.Resize((int(dimensions[0]), int(dimensions[1]))),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

        img = Image.open(path)
        image_tensor = transform(img)  # [1,H,W], normalized to [0,1]
        image_tensor = image_tensor
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

        self.empty_tof = self._prepare_image(app_settings.no_anatomy_img,
                                             (int(self.anatomy_height), int(self.anatomy_width)))

    def _build_mat_files_index(self):
        pattern = re.compile(r'ToF(.*)_(\d+)\.mat')
        for file_name in os.listdir(self.tof_path):
            match = pattern.match(file_name)
            if match:
                tumor_id = match.group(2)
                if tumor_id in self.file_index:
                    self.file_index[tumor_id]['mat'] = os.path.join(self.tof_path, file_name)





    def _generate_tof_tensor(self,mat_data):

        # Extract data from mat_data
        xs_sources = mat_data['xs_sources'].squeeze()
        ys_sources = mat_data['ys_sources'].squeeze()
        xs_receivers = mat_data['xs_receivers'].squeeze()
        ys_receivers = mat_data['ys_receivers'].squeeze()
        t_obs = mat_data['t_obs']

        num_sources = len(xs_sources)
        num_receivers = len(xs_receivers)

        # Initialize the tensor with shape (num_sources, W, H)
        tof_tensor = torch.ones((num_sources, int(self.anatomy_width), int(self.anatomy_height)))

        # Populate the tensor
        for source_idx in range(num_sources):
            # Get source coordinates
            source_x = math.floor(xs_sources[source_idx])
            source_y = math.floor(ys_sources[source_idx])

            # Place -1 at the source's location
            tof_tensor[source_idx, source_y, source_x] = -1

            for receiver_idx in range(num_receivers):
                # Get receiver coordinates
                receiver_x = math.floor(xs_receivers[receiver_idx])
                receiver_y = math.floor(ys_receivers[receiver_idx])

                # Place normalized ToF value at the receiver's location
                tof_tensor[source_idx, receiver_y, receiver_x] = t_obs[source_idx, receiver_idx] * self.tof_scale_factor

        return tof_tensor

    def _create_known_indices(self, mat_data):
        xs_sources = mat_data['xs_sources'].squeeze().astype(int)
        xs_receivers = mat_data['xs_receivers'].squeeze().astype(int)
        ys_receivers = mat_data['ys_receivers'].squeeze().astype(int)

        num_sources = len(xs_sources)
        num_receivers = len(xs_receivers)

        # Initialize mask tensor with False
        mask = torch.zeros((num_sources, int(self.anatomy_width), int(self.anatomy_height)), dtype=torch.bool)

        for source_idx in range(num_sources):
            for receiver_idx in range(num_receivers):
                receiver_x = xs_receivers[receiver_idx]
                receiver_y = ys_receivers[receiver_idx]
                mask[source_idx, receiver_y, receiver_x] = True


        return mask


    def _create_boundary_indices(self, mat_data):
        xs_sources = mat_data['xs_sources'].squeeze().astype(int)
        ys_sources = mat_data['ys_sources'].squeeze().astype(int)
        xs_receivers = mat_data['xs_receivers'].squeeze().astype(int)
        ys_receivers = mat_data['ys_receivers'].squeeze().astype(int)

        num_sources = len(xs_sources)
        num_receivers = len(xs_receivers)

        # Initialize mask tensor with False
        mask = torch.zeros(( num_sources, int(self.anatomy_width), int(self.anatomy_height)), dtype=torch.bool)

        for source_idx in range(num_sources):

            # Mark the source location
            source_x = xs_sources[source_idx].astype(int)
            source_y = ys_sources[source_idx].astype(int)
            # Mark sources positions
            mask[ source_idx, source_y, source_x] = True

            # Mark all receiver locations for this source
            for receiver_idx in range(num_receivers):
                receiver_x = xs_receivers[receiver_idx]
                receiver_y = ys_receivers[receiver_idx]
                mask[source_idx, receiver_y, receiver_x] = True
        return mask
