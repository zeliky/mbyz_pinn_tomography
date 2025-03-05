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



        """
            c0 = 0.1200; % speed of sound in air
            c1 = 0.1520;  % speed of sound in "anatomy" 1520-1550  0.0030 in cm/µs.
            c2 = 0.1440;  % speed of sound in fat 1440-1470 in cm/µs.
            c3 = 0.1560;  % speed of sound in cancerous tumors 1550-1600 in cm/µs.
            c4 = 0.1530;  % speed of sound in benign tumors 1530-1580 in cm/µs.
        """
        self.min_sos = app_settings.min_sos
        self.max_sos = app_settings.max_sos

        self.min_tof = app_settings.min_tof
        self.max_tof = app_settings.max_tof


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

        empty_anatomy = self._prepare_mat_data(app_settings.no_anatomy_mat)
        normalized_tof = (empty_anatomy['raw_tof'] - self.min_tof) / (self.max_tof - self.min_tof)
        self.empty_tof = np.expand_dims(np.repeat(np.repeat(normalized_tof, 4, axis=0), 4, axis=1), axis=0)

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
        #tof_img = self._prepare_image(entry['tof'], anatomy_dimensions)
        #tof_img = self._prepare_image(entry['tof'], tof_dimensions)
        #anatomy_img = self._prepare_image(entry['anatomy'], anatomy_dimensions)
        #anatomy_img = self._prepare_image(entry['anatomy'], tof_dimensions)

        #use real tof data instead of the data encoded in the image since resize and image manipulations change the real TOF values
        #tof_img = np.expand_dims(mat_data['raw_tof'], axis=0)
        #raw_tof = mat_data['raw_tof']

        normalized_tof = (mat_data['raw_tof']-self.min_tof)/(self.max_tof-self.min_tof)
        #normalized_tof = std_norm(mat_data['raw_tof'])
        tof_img =  np.expand_dims(np.repeat(np.repeat(normalized_tof, 4, axis=0), 4, axis=1), axis=0)

        #tof_img -= self.empty_tof


        normalized_anatomy = (mat_data['sos'] - self.min_sos) / (self.max_sos - self.min_sos)
        #normalized_anatomy = std_norm(mat_data['sos'])
        anatomy_img = np.expand_dims(normalized_anatomy, axis=0)

        return {
            'anatomy': anatomy_img,
            'tof': tof_img,
            'sos': mat_data['sos'],
            'raw_tof': mat_data['raw_tof'],
            #'raw_sos': mat_data['V'],
            #'expanded_tof': mat_data['expanded_tof'],
            'tof_maps': mat_data['tof_maps'],
            'x_s': mat_data['x_s'],
            'x_r': mat_data['x_r'],
        }


    def _prepare_mat_data(self, path):
        #log_message(f'loading mat file {path}')
        mat_data = loadmat(path)

        source_positions = np.array(mat_data['sources']).transpose() if 'sources' in mat_data else [] # [num_sources, 2]
        receiver_positions =np.array(mat_data['receivers']).transpose() if 'receivers' in mat_data else [] # [num_receivers, 2]
        sos =np.array(mat_data['V'])   if 'V' in mat_data else []  # [num_receivers, 2]

        t_obs = np.array(mat_data['t_obs'])  # [num_sources, num_receivers]
        expanded_tof = np.array(mat_data['exp_t_obs']) if 'exp_t_obs' in mat_data else []# [num_sources* num_receivers , 5]   (xs,ys,xr,yr,tobs(s,r))

        tof_maps = np.array(mat_data['tmap']) if 'tmap' in mat_data else np.array([])# [num_sources, 128x128]


        return {
           'x_s': source_positions,
           'x_r': receiver_positions ,
           'raw_tof' : t_obs,
           'expanded_tof' : expanded_tof,
           'sos' : sos,
           'tof_maps' : tof_maps
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







    def DEP_create_known_indices(self, mat_data):
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


    def DEP_create_boundary_indices(self, mat_data):
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


def std_norm(tensor, method="standard", factor=10.0):
    if method == "standard":
        mean_val = tensor.mean()
        std_val = tensor.std() + 1e-8  # prevent division by zero
        normalized = (tensor - mean_val) / std_val
        return normalized

    elif method == "scale":
        return tensor * factor

    else:
        raise ValueError("Unknown method. Use 'standard' or 'scale'.")