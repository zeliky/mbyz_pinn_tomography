import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
from dataset import TofDataset
from models.eikonal_solver import EikonalSolverMultiLayer
from settings import  app_settings
from physics import _to_mps, _to_sec, eikonal_loss_multi
import matplotlib.pyplot as plt
from logger import log_image

source= (2,64)
solver = EikonalSolverMultiLayer(num_layers=3, speed_of_sound=1450, domain_size=0.128, grid_resolution=128)
dataset = TofDataset(['train'])
#d = dataset.__getitem__(idx =1)

#sos_pred = _to_mps(d['anatomy']).unsqueeze(0)

#print(f"Speed of sound range: {sos_pred.min().item()} - {sos_pred.max().item()}")

#T_init = torch.full((1, 1, app_settings.anatomy_width,app_settings.anatomy_height), 1e18, device=sos_pred.device)
#T_init[0, 0, source[0], source[1]] = 0  # Source at zero



"""
last_s = (None,None)
for xs,ys,xr,yr,tof in d['known_tof']:
    new_s = (int(xs), int(ys))
    if last_s[0]!= new_s[0] or last_s[1]!= new_s[1]:
        print(new_s)
        print('----------------------------------')
        T_init = torch.full((1, 1, app_settings.anatomy_width, app_settings.anatomy_height), 1e18, device=sos_pred.device)
        T_init[0, 0, new_s[0], new_s[1]] = 0  # Source at zero
        T = solver(T_init, sos_pred)
    last_s = new_s

    p = T[0,0,int(xr-1),int(yr-1)].item()
    r = tof*1e-7
    diff = (max(p,r)-min(p,r))/max(p,r)*100
    print(f"{xs},{ys} {xr},{yr}  <=> {tof*1e-7} :  {diff}%")
#print(T.tolist())

"""


val_loader = DataLoader(dataset, batch_size=1, shuffle=False)
for batch in val_loader:

    sos_pred = batch['anatomy']
    sources = batch['x_s']

    for src in sources[0].squeeze():
        s = (int(src[0]), int(src[1]))
        loss = eikonal_loss_multi(sos_pred, solver, s, roi_start=40, roi_end=80, eps=1e-8)
        print(loss)

    break


    """
    for known_tof in known_tofs:
        for xs, ys, xr, yr, tof in known_tof:
            new_s = (int(xs), int(ys))
            if last_s[0] != new_s[0] or last_s[1] != new_s[1]:
                print(new_s)
                print('----------------------------------')
                e
            print(xs, ys, xr, yr, tof)
    """