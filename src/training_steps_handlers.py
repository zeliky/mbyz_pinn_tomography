import torch.nn as nn
import torch
import numpy as np
from physics import Solver, eikonal_loss_multi, eikonal_loss, initial_loss, boundary_loss,_to_mps
from graph.network import GraphDataset
from logger import log_message, log_image
from settings import  app_settings
import random
import math


class BaseTrainingStep:
    def __init__(self):
        self.model = None
        self.device = None
        self.criterion = None
        self.w_criterion = None

    def init(self, model, device):
        self.model = model
        self.device = device
        self.criterion = nn.MSELoss()
        self.w_criterion = MultiRangeWeightedMSELoss()

    def set_train_mode(self):
        self.model.train()

    def set_eval_mode(self):
        self.model.eval()

    def perform_step(self, batch):
        pass
    def eval_model(self,batch):
        input_args = self.training_step_handler.get_model_input_data(batch)
        return self.model(*input_args)

    def get_model_input_data(self, batch):
        return batch['tof'].float().to(self.device)





class MultiRangeWeightedMSELoss(nn.Module):
    def __init__(self):
        """
        ranges_weights: List of (low_threshold, high_threshold, weight) tuples
        default_weight: Weight for values outside all specified ranges
        """
        super(MultiRangeWeightedMSELoss, self).__init__()
        self.min_sos = app_settings.min_sos
        self.max_sos = app_settings.max_sos
        """
            c0 = 0.1200; % speed of sound in air
            c2 = 0.1440;  % speed of sound in fat 1440-1470 in cm/µs.
            c1 = 0.1520;  % speed of sound in "anatomy" 1520-1550  0.0030 in cm/µs.            
            c3 = 0.1560;  % speed of sound in cancerous tumors 1550-1600 in cm/µs.
            c4 = 0.1530;  % speed of sound in benign tumors 1530-1580 in cm/µs.
        """
        ranges_weights =[
            (self.min_sos,   0.1400, 0),
            (0.1400,         0.4900, 0),
            (0.1500,         0.1550, 500),
            (0.1550,         self.max_sos, 1000)  # cancerous tumour
        ]
        self.ranges_weights= []
        for min_val, max_val, weight in ranges_weights:
            self.ranges_weights.append((
                self._normalize_sos(min_val),
                self._normalize_sos(max_val),
                weight
            ))
        self.default_weight = 0

    def _normalize_sos(self,val):
        return (val - self.min_sos) / (self.max_sos - self.min_sos)
    def forward(self, pred, target):
        mse = (pred - target) ** 2  # Standard MSE

        # Initialize weight map with default weight
        weight_map = torch.full_like(target, self.default_weight)

        # Apply weights for each range
        #cnt = target.size(2) * target.size(3)
        for low, high, weight in self.ranges_weights:
            mask = (target > low) & (target < high)
            #nz = torch.count_nonzero(mask)
            #print(f"{low}-{high} count {nz} prec : {nz/cnt}")
            weight_map[mask] += weight
        #print("---------------------------------------")
        
        # Apply weighted MSE
        weighted_mse = mse * weight_map

        # Normalize by the number of elements
        return weighted_mse.mean()



class DualHeadGATTrainingStep(BaseTrainingStep):
    def __init__(self, ** kwargs):
        super().__init__()
        c_init = kwargs.get('c_init', 0.12)
        self.x_range = kwargs.get('x_range', (32, 96))
        self.y_range = kwargs.get('y_range', (32, 96))
        self.nx = kwargs.get('nx', 64)
        self.ny = kwargs.get('ny', 64)

        self.gd = GraphDataset(c_init=c_init,x_range=self.x_range,y_range=self.y_range, nx=self.nx, ny=self.ny)


    def perform_step(self, batch):
        sources_positions = batch['x_s'].squeeze()
        receivers_positions = batch['x_r'].squeeze()
        tof = batch['raw_tof'].squeeze().float().to(self.device)

        all_tof_maps = batch['tof_maps'].squeeze().float().to(self.device)
        sos = batch['sos'].squeeze().float().to(self.device)
        c_true = self._get_region_of_interest(sos)
        # build the source graph only once
        if not self.gd.initialized:
            self.gd.build(sources_positions, receivers_positions)

        num_sources = 10
        selected_sources = random.sample(range(32), k=num_sources)


        sos_list = []
        loss_tof_total = 0.0
        loss_bc_total = 0.0
        pde_loss_total = 0.0
        for i, data in self.gd.get_graph(tof, selected_sources, self.device):
            pred = self.model(data.x, data.edge_index, self.gd.fixed_tof_mask)
            boundary ,tof_pred, sos_pred = self._extract_tof_sos(pred)
            sos_list.append(sos_pred)

            T_true = self._get_region_of_interest(all_tof_maps[i].T)
            # ompare between tof[i,:] (values on receivers and predicted value on receivers
            bc_loss_i = self.criterion(boundary, tof[i,:])
            loss_bc_total += bc_loss_i

            # consider TOF values that receive messages
            loss_tof_i = self.criterion(tof_pred, T_true)
            loss_tof_total += loss_tof_i

            pde_loss_total += self.eikonal_loss(tof_pred, sos_pred)
            #print(sos_pred)
            print(f"tof_pred min{torch.min(tof_pred)} max{torch.max(tof_pred)} mean:{torch.mean(tof_pred)}")
            #print(f"-- sos_pred min{torch.min(sos_pred)} max{torch.max(sos_pred)} mean:{torch.mean(sos_pred)}")
            if torch.mean(tof_pred) == 0:
                print(f"ERROR!!!!- tof_pred min{torch.min(tof_pred)} max{torch.max(tof_pred)} mean:{torch.mean(tof_pred)}")
                exit()
            print(f"- boundary min{torch.min(boundary)} max{torch.max(boundary)} mean:{torch.mean(boundary)}")
            #print(f"+ tof_true min{torch.min(T_true)} max{torch.max(T_true)} mean:{torch.mean(T_true)}")

        c_stack = torch.stack(sos_list, dim=0)  # shape = (num_sources, nx, ny)
        sos_pred = c_stack.mean(dim=0)
        loss_sos = self.criterion(sos_pred, c_true)

        #loss_tof_total /= num_sources
        #loss_bc_total /= num_sources
        #pde_loss_total /= num_sources  # average across sources

        tw, cw, pw, bw = _balance_weights(loss_tof_total, loss_sos, pde_loss_total, loss_bc_total)

        data_loss = tw*loss_tof_total +  cw * loss_sos
        total_loss = data_loss + pw * pde_loss_total + bw * loss_bc_total

        log_message(f"---total_loss:{total_loss}  loss_c: {loss_sos} loss_tof:{loss_tof_total} pde_loss:{pde_loss_total} bc_loss:{loss_bc_total} bw:{bw} cw:{cw} tw:{tw} pw:{pw}")


        return total_loss, data_loss, pde_loss_total, pde_loss_total

    def eval_model(self, batch):
        sources_positions = batch['x_s'].squeeze()
        receivers_positions = batch['x_r'].squeeze()
        tof = batch['raw_tof'].squeeze().float().to(self.device)
        if not self.gd.initialized:
            self.gd.build(sources_positions, receivers_positions)

        num_sources = 10
        sos_list = []
        selected_sources = random.sample(range(32), k=num_sources)
        for i, data in self.gd.get_graph(tof, selected_sources, self.device):
            pred = self.model(data.x, data.edge_index, self.gd.fixed_tof_mask)
            sos_i = self._extract_sos(pred)
            sos_list.append(sos_i)

        c_stack = torch.stack(sos_list, dim=0)  # shape = (num_sources, nx, ny)
        c_avg = c_stack.mean(dim=0)
        c_pred = np.expand_dims(np.expand_dims(c_avg.cpu(), axis=0), axis=0)
        c_pred = (c_pred - app_settings.min_sos) / (app_settings.max_sos - app_settings.min_sos)
        return c_pred


    def eval_tof(self, batch):
        tofs_true, tofs_pred = [],[]
        sources_positions = batch['x_s'].squeeze()
        receivers_positions = batch['x_r'].squeeze()
        all_tof_maps = batch['tof_maps'].squeeze().float()
        tof = batch['raw_tof'].squeeze().float().to(self.device)
        if not self.gd.initialized:
            self.gd.build(sources_positions, receivers_positions)
        num_sources = 10
        selected_sources = range(32)
        for i, data in self.gd.get_graph(tof, selected_sources, self.device):
            pred = self.model(data.x, data.edge_index, self.gd.fixed_tof_mask)

            boundary, tof_pred, sos_pred = self._extract_tof_sos(pred)
            print('boundary')
            print(boundary.tolist())
            print('tof i')
            print(tof[i,:].tolist())

            tof_pred = self._extract_tof(pred)
            tof_pred = tof_pred.cpu()
            tofs_pred.append(tof_pred)

            tof_true = all_tof_maps[i].T
            tof_pred = tof_true.cpu()
            tofs_true.append(tof_pred)

        return  tofs_true, tofs_pred



    def get_model_input_data(self, batch):
        sources_positions = batch['x_s'].squeeze()
        receivers_positions = batch['x_r'].squeeze()

    def eikonal_loss(self, T, c):
        eps = 1e-8

        dy, dx = torch.gradient(
            T,
            spacing=(1.0 , 1.0)
        )
        grad_mag = torch.sqrt(dx ** 2 + dy ** 2 + eps)
        residual = grad_mag - 1.0 / (c + eps)
        return torch.mean(residual ** 2)

    def _extract_sos(self, pred):
        # T = pred[:, 0]
        # c = pred[:, 1]
        # Indices 64: onward  (after 32 sources and 32 receivers are the mesh nodes)
        mesh_c_2D = pred[:, 1][64:].reshape(self.nx, self.ny) # all interior mesh node
        return mesh_c_2D

    def _extract_tof(self, pred):
        # T = pred[:, 0]
        # c = pred[:, 1]
        # Indices 64: onward  (after 32 sources and 32 receivers are the mesh nodes)
        mesh_T_2D = pred[:, 0][64:].reshape(self.nx, self.ny) # all interior mesh node
        return mesh_T_2D


    def _extract_tof_sos(self, pred):
        # T = pred[:, 0]
        # c = pred[:, 1]
        # Indices 64: onward  (after 32 sources and 32 receivers are the mesh nodes)
        boundary = pred[:, 0][32:64]  # receiver nodes
        mesh_T_2D = pred[:, 0][64:].reshape(self.nx, self.ny)  # all interior mesh node
        mesh_c_2D = pred[:, 1][64:].reshape(self.nx, self.ny) # all interior mesh node
        return boundary, mesh_T_2D, mesh_c_2D

    def _get_region_of_interest(self ,d):
        x1, x2 = self.x_range  # (32, 96)
        y1, y2 = self.y_range  # (32, 96)
        return d[x1:x2, y1:y2]

    def get_model_input_data(self, batch):
        pass


class TOFtoSOSPINNLinerTrainingStep(BaseTrainingStep):
    def __init__(self):
        super().__init__()
        self.X_tensor = None
        self.Y_tensor = None
        self.epsilon = 1e-10

    def init(self, model, device):
        super().init( model, device)
        x_vals = np.linspace(1, app_settings.anatomy_width, app_settings.anatomy_width)
        y_vals = np.linspace(1, app_settings.anatomy_height, app_settings.anatomy_height)
        X, Y = np.meshgrid(x_vals, y_vals)
        self.X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True).reshape(-1, 1).to(self.device)
        self.Y_tensor = torch.tensor(Y, dtype=torch.float32, requires_grad=True).reshape(-1, 1).to(self.device)

    def perform_step(self, batch):
        c_true = batch['sos'].squeeze().float().to(self.device)
        selected_sources, x, y, tof = self.get_model_input_data(batch)
        c_pred = self.model(x, y, tof)[:, 0]

        if self.model.training:
            return self.pinn_loss(tof, c_pred, c_true,  domain_xy=[x, y])
        else:

            h, w = c_true.shape
            c_pred = c_pred.reshape(h, w)
            return self.data_loss( c_pred, c_true)

    def eval_model(self, batch):
        selected_sources, x, y, tof_val = self.training_step_handler.get_model_input_data(batch)
        c_pred = self.model(x, y, tof_val)
        _, _, h, w = anatomy.shape
        c_pred = np.expand_dims(np.expand_dims(c_pred.reshape(h, w).cpu(), axis=0), axis=0)
        c_pred = (c_pred - app_settings.min_sos) / (app_settings.max_sos - app_settings.min_sos)
        return c_pred

    def get_model_input_data(self, batch):
        all_tof_maps = batch['tof_maps'].squeeze()
        num_sources = self.model.num_sources
        selected_sources = random.choices(range(num_sources), k=num_sources)
        tof_maps = all_tof_maps[selected_sources]

        ToF_tensors = [torch.tensor(tof, dtype=torch.float32).reshape(-1, 1) for tof in tof_maps]
        ToF_combined = torch.cat(ToF_tensors, dim=1).to(self.device)  # Combine ToF maps as input features

        sample_perc = 1
        num_samples = int(sample_perc * self.X_tensor.shape[0])  # Use only part of the tof messured until I can ran it with source / dest locations only
        sample_indices = np.random.choice(self.X_tensor.shape[0], num_samples, replace=False)
        sample_indices = sorted(sample_indices)

        X_sampled = self.X_tensor[sample_indices]
        Y_sampled = self.Y_tensor[sample_indices]
        ToF_sampled = ToF_combined[sample_indices]
        return selected_sources, X_sampled, Y_sampled, ToF_sampled

    def pinn_loss(self, tof_map, c_pred,c_true, domain_xy):
        # Compute predicted travel time from learned SoS
        T_pred_sources = []
        for s in range(num_sources):
            T_pred_s = fast_marching_2d(c_pred, sources[s])  # shape [num_samples] or [res*res]
            T_pred_grad_x = torch.autograd.grad(T_pred_s.sum(), domain_xy[0], create_graph=True)[0]
            T_pred_grad_y = torch.autograd.grad(T_pred_s.sum(), domain_xy[1], create_graph=True)[0]

            # Compute gradient magnitude
            grad_T_pred = torch.sqrt(torch.clamp(T_pred_grad_x ** 2 + T_pred_grad_y ** 2, min=self.epsilon))

            # Compute eikonal loss (ensure predicted wavefront follows physics)
            loss_eikonal = torch.mean((grad_T_pred - 1 / (c_pred + self.epsilon)) ** 2)

            loss_tof = torch.mean((T_pred - tof_map) ** 2) / (torch.var(tof_map) + self.epsilon)


        # Smoothness regularization
        loss_smooth = torch.mean(torch.abs(torch.gradient(c_pred)[0]))

        # MSE loss
        h, w = c_true.shape
        c_pred = c_pred.reshape(h, w)
        mse_loss = self.criterion(c_pred, c_true)

        # balance losses
        w_eik = min(_balance_order_of_magnitude(mse_loss,loss_eikonal), 0.000001)
        w_tof =  max(_balance_order_of_magnitude(mse_loss, loss_tof),0.1)
        w_smooth = max(_balance_order_of_magnitude(w_eik*loss_eikonal, loss_smooth) / 10, 0.0001)

        #w_eik = 1
        #w_tof = 1
        #w_smooth = 1
        #w_mse = 1
        loss_total = w_mse*mse_loss + w_eik*loss_eikonal + w_tof * loss_tof + w_smooth * loss_smooth


        log_message(
            f"total_loss:{loss_total} mse_loss:{mse_loss} loss_eikonal:{loss_eikonal} (w{w_eik})   loss_tof: {loss_tof} (w{w_tof})  loss_smooth:{loss_smooth} (w{w_smooth})")
        return loss_total, loss_tof, loss_eikonal, loss_smooth

    def data_loss(self, sos_pred, sos_true):
        n_mse_loss = self.criterion(sos_pred, sos_true)
        w_mse_loss = self.w_criterion(sos_pred, sos_true)
        w = 0
        if n_mse_loss < 0.1:
            w = _balance_order_of_magnitude(n_mse_loss, w_mse_loss) / 10
        loss_total = n_mse_loss + w * w_mse_loss
        log_message(
            f"total_loss:{loss_total}  n_mse_loss: {n_mse_loss} w_mse_loss:{w_mse_loss} balance:{w }")
        return loss_total, n_mse_loss, w_mse_loss, w


class TofToSosUNetTrainingStep(BaseTrainingStep):
    def __init__(self):
        super().__init__()
        self.solver = Solver()
    def perform_step(self, batch):
        tof_tensor = batch['tof'].float().to(self.device)
        sos_tensor = batch['anatomy'].float().to(self.device)
        #sources = batch['x_s'].to(self.device)
        #receivers = batch['x_r'].to(self.device)

        sos_pred = self.model(tof_tensor)
        n_mse_loss = self.criterion(sos_pred, sos_tensor)
        w_mse_loss = self.w_criterion(sos_pred, sos_tensor)
        w = 0
        if n_mse_loss<0.1:
            w = _balance_order_of_magnitude(n_mse_loss, w_mse_loss)/10
        mse_loss = n_mse_loss + w*w_mse_loss
        #mse_loss = w_mse_loss

        log_message(f"total_loss:{mse_loss} mse_loss: {n_mse_loss} w_mse_loss:{w_mse_loss} balance:{w}")

        pde_loss = 0
        aw = 1
        bw = 1e4


        total_loss = aw*mse_loss +  bw*pde_loss
        #log_message(f"total_loss:{total_loss} mse_loss: {mse_loss} pde_loss:{total_pde} bc_loss:{total_bc}")
        #log_message(f"total_loss: {total_loss:.4e} ... mse_loss: {aw*mse_loss:.4e} ... pde_loss: {bw*pde_loss:.4e}")
        weighted_mse_loss = aw*mse_loss
        weighted_pde_loss = bw*pde_loss
        weighted_bc_loss = 0
        #return total_loss
        return total_loss, weighted_mse_loss, weighted_pde_loss, weighted_bc_loss

    def __DEP__perform_step(self, batch):
        tof_tensor = batch['tof'].to(self.device)
        sos_tensor = batch['anatomy'].to(self.device)
        sources = batch['x_s'].to(self.device)
        receivers = batch['x_r'].to(self.device)

        sos_pred = self.model(tof_tensor)
        mse_loss_all = self.criterion(sos_pred, sos_tensor)

        roi_start = 50
        roi_end = 70
        eps = 1e-8
        pred_sos_clipped = sos_pred[:, :, roi_start:roi_end, roi_start:roi_end].clamp(min=eps)
        sos_true_clipped = sos_tensor[:, :, roi_start:roi_end, roi_start:roi_end].clamp(min=eps)
        mse_loss_roi = self.criterion(pred_sos_clipped, sos_true_clipped)
        mse_loss = mse_loss_all + 2*mse_loss_roi


        pde_loss = 0.0
        k_count = 32
        selected_sources = random.choices(sources[0].squeeze(),k=k_count)
        #print(selected_sources)

        rec_tuples = []
        for rec in receivers[0].squeeze():
            rec_tuples.append((int(rec[0])-1, int(rec[1])-1))

        for src in selected_sources:
            s = (int(src[0])-1, int(src[1])-1)

            arrival_times, tof = self.solver.tof_domain(sos=sos_pred, source=s, receivers=rec_tuples)
            pde_loss += eikonal_loss_multi(sos_pred, arrival_times)
            #print(pde_loss)
        pde_loss/=k_count


        aw = 1
        bw = 1e4


        total_loss = aw*mse_loss +  bw*pde_loss
        #log_message(f"total_loss:{total_loss} mse_loss: {mse_loss} pde_loss:{total_pde} bc_loss:{total_bc}")
        log_message(f"total_loss: {total_loss:.4e} ... mse_loss: {aw*mse_loss:.4e} ... pde_loss: {bw*pde_loss:.4e}")
        weighted_mse_loss = aw*mse_loss
        weighted_pde_loss = bw*pde_loss
        weighted_bc_loss = 0
        #return total_loss
        return total_loss, weighted_mse_loss, weighted_pde_loss, weighted_bc_loss

class TofPredictorTrainingStep(BaseTrainingStep):
    def __init__(self, **kwargs):
        super().__init__()
        self.grid_w = kwargs.get('grid_w',128)
        self.grid_h = kwargs.get('grid_h',128)

    def perform_step(self, batch):
        tof_tensor = batch['tof'].to(self.device)
        sos_tensor = batch['anatomy'].to(self.device)
        x_s = batch['x_s'].float()
        x_r = batch['x_r'].float()

        pred_tof = self.model(tof_tensor)

        total_loss = 0.0
        for idx,  pred in enumerate(pred_tof):
            known_t = tof_tensor[idx].squeeze()
            src_loc = (x_s[idx] * self.grid_h).int()
            rec_loc = (x_r[idx] * self.grid_w).int()

            loss_pde = 1e-3 * eikonal_loss(pred, sos_tensor[idx])
            ic_loss  = 1e2 * initial_loss(pred, src_loc)
            bc_loss  =  1e2* boundary_loss(pred, known_t, src_loc, rec_loc)

            total_loss +=  loss_pde +  ic_loss +  bc_loss

            #log_message(f"pde: {loss_pde} ic_loss: {ic_loss} bc_loss: {bc_loss}")


        #log_message(f"loss: {total_loss} ")
        return total_loss


class CombinedSosTofTrainingStep(BaseTrainingStep):
    def __init__(self, **kwargs):
        super().__init__()

    def perform_step(self, batch):
        tof_tensor = batch['tof'].float().to(self.device)
        sos_tensor = batch['anatomy'].to(self.device)
        positions_mask = batch['positions_mask'].float().to(self.device)
        raw_tof = batch['raw_tof'].float().to(self.device)
        sources_coords = batch['x_s'].to(self.device)
        receivers_coords = batch['x_s'].to(self.device)


        pred_sos, pred_tof= self.model(tof_tensor, positions_mask)

        mse_loss = self.criterion(pred_sos, sos_tensor)

        total_pde = 0
        total_bc = 0
        c=0
        for idx, pred in enumerate(pred_tof):
            c += 1
            pde_loss = eikonal_loss(pred, pred_sos[idx])
            #pde_loss = eikonal_loss(pred, sos_tensor[idx])
            total_pde += pde_loss

            bc_loss = boundary_loss(pred,  raw_tof[idx], positions_mask[idx])
            #bc_loss = boundary_loss(pred,  raw_tof[idx], sources_coords[idx], receivers_coords[idx] )

            total_bc += bc_loss

        aw = 1e-2
        bw = 1
        cw = 1e-2
        total_loss = aw*mse_loss +  bw*total_pde + cw*total_bc
        #log_message(f"total_loss:{total_loss} mse_loss: {mse_loss} pde_loss:{total_pde} bc_loss:{total_bc}")
        log_message(f"total_loss: {total_loss:.4e} ... mse_loss: {aw*mse_loss:.4e} ... pde_loss: {bw*total_pde:.4e} ... bc_loss: {cw*total_bc:.4e}")
        weighted_mse_loss = aw*mse_loss
        weighted_pde_loss = bw*total_pde
        weighted_bc_loss = cw*total_bc
        #return total_loss
        return total_loss, weighted_mse_loss, weighted_pde_loss, weighted_bc_loss



class TofPredictorTrainingStep_V1(BaseTrainingStep):
    def __init__(self, **kwargs):
        super().__init__()
        self.num_collocation = kwargs.get('num_collocation', 60*60)
        self.x_min = kwargs.get('x_min', 0.3)
        self.x_max = kwargs.get('x_max', 0.7)
        self.y_min = kwargs.get('y_min', 0.3)
        self.y_max = kwargs.get('x_max', 0.7)
        #self.domain_coords = self.prepare_domain_coords()



    def perform_step(self, batch):
        x_s = batch['x_s'].float().to(self.device)
        x_r = batch['x_r'].float().to(self.device)
        known_tof = batch['x_o'].float().to(self.device)

        raw_t_obs = batch['raw_t_obs'].float().to(self.device)
        #sos_tensor = batch['anatomy'].to(self.device)


        T_pred = self.model(known_tof, x_s)
        ic_loss = 0.0

        for rec_idx, pred_tof in enumerate(T_pred):
            ic_loss += self.criterion(pred_tof,  torch.zeros_like(pred_tof))

        ic_loss = ic_loss


        T_pred = self.model(known_tof, x_r)
        bc_loss = 0.0

        for rec_idx, pred_tof in enumerate(T_pred):
            print(raw_t_obs[rec_idx].shape)
            bc_loss += self.criterion(pred_tof, raw_t_obs[rec_idx])
        bc_loss = bc_loss


        # xy = self.domain_coords
        # For each source s:
        #grad_T = torch.autograd.grad(T_pred[:, s].sum(), xy, create_graph=True)[0]
        #Tx = grad_T[:, 0]  # partial derivative wrt x
        #Ty = grad_T[:, 1]  # partial derivative wrt y

        #grad_norm = torch.sqrt(Tx ** 2 + Ty ** 2)
        #pde_residual = grad_norm * c(xy) - 1.0  # instead of  grad_norm - 1/c(x,y) to avoid divide by zero


        #loss = (pde_residual ** 2).mean()

        # todo sum combine with data loss

        loss = bc_loss + ic_loss * 1e6

        return loss


    def prepare_domain_coords(self):
        coords = torch.rand(self.num_collocation, 2, device=self.device)
        coords[:, 0] = coords[:, 0] * (self.x_max - x_min) + x_min
        coords[:, 1] = coords[:, 1] * (self.y_max - y_min) + y_min
        coords.requires_grad_(True)
        return coords


def _balance_order_of_magnitude(a, b):
    # Calculate the orders of magnitude of a and b
    eps = 1e-16
    order_a = math.log10(a+eps)
    order_b = math.log10(b+eps)

    # Calculate the difference in orders of magnitude
    difference = order_a - order_b

    # Find the weight to balance the orders of magnitude
    weight = 10 ** math.ceil(difference)

    return weight


def _compute_laplacian_loss(c_pred, X, Y):
    """
    Compute smoothness loss using Laplacian second derivatives.
    """
    T_xx = torch.autograd.grad(c_pred.sum(), X, create_graph=True)[0]
    T_yy = torch.autograd.grad(c_pred.sum(), Y, create_graph=True)[0]
    laplacian_loss = torch.mean(T_xx ** 2 + T_yy ** 2)
    return laplacian_loss


def _balance_weights(loss_tof, loss_sos, loss_pde,loss_bc):

    if loss_bc> 1600 or loss_tof > 1000:
        pow = -1 * math.ceil(math.log10(loss_tof)) + random.randint(-5,0)
        wt = 10 ** pow

        pow = -1 * math.ceil(math.log10(loss_sos)) + random.randint(-5,0)
        ws = 10 ** pow

        pow = -1 * math.ceil(math.log10(loss_pde)) + random.randint(-5, 0)
        wp = 10 ** pow

        #pow = -1 * math.ceil(math.log10(loss_bc)) + random.randint(-5, 0)
        pow = -1 * math.ceil(math.log10(loss_bc))
        wb = 10 ** pow

        #return wt, ws, wp, wb
        return wt, 0, 0, wb


    wt = 1.0 / (loss_tof.detach() + 1e-8)
    ws = 1.0 / (loss_sos.detach() + 1e-8)
    wp = 1.0 / (loss_pde + 1e-8)
    wb = 1.0 / (loss_bc.detach() + 1e-8)




    # Then scale so sum of weights = 1 or some constant
    sum_w = wt + ws + wb + wp
    wt /= sum_w
    ws /= sum_w
    wp /= sum_w
    wb /= sum_w
    #return wt, ws, wp, wb
    return wt, ws, wp, wb
