import torch.nn as nn
import torch
from physics import Solver, eikonal_loss_multi, eikonal_loss, initial_loss, boundary_loss,_to_mps
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


class GradientMatchingLoss(nn.Module):
    def __init__(self, reduction='mean'):
        """
        reduction: 'mean' or 'sum'
        """
        super(GradientMatchingLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        """
        pred, target: (N, C, H, W) tensors (e.g., images)
        """
        # Compute gradients along x-axis
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]

        # Compute gradients along y-axis
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        # L1 difference of gradients
        loss_x = (pred_dx - target_dx).abs()
        loss_y = (pred_dy - target_dy).abs()

        # Combine losses
        if self.reduction == 'mean':
            loss = loss_x.mean() + loss_y.mean()
        elif self.reduction == 'sum':
            loss = loss_x.sum() + loss_y.sum()
        else:
            # No reduction: returns the raw 2D loss map
            loss = torch.cat([loss_x, loss_y], dim=-1)

        return loss


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
            (self.min_sos,   0.1400, 1000),      # surrounding env
            (0.1400,         0.1520, 1000),     # fat
            (0.1520,         0.1550, 500),     # benign tumour
            (0.1550,         self.max_sos, 500)  # cancerous tumour
        ]
        self.ranges_weights= []
        for min_val, max_val, weight in ranges_weights:
            self.ranges_weights.append((
                self._normalize_sos(min_val),
                self._normalize_sos(max_val),
                weight
            ))
        self.default_weight = 1

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
        if n_mse_loss<0.5:
            w = _balance_order_of_magnitude(n_mse_loss, w_mse_loss)
        #mse_loss = n_mse_loss + w*w_mse_loss
        mse_loss = w_mse_loss

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


