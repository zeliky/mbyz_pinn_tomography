import torch.nn as nn
import torch
from physics import eikonal_loss, initial_loss, boundary_loss
from logger import log_message
class BaseTrainingStep:
    def __init__(self):
        self.model = None
        self.device = None
        self.criterion = None

    def init(self, model, device):
        self.model = model
        self.device = device
        self.criterion = nn.MSELoss()

    def set_train_mode(self):
        self.model.train()

    def set_eval_mode(self):
        self.model.eval()

    def perform_step(self, batch):
        pass


class TofToSosUNetTrainingStep(BaseTrainingStep):
    def __init__(self):
        super().__init__()


    def perform_step(self, batch):
        tof_tensor = batch['tof'].to(self.device)
        sos_tensor = batch['anatomy'].to(self.device)
        sos_pred = self.model(tof_tensor)
        loss = self.criterion(sos_pred, sos_tensor)
        return loss

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
        #tof_tensor = batch['tof'].to(self.device)

        tof_tensor = batch['tof'].float().to(self.device)
        sos_tensor = batch['anatomy'].to(self.device)
        x_s = batch['x_s'].float()
        x_r = batch['x_r'].float()

        pred_sos, pred_tof= self.model(tof_tensor)
        #print(pred_sos.shape)
        #print(pred_tof.shape)

        mse_loss = self.criterion(pred_sos, sos_tensor)


        total_pde = 0
        c= 0
        for idx, pred in enumerate(pred_tof):
            c+=1
            pde_loss =  eikonal_loss(pred, pred_sos)
            total_pde += pde_loss
            #log_message(f"mse_loss: {mse_loss} p_pde_loss: {pde_loss} ")

        total_loss = mse_loss + total_pde/c
        log_message(f"mse_loss: {mse_loss} pde_loss:{total_pde}")

        return total_loss



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



