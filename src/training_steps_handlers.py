import torch.nn as nn


class BaseTrainingStep:
    def __init__(self):
        self.model = None

    def init(self, model, device):
        self.model = model
        self.device = device

    def set_train_mode(self):
        self.model.train()

    def set_eval_mode(self):
        self.model.eval()

    def perform_step(self, batch):
        pass


class SosResnetTrainingStep(BaseTrainingStep):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def perform_step(self, batch):
        tof_tensor = batch['tof'].to(self.device)
        sos_tensor = batch['anatomy'].to(self.device)
        sos_pred = self.model(tof_tensor)
        loss = self.criterion(sos_pred, sos_tensor)
        return loss

