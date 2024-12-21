import torch.nn as nn
import segmentation_models_pytorch as smp


class TofToSosUNetModel(nn.Module):
    def __init__(self):
        super(TofToSosUNetModel, self).__init__()
        self.unet = smp.Unet(
            encoder_name='resnet34',      # Choose an encoder
            encoder_weights=None,         # No pre-training on ImageNet
            in_channels=1,                # TOF matrix has 1 channel
            classes=1,                    # SOS map has 1 channel
        )

    def forward(self, x):
        return self.unet(x)


class TofPredictorModel(nn.Module):
     def __init__(self, input_dim=5, hidden_dim=64, output_dim=32):
        super().__init__()

        # 1) LSTM aggregator that encodes known measurements
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 64)  # final layer to get context

        # 2) MLP that takes (x,y) + context -> 32 time-of-flights
        self.predictor = nn.Sequential(
            nn.Linear(2 + 64, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, output_dim),  # 32
        )

     def forward(self, known_tofs, xy):
        """
        known_tofs: shape [N, 5] or [1, N, 5]
            (the known source-receiver-tof data)
        xy: shape [M, 2]
            (the query points for which we want to predict 32 time-of-flights)
        """
        # 1) Encode the known_tofs
        #    If needed, expand known_tofs to [1, N, 5] for LSTM.
        if len(known_tofs.shape) == 2:
            known_tofs = known_tofs.unsqueeze(0)  # shape [1, N, 5]

        # LSTM forward
        _, (h, c) = self.encoder(known_tofs)  # h.shape: [1, batch=1, hidden_dim]
        hidden = h[-1]                        # shape: [hidden_dim]
        context = self.fc(hidden)             # shape: [64]

        # 2) Predict for each (x,y) in xy
        #    We replicate the single context for each row in xy
        context_batched = context.unsqueeze(0).expand(xy.size(0), -1)  # [M, 64]
        mlp_input = torch.cat([xy, context_batched], dim=1)            # [M, 66]
        T_pred = self.predictor(mlp_input)                             # [M, 32]

        return T_pred


