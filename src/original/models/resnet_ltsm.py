import torch
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
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True).float()
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
         known_tofs: shape [B, N, 5]
             (the known source-receiver-tof data for each sample in the batch)
         xy: shape [B, M, 2]
             (the query points for which we want to predict 32 time-of-flights)
         Returns:
             T_pred: shape [B, M, 32]
         """
         # Run LSTM on known measurements
         # LSTM outputs:
         #   output: [B, N, hidden_dim] (if batch_first=True)
         #   (h, c): [num_layers, B, hidden_dim]
         _, (h, c) = self.encoder(known_tofs)

         # Take the last layer’s hidden state: shape [B, hidden_dim]
         hidden = h[-1]

         # Map hidden state to 64-dim context
         context = self.fc(hidden)  # [B, 64]

         # Expand context to [B, M, 64] to match xy shape [B, M, 2]
         B, M, _ = xy.shape
         context_expanded = context.unsqueeze(1).expand(-1, M, -1)

         # Concatenate (x,y) with context along the last dimension => [B, M, 66]
         xyc = torch.cat((xy, context_expanded), dim=2)

         # Predict time-of-flights => [B, M, 32]
         T_pred = self.predictor(xyc)
         return T_pred


