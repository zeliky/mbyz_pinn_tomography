"""UNet initializer: learned 32x32 -> 128x128 TOF-to-SoS super-resolution.

Stage 0: SRInitializerNet predicts normalized non-negative residual delta_c0 above c_base.
UNetInitializer loads checkpoint and builds c0 = c_base + delta_c0_pred.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tomo.initializers.base import Initializer
from tomo.state.observation import Observation
from tomo.state.sos_state import SoSState
from tomo.utils.units import to_physical_sos, to_scaled_sos


def normalized_c_base_from_config(min_sos: float, max_sos: float, c_base: float) -> float:
    """Compute normalized baseline in [0, 1] from physical c_base and SOS range.

    Uses same mapping as tomo.utils.units.to_scaled_sos (clamped to [0, 1]).
    """
    return float(to_scaled_sos(c_base, min_sos, max_sos, clamp=True))


def delta_target_from_anatomy(
    anatomy: torch.Tensor, normalized_c_base: float
) -> torch.Tensor:
    """Non-negative residual target: clamp(normalized_sos - normalized_c_base, 0, 1)."""
    return (anatomy - normalized_c_base).clamp(min=0.0, max=1.0)


def c0_normalized_from_delta(
    normalized_c_base: float, delta: torch.Tensor
) -> torch.Tensor:
    """Reconstruct normalized c0 from baseline and predicted delta; output in [0, 1]."""
    return (normalized_c_base + delta).clamp(0.0, 1.0)


class ResidualBlock(nn.Module):
    """Residual block with batch normalization (ported from legacy tof_to_sos_net)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class TOFGuidedAttention(nn.Module):
    """
    TOF-guided attention: uses TOF input to produce attention weights over features.
    Optional ablation for Stage 0 (use_attention=True).
    """

    def __init__(self, feature_channels: int, tof_channels: int = 1) -> None:
        super().__init__()
        self.tof_attention = nn.Sequential(
            nn.Conv2d(tof_channels, feature_channels // 4, 3, padding=1),
            nn.BatchNorm2d(feature_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels // 4, feature_channels // 2, 3, padding=1),
            nn.BatchNorm2d(feature_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels // 2, feature_channels, 3, padding=1),
            nn.Sigmoid(),
        )
        self.feature_refine = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(
        self, features: torch.Tensor, tof_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tof_resized = F.interpolate(
            tof_input, size=features.shape[-2:], mode="bilinear", align_corners=False
        )
        attention_weights = self.tof_attention(tof_resized)
        attended = features * attention_weights
        refined = self.feature_refine(attended)
        return self.gamma * refined + features, attention_weights


class SRInitializerNet(nn.Module):
    """
    Super-resolution net: raw_tof [B, 1, 32, 32] -> delta_c0 [B, 1, 128, 128].
    Output is normalized non-negative residual above c_base; call sites compute
    normalized_c0 = normalized_c_base + delta_c0. Encoder-decoder with optional
    TOFGuidedAttention in bottleneck.
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        base_filters: int = 64,
        use_attention: bool = False,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.use_residual = use_residual

        # Encoder: 32 -> 16 -> 8 -> 4
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(input_channels, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
        )
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
        )
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
        )
        self.enc_conv4 = nn.Sequential(
            nn.Conv2d(base_filters * 4, base_filters * 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 8),
            nn.ReLU(inplace=True),
        )

        if use_attention:
            self.bottleneck_tof_attention = TOFGuidedAttention(
                base_filters * 8, tof_channels=1
            )
        bottleneck_layers: list[nn.Module] = []
        if use_residual:
            for _ in range(3):
                bottleneck_layers.append(
                    ResidualBlock(base_filters * 8, base_filters * 8)
                )
        self.bottleneck = nn.Sequential(*bottleneck_layers)

        # Decoder: 4 -> 8 -> 16 -> 32 -> 64 -> 128
        self.dec_conv1 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 8, base_filters * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
        )
        self.dec_conv2 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
        )
        self.dec_conv3 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 2, base_filters, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
        )
        self.dec_conv4 = nn.Sequential(
            nn.ConvTranspose2d(base_filters, base_filters // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters // 2),
            nn.ReLU(inplace=True),
        )
        self.dec_conv5 = nn.Sequential(
            nn.ConvTranspose2d(base_filters // 2, base_filters // 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters // 4),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_filters // 4, output_channels, 3, padding=1),
            nn.Sigmoid(),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, raw_tof: torch.Tensor) -> torch.Tensor:
        """Input [B, 1, 32, 32], output delta_c0 [B, 1, 128, 128] in [0, 1]."""
        tof_input = raw_tof
        enc1 = self.enc_conv1(raw_tof)
        enc2 = self.enc_conv2(enc1)
        enc3 = self.enc_conv3(enc2)
        enc4 = self.enc_conv4(enc3)

        if self.use_attention:
            attended, _ = self.bottleneck_tof_attention(enc4, tof_input)
            bottleneck = self.bottleneck(attended)
        else:
            bottleneck = self.bottleneck(enc4)

        dec1 = self.dec_conv1(bottleneck)
        dec2 = self.dec_conv2(dec1)
        dec3 = self.dec_conv3(dec2)
        dec4 = self.dec_conv4(dec3)
        dec5 = self.dec_conv5(dec4)
        return self.final_conv(dec5)


class UNetInitializer(Initializer):
    """
    Learned initializer: TOF -> c0 grid via c_base + delta_c0_pred. Outputs SoSState.
    When _net is set, runs SRInitializerNet and builds c0 from normalized_c_base + delta.
    """

    def __init__(
        self,
        num_nodes: int = 64,
        c0_fallback: float = 1.5,
        min_sos: float = 0.1,
        max_sos: float = 2.1,
        c_base: float = 1.5,
        in_channels: int = 32,
        device: torch.device | None = None,
        net: nn.Module | None = None,
    ) -> None:
        self.num_nodes = num_nodes
        self.c0_fallback = c0_fallback
        self._min_sos = min_sos
        self._max_sos = max_sos
        self._c_base = c_base
        self._device = device
        self._net = net
        self._normalized_c_base: float | None = None

    def _get_normalized_c_base(self) -> float:
        if self._normalized_c_base is None:
            self._normalized_c_base = normalized_c_base_from_config(
                self._min_sos, self._max_sos, self._c_base
            )
        return self._normalized_c_base

    def load_checkpoint(self, path: str, device: torch.device | None = None) -> None:
        """Load SRInitializerNet state_dict from checkpoint. Creates net if needed."""
        ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        if self._net is None:
            self._net = SRInitializerNet()
        self._net.load_state_dict(state, strict=True)
        if device is not None:
            self._net = self._net.to(device)
        self._device = device or self._device

    def __call__(self, observation: Observation) -> SoSState:
        device = self._device or observation.tof_observed.device
        if self._net is None:
            c_values = torch.full(
                (self.num_nodes,),
                self.c0_fallback,
                device=device,
                dtype=observation.tof_observed.dtype,
            )
            return SoSState(c_values=c_values, step_idx=0)

        tof = observation.tof_observed
        if tof.dim() == 2:
            raw_tof = tof.unsqueeze(0).unsqueeze(0)
        elif tof.dim() == 3:
            raw_tof = tof.unsqueeze(0) if tof.shape[0] != 1 else tof
        else:
            raw_tof = tof.reshape(1, 1, 32, 32)
        if raw_tof.shape[-2:] != (32, 32):
            raw_tof = F.interpolate(
                raw_tof.float(),
                size=(32, 32),
                mode="bilinear",
                align_corners=False,
            )
        raw_tof = raw_tof.to(device)
        self._net.eval()
        with torch.no_grad():
            delta_pred = self._net(raw_tof)
        norm_base = self._get_normalized_c_base()
        c0_normalized = c0_normalized_from_delta(norm_base, delta_pred)
        c0_physical = to_physical_sos(
            c0_normalized.cpu().numpy(), self._min_sos, self._max_sos
        )
        c0_physical = torch.from_numpy(c0_physical).to(
            device=c0_normalized.device, dtype=c0_normalized.dtype
        )
        c_values = c0_physical.reshape(-1)
        if c_values.shape[0] != self.num_nodes and self.num_nodes > 0:
            if c_values.shape[0] >= self.num_nodes:
                c_values = c_values[: self.num_nodes]
            else:
                c_values = F.pad(
                    c_values,
                    (0, self.num_nodes - c_values.shape[0]),
                    value=self.c0_fallback,
                )
        return SoSState(c_values=c_values.to(device), step_idx=0)
