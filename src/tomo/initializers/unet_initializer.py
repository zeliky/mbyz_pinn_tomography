"""UNet initializer: SR (tof_grid) or backprojection+U-Net from measurement grid [S,R].

Stage 0: predicts residual delta_c0 above c_base; UNetInitializer builds c0 = c_base + delta_c0_pred.
Backprojection model requires Observation.layout_metadata with x_s, x_r tensors.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, ClassVar

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


def _infer_legacy_model_type(state: dict[str, Any]) -> str:
    if any(k.startswith("backprojection.") for k in state):
        return "stage0_backproj_unet"
    return "sr_initializer_net"


def _num_decoder_upsample_stages(
    tof_input_size: int, output_size: int
) -> int:
    """Decoder upsample stages so bottleneck * 2^k = output. Bottleneck = tof / 8."""
    ratio = (output_size * 8) / tof_input_size
    k = int(round(math.log2(ratio)))
    if k < 1 or 2**k != ratio:
        raise ValueError(
            f"Unsupported tof_output combo: tof={tof_input_size} output={output_size} "
            f"-> ratio={ratio} must be power of 2 (got k={k}, 2^k={2**k})."
        )
    return k


class SRInitializerNet(nn.Module):
    """
    Super-resolution net: raw_tof [B, 1, H_tof, W_tof] -> delta_c0 [B, 1, H_out, W_out].
    Output is normalized non-negative residual above c_base. Encoder-decoder with
    optional TOFGuidedAttention in bottleneck. Decoder depth is computed from
    tof_input and output sizes so output matches anatomy grid.
    """

    model_type: ClassVar[str] = "sr_initializer_net"

    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        base_filters: int = 64,
        use_attention: bool = False,
        use_residual: bool = True,
        tof_input_height: int = 64,
        tof_input_width: int = 64,
        output_height: int = 128,
        output_width: int = 128,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.use_residual = use_residual
        self._tof_input_height = tof_input_height
        self._tof_input_width = tof_input_width
        self._output_height = output_height
        self._output_width = output_width

        k_h = _num_decoder_upsample_stages(tof_input_height, output_height)
        k_w = _num_decoder_upsample_stages(tof_input_width, output_width)
        if k_h != k_w:
            raise ValueError(
                f"tof_input/output must yield same k: got k_h={k_h} k_w={k_w}."
            )
        self._num_upsample_stages = k_h

        # Encoder: 3 stride-2 stages -> bottleneck = tof / 8
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

        # Decoder: k upsample stages. Channel progression: 8b -> 4b -> 2b -> b -> b/2 -> b/4
        ch_pairs = [
            (base_filters * 8, base_filters * 4),
            (base_filters * 4, base_filters * 2),
            (base_filters * 2, base_filters),
            (base_filters, base_filters // 2),
            (base_filters // 2, base_filters // 4),
        ]
        self.dec_convs = nn.ModuleList()
        for i in range(self._num_upsample_stages):
            inc, outc = ch_pairs[i]
            self.dec_convs.append(
                nn.Sequential(
                    nn.ConvTranspose2d(inc, outc, 4, stride=2, padding=1),
                    nn.BatchNorm2d(outc),
                    nn.ReLU(inplace=True),
                )
            )
        last_out = ch_pairs[self._num_upsample_stages - 1][1]
        if last_out != base_filters // 4:
            self.dec_bridge = nn.Sequential(
                nn.Conv2d(last_out, base_filters // 4, 3, padding=1),
                nn.BatchNorm2d(base_filters // 4),
                nn.ReLU(inplace=True),
            )
        else:
            self.dec_bridge = nn.Identity()
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
        """Input [B, 1, H_tof, W_tof], output delta_c0 [B, 1, H_out, W_out] in [0, 1]."""
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

        x = bottleneck
        for dec in self.dec_convs:
            x = dec(x)
        x = self.dec_bridge(x)
        return self.final_conv(x)

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "tof_input_height": self._tof_input_height,
            "tof_input_width": self._tof_input_width,
            "output_height": self._output_height,
            "output_width": self._output_width,
        }


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
        expected_model_type: str | None = None,
        tof_input_height: int = 64,
        tof_input_width: int = 64,
    ) -> None:
        self.num_nodes = num_nodes
        self.c0_fallback = c0_fallback
        self._min_sos = min_sos
        self._max_sos = max_sos
        self._c_base = c_base
        self._device = device
        self._net = net
        self._normalized_c_base: float | None = None
        self._expected_model_type = expected_model_type
        self._checkpoint_model_type: str | None = None
        self._tof_input_height = tof_input_height
        self._tof_input_width = tof_input_width
        if net is not None:
            self._checkpoint_model_type = str(
                getattr(net, "model_type", self._checkpoint_model_type or "sr_initializer_net")
            )

    def _get_normalized_c_base(self) -> float:
        if self._normalized_c_base is None:
            self._normalized_c_base = normalized_c_base_from_config(
                self._min_sos, self._max_sos, self._c_base
            )
        return self._normalized_c_base

    def load_checkpoint(self, path: str, device: torch.device | None = None) -> None:
        """Load weights; ``model_type`` in checkpoint must match ``expected_model_type`` when set."""
        ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        if not isinstance(state, dict):
            raise TypeError("Checkpoint model_state_dict must be a dict")

        model_type_in_ckpt = ckpt.get("model_type")
        if model_type_in_ckpt is None:
            model_type_in_ckpt = _infer_legacy_model_type(state)
            warnings.warn(
                f"Checkpoint {path} missing 'model_type'; inferred {model_type_in_ckpt!r} "
                "(prefer saving explicit model_type).",
                stacklevel=2,
            )
        model_type_in_ckpt = str(model_type_in_ckpt)

        if (
            self._expected_model_type is not None
            and model_type_in_ckpt != self._expected_model_type
        ):
            raise ValueError(
                f"Checkpoint model_type={model_type_in_ckpt!r} != expected "
                f"{self._expected_model_type!r}"
            )

        if self._net is None:
            if model_type_in_ckpt == "stage0_backproj_unet":
                raise ValueError(
                    "Checkpoint is stage0_backproj_unet but net was not provided; "
                    "instantiate Stage0BackprojUNet (e.g. via Hydra) and pass net=..."
                )
            self._net = SRInitializerNet()

        self._net.load_state_dict(state, strict=True)
        self._checkpoint_model_type = model_type_in_ckpt
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

        raw_mt = (
            self._checkpoint_model_type
            or self._expected_model_type
            or getattr(self._net, "model_type", "sr_initializer_net")
        )
        mt = raw_mt if isinstance(raw_mt, str) else "sr_initializer_net"

        tof = observation.tof_observed
        if mt == "stage0_backproj_unet":
            if tof.dim() == 2:
                raw_tof = tof.unsqueeze(0).unsqueeze(0)
            elif tof.dim() == 3:
                raw_tof = tof.unsqueeze(0)
            else:
                raw_tof = tof
            meta = observation.layout_metadata or {}
            x_s = meta.get("x_s")
            x_r = meta.get("x_r")
            if x_s is None or x_r is None:
                raise ValueError(
                    "stage0_backproj_unet requires Observation.layout_metadata with 'x_s' and 'x_r'"
                )
            x_s = x_s.to(device)
            x_r = x_r.to(device)
            if x_s.dim() == 2:
                x_s = x_s.unsqueeze(0)
            if x_r.dim() == 2:
                x_r = x_r.unsqueeze(0)
            raw_tof = raw_tof.to(device)
            self._net.eval()
            with torch.no_grad():
                delta_pred = self._net(raw_tof, x_s=x_s, x_r=x_r)
        else:
            target_hw = (self._tof_input_height, self._tof_input_width)
            if tof.dim() == 2:
                raw_tof = tof.unsqueeze(0).unsqueeze(0)
            elif tof.dim() == 3:
                raw_tof = tof.unsqueeze(0) if tof.shape[0] != 1 else tof
            else:
                raw_tof = tof
            if raw_tof.shape[-2:] != target_hw:
                raw_tof = F.interpolate(
                    raw_tof.float(),
                    size=target_hw,
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
        # Keep 128x128 map for Stage 0/1; flatten only for c_values (graph-based stages)
        c_map_2d = c0_physical
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
        return SoSState(c_values=c_values.to(device), step_idx=0, c_map_2d=c_map_2d.to(device))
