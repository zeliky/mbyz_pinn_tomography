"""
TOF-to-SOS Super-Resolution Network
Converts 32x32 TOF matrix to 128x128 SOS map with 4x spatial upsampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention2D(nn.Module):
    """Self-attention module for capturing global TOF relationships."""
    
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Generate query, key, value
        proj_query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, H * W)
        proj_value = self.value(x).view(batch_size, -1, H * W)
        
        # Compute attention
        attention = torch.bmm(proj_query, proj_key)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        # Residual connection with learnable weight
        out = self.gamma * out + x
        return out


class ResidualBlock(nn.Module):
    """Residual block with batch normalization."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x):
        residual = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        return F.relu(out + residual)


class TOFToSOSSuperResNet(nn.Module):
    """
    Super-resolution network for TOF-to-SOS conversion.
    
    Input: 32x32x1 TOF matrix (source-receiver travel times)
    Output: 128x128x1 SOS map (speed of sound in spatial domain)
    
    Architecture: Encoder-Decoder with attention and residual connections
    """
    
    def __init__(self, 
                 input_channels=1, 
                 output_channels=1,
                 base_filters=64,
                 use_attention=True,
                 use_residual=True):
        super().__init__()
        
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # Encoder: Progressive downsampling with feature extraction
        # 32x32x1 -> 32x32x64
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(input_channels, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )
        
        # 32x32x64 -> 16x16x128
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True)
        )
        
        # 16x16x128 -> 8x8x256
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True)
        )
        
        # 8x8x256 -> 4x4x512
        self.enc_conv4 = nn.Sequential(
            nn.Conv2d(base_filters * 4, base_filters * 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 8),
            nn.ReLU(inplace=True)
        )
        
        # Bottleneck: Global feature processing
        bottleneck_layers = []
        
        # Self-attention for global TOF relationships
        if self.use_attention:
            bottleneck_layers.append(SelfAttention2D(base_filters * 8))
        
        # Residual blocks for feature refinement
        if self.use_residual:
            for _ in range(3):
                bottleneck_layers.append(ResidualBlock(base_filters * 8, base_filters * 8))
        
        self.bottleneck = nn.Sequential(*bottleneck_layers)
        
        # Decoder: Progressive upsampling to 128x128
        # 4x4x512 -> 8x8x256
        self.dec_conv1 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 8, base_filters * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True)
        )
        
        # 8x8x256 -> 16x16x128
        self.dec_conv2 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True)
        )
        
        # 16x16x128 -> 32x32x64
        self.dec_conv3 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 2, base_filters, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )
        
        # 32x32x64 -> 64x64x32
        self.dec_conv4 = nn.Sequential(
            nn.ConvTranspose2d(base_filters, base_filters // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters // 2),
            nn.ReLU(inplace=True)
        )
        
        # 64x64x32 -> 128x128x16
        self.dec_conv5 = nn.Sequential(
            nn.ConvTranspose2d(base_filters // 2, base_filters // 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters // 4),
            nn.ReLU(inplace=True)
        )
        
        # 128x128x16 -> 128x128x1 (final SOS output)
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_filters // 4, output_channels, 3, padding=1),
            nn.Sigmoid()  # SOS values typically in [0.8, 2.1] range - will be scaled
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input TOF matrix [batch_size, 1, 32, 32]
            
        Returns:
            sos_map: Predicted SOS map [batch_size, 1, 128, 128]
        """
        # Encoder path
        enc1 = self.enc_conv1(x)      # 32x32x64
        enc2 = self.enc_conv2(enc1)   # 16x16x128
        enc3 = self.enc_conv3(enc2)   # 8x8x256
        enc4 = self.enc_conv4(enc3)   # 4x4x512
        
        # Bottleneck processing
        bottleneck = self.bottleneck(enc4)  # 4x4x512
        
        # Decoder path
        dec1 = self.dec_conv1(bottleneck)  # 8x8x256
        dec2 = self.dec_conv2(dec1)        # 16x16x128
        dec3 = self.dec_conv3(dec2)        # 32x32x64
        dec4 = self.dec_conv4(dec3)        # 64x64x32
        dec5 = self.dec_conv5(dec4)        # 128x128x16
        
        # Final SOS prediction
        sos_map = self.final_conv(dec5)    # 128x128x1
        
        return sos_map
    
    def get_model_info(self):
        """Return information about the model architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': '32x32x1',
            'output_size': '128x128x1',
            'upsampling_factor': 4,
            'use_attention': self.use_attention,
            'use_residual': self.use_residual
        }


class TOFToSOSLightNet(TOFToSOSSuperResNet):
    """Lightweight version of TOF-to-SOS network for faster training/inference."""
    
    def __init__(self):
        super().__init__(
            base_filters=32,
            use_attention=False,
            use_residual=True
        )


class TOFToSOSHeavyNet(TOFToSOSSuperResNet):
    """Heavy version of TOF-to-SOS network for maximum accuracy."""
    
    def __init__(self):
        super().__init__(
            base_filters=96,
            use_attention=True,
            use_residual=True
        )


def create_tof_to_sos_net(model_size='medium', **kwargs):
    """
    Factory function to create TOF-to-SOS networks of different sizes.
    
    Args:
        model_size: 'light', 'medium', or 'heavy'
        **kwargs: Additional arguments to override defaults
    
    Returns:
        TOF-to-SOS network instance
    """
    if model_size == 'light':
        return TOFToSOSLightNet()
    elif model_size == 'medium':
        return TOFToSOSSuperResNet(**kwargs)
    elif model_size == 'heavy':
        return TOFToSOSHeavyNet()
    else:
        raise ValueError(f"Unknown model size: {model_size}. Choose from 'light', 'medium', 'heavy'")


# Utility functions for data preprocessing
def normalize_tof(tof_matrix, tof_min=700, tof_max=900):
    """Normalize TOF values to [0, 1] range."""
    return (tof_matrix - tof_min) / (tof_max - tof_min)


def denormalize_tof(tof_normalized, tof_min=700, tof_max=900):
    """Denormalize TOF values back to original range."""
    return tof_normalized * (tof_max - tof_min) + tof_min


def normalize_sos(sos_map, sos_min=0.8, sos_max=2.1):
    """Normalize SOS values to [0, 1] range."""
    return (sos_map - sos_min) / (sos_max - sos_min)


def denormalize_sos(sos_normalized, sos_min=0.8, sos_max=2.1):
    """Denormalize SOS values back to original range."""
    return sos_normalized * (sos_max - sos_min) + sos_min


if __name__ == "__main__":
    # Test the network
    print("Testing TOF-to-SOS Super-Resolution Network...")
    
    # Test different model sizes
    for size in ['light', 'medium', 'heavy']:
        print(f"\n=== Testing {size.upper()} model ===")
        
        model = create_tof_to_sos_net(model_size=size)
        info = model.get_model_info()
        
        print(f"Parameters: {info['total_parameters']:,}")
        print(f"Input size: {info['input_size']}")
        print(f"Output size: {info['output_size']}")
        print(f"Upsampling factor: {info['upsampling_factor']}x")
        
        # Test forward pass
        dummy_tof = torch.randn(2, 1, 32, 32)  # Batch of 2 TOF matrices
        with torch.no_grad():
            sos_pred = model(dummy_tof)
        
        print(f"Input shape: {dummy_tof.shape}")
        print(f"Output shape: {sos_pred.shape}")
        
        # Memory usage estimate
        param_memory = info['total_parameters'] * 4 / (1024**2)  # MB
        print(f"Estimated parameter memory: {param_memory:.1f} MB")
    
    print("\n✅ All models tested successfully!")
