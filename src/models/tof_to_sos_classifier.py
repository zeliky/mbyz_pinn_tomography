"""
TOF-to-SOS Binary Classification Network
Converts 32x32 TOF matrix to 128x128 binary mask indicating "interesting" SOS regions (>1.5).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .tof_to_sos_net import TOFToSOSSuperResNet, SelfAttention2D, ResidualBlock


class TOFToSOSClassifier(TOFToSOSSuperResNet):
    """
    Binary classification network for detecting interesting SOS regions.
    
    Input: 32x32x1 TOF matrix (source-receiver travel times)
    Output: 128x128x1 binary probability map (probability of SOS > 1.5)
    
    Architecture: Same as TOFToSOSSuperResNet but with classification output
    """
    
    def __init__(self, 
                 input_channels=1, 
                 output_channels=1,
                 base_filters=64,
                 use_attention=True,
                 use_residual=True,
                 sos_threshold=1.5):
        # Initialize parent class but we'll override the final layer
        super().__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            base_filters=base_filters,
            use_attention=use_attention,
            use_residual=use_residual
        )
        
        self.sos_threshold = sos_threshold
        
        # Replace the final layer for binary classification
        # 128x128x16 -> 128x128x1 (binary probability output)
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_filters // 4, base_filters // 8, 3, padding=1),
            nn.BatchNorm2d(base_filters // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters // 8, output_channels, 3, padding=1)
        )
        
        # Re-initialize the new layers
        self._initialize_final_layer()
    
    def _initialize_final_layer(self):
        """Initialize the new final classification layers."""
        for m in self.final_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the classification network.
        
        Args:
            x: Input TOF matrix [batch_size, 1, 32, 32]
            
        Returns:
            prob_map: Binary probability map [batch_size, 1, 128, 128]
                     Values in [0, 1] indicating probability of SOS > threshold
        """
        # Encoder path (same as parent)
        enc1 = self.enc_conv1(x)      # 32x32x64
        enc2 = self.enc_conv2(enc1)   # 16x16x128
        enc3 = self.enc_conv3(enc2)   # 8x8x256
        enc4 = self.enc_conv4(enc3)   # 4x4x512
        
        # Bottleneck processing (same as parent)
        bottleneck = self.bottleneck(enc4)  # 4x4x512
        
        # Decoder path (same as parent)
        dec1 = self.dec_conv1(bottleneck)  # 8x8x256
        dec2 = self.dec_conv2(dec1)        # 16x16x128
        dec3 = self.dec_conv3(dec2)        # 32x32x64
        dec4 = self.dec_conv4(dec3)        # 64x64x32
        dec5 = self.dec_conv5(dec4)        # 128x128x16
        
        # Final binary classification prediction
        prob_map = self.final_conv(dec5)   # 128x128x1
        
        return prob_map
    
    def predict_binary_mask(self, x, threshold=0.5):
        """
        Get binary predictions from probability map.
        
        Args:
            x: Input TOF matrix [batch_size, 1, 32, 32]
            threshold: Probability threshold for binary classification
            
        Returns:
            binary_mask: Binary mask [batch_size, 1, 128, 128] with 0s and 1s
        """
        with torch.no_grad():
            prob_map = self.forward(x)
            binary_mask = (prob_map > threshold).float()
        return binary_mask
    
    def get_model_info(self):
        """Return information about the classification model."""
        info = super().get_model_info()
        info.update({
            'model_type': 'binary_classifier',
            'sos_threshold': self.sos_threshold,
            'output_type': 'probability_map'
        })
        return info


class TOFToSOSLightClassifier(TOFToSOSClassifier):
    """Lightweight version of TOF-to-SOS classifier."""
    
    def __init__(self, sos_threshold=1.5):
        super().__init__(
            base_filters=32,
            use_attention=False,
            use_residual=True,
            sos_threshold=sos_threshold
        )


class TOFToSOSHeavyClassifier(TOFToSOSClassifier):
    """Heavy version of TOF-to-SOS classifier for maximum accuracy."""
    
    def __init__(self, sos_threshold=1.5):
        super().__init__(
            base_filters=96,
            use_attention=True,
            use_residual=True,
            sos_threshold=sos_threshold
        )


def create_tof_to_sos_classifier(model_size='medium', sos_threshold=1.5, **kwargs):
    """
    Factory function to create TOF-to-SOS classifiers of different sizes.
    
    Args:
        model_size: 'light', 'medium', or 'heavy'
        sos_threshold: SOS threshold for binary classification
        **kwargs: Additional arguments to override defaults
    
    Returns:
        TOF-to-SOS classifier instance
    """
    kwargs['sos_threshold'] = sos_threshold
    
    if model_size == 'light':
        return TOFToSOSLightClassifier(sos_threshold=sos_threshold)
    elif model_size == 'medium':
        return TOFToSOSClassifier(**kwargs)
    elif model_size == 'heavy':
        return TOFToSOSHeavyClassifier(sos_threshold=sos_threshold)
    else:
        raise ValueError(f"Unknown model size: {model_size}. Choose from 'light', 'medium', 'heavy'")


# Utility functions for binary classification data preprocessing
def create_binary_mask_from_sos(sos_map, threshold=1.5):
    """
    Convert SOS map to binary mask.
    
    Args:
        sos_map: SOS values [batch_size, 1, 128, 128]
        threshold: SOS threshold for binary classification
        
    Returns:
        binary_mask: Binary mask [batch_size, 1, 128, 128] with 0s and 1s
    """
    return (sos_map > threshold).float()


def compute_class_weights(binary_masks):
    """
    Compute class weights for handling imbalanced data.
    
    Args:
        binary_masks: Binary masks [batch_size, 1, 128, 128]
        
    Returns:
        pos_weight: Weight for positive class (interesting regions)
    """
    total_pixels = binary_masks.numel()
    positive_pixels = binary_masks.sum().item()
    negative_pixels = total_pixels - positive_pixels
    
    if positive_pixels == 0:
        return torch.tensor(1.0)
    
    # Weight positive class more heavily since it's rare
    pos_weight = negative_pixels / positive_pixels
    return torch.tensor(pos_weight)


