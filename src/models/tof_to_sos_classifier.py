"""
TOF-to-SOS Binary Classification Network
Converts 32x32 TOF matrix to 128x128 binary mask indicating "interesting" SOS regions (>1.5).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .tof_to_sos_net import TOFToSOSSuperResNet, SelfAttention2D, ResidualBlock


class PhysicsAttention(nn.Module):
    """
    Physics-informed spatial attention module that highlights regions where physics residuals are high.
    Focuses capacity on curved-ray artifacts by combining physics residual estimation and curvature detection.
    """
    
    def __init__(self, feature_channels, tof_channels=1):
        super().__init__()
        self.feature_channels = feature_channels
        self.tof_channels = tof_channels
        
        # Lightweight physics residual estimator
        self.physics_estimator = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels//4, 3, padding=1),
            nn.BatchNorm2d(feature_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels//4, 1, 3, padding=1)
        )
        
        # Lightweight curvature detector using gradient analysis
        self.curvature_detector = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),  # TOF + gradients
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 3, padding=1),
            nn.Sigmoid()  # Curvature probability
        )
        
        # Sobel filters for gradient computation (fixed weights)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.unsqueeze(0).unsqueeze(0))
        self.register_buffer('sobel_y', sobel_y.unsqueeze(0).unsqueeze(0))
        
        # Attention fusion network
        self.attention_fusion = nn.Sequential(
            nn.Conv2d(feature_channels + 2, feature_channels//2, 3, padding=1),
            nn.BatchNorm2d(feature_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels//2, feature_channels, 3, padding=1),
            nn.Sigmoid()  # Final attention weights
        )
        
        # Learnable combination weights
        self.physics_weight = nn.Parameter(torch.tensor(0.5))
        self.curvature_weight = nn.Parameter(torch.tensor(0.5))
        
        # Learnable mixing parameter for residual connection
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, features, tof_input):
        """
        Args:
            features: [B, C, H, W] - bottleneck feature maps (4x4x512)
            tof_input: [B, 1, 32, 32] - original TOF input
            
        Returns:
            attended_features: [B, C, H, W] - physics-guided attended features
            auxiliary_outputs: dict with attention maps and physics residuals
        """
        # Estimate physics residuals from current features
        physics_residual = self.physics_estimator(features)  # [B, 1, 4, 4]
        
        # Detect curved-ray artifacts in TOF input
        # Compute TOF gradients using Sobel filters
        grad_x = F.conv2d(tof_input, self.sobel_x, padding=1)
        grad_y = F.conv2d(tof_input, self.sobel_y, padding=1)
        
        # Combine TOF with gradients for curvature detection
        tof_with_grads = torch.cat([tof_input, grad_x, grad_y], dim=1)  # [B, 3, 32, 32]
        curvature_map = self.curvature_detector(tof_with_grads)  # [B, 1, 32, 32]
        
        # Resize to match feature resolution
        physics_residual_resized = F.interpolate(physics_residual, size=features.shape[-2:], mode='bilinear', align_corners=False)
        curvature_map_resized = F.interpolate(curvature_map, size=features.shape[-2:], mode='bilinear', align_corners=False)
        
        # Combine physics insights with learnable weights
        physics_attention = (self.physics_weight * physics_residual_resized + 
                           self.curvature_weight * curvature_map_resized)
        
        # Fuse with features to create final spatial attention
        attention_input = torch.cat([features, physics_residual_resized, curvature_map_resized], dim=1)
        spatial_attention = self.attention_fusion(attention_input)
        
        # Apply attention to features with residual connection
        attended_features = features * spatial_attention
        output = self.gamma * attended_features + features
        
        # Prepare auxiliary outputs for monitoring/loss computation
        auxiliary_outputs = {
            'spatial_attention': spatial_attention,
            'physics_residual': physics_residual,
            'curvature_map': curvature_map,
            'physics_weight': self.physics_weight,
            'curvature_weight': self.curvature_weight
        }
        
        return output, auxiliary_outputs


class TOFToSOSClassifier(TOFToSOSSuperResNet):
    """
    Binary classification network for detecting interesting SOS regions.
    
    Input: 32x32x1 TOF matrix (source-receiver travel times)
    Output: 128x128x1 binary probability map (probability of SOS > 1.5)
    
    Architecture: Same as TOFToSOSSuperResNet but with classification output and PhysicsAttention
    """
    
    def __init__(self, 
                 input_channels=1, 
                 output_channels=1,
                 base_filters=32,  # Changed default from 64 to 32
                 use_attention=True,
                 use_residual=True,
                 sos_threshold=1.5):
        # Initialize parent class but we'll override attention and final layer
        super().__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            base_filters=base_filters,
            use_attention=False,  # We'll replace with PhysicsAttention
            use_residual=use_residual
        )
        
        self.sos_threshold = sos_threshold
        self.use_attention = use_attention
        
        # Replace TOFGuidedAttention with PhysicsAttention
        if self.use_attention:
            self.bottleneck_physics_attention = PhysicsAttention(base_filters * 8, tof_channels=1)
        
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
        
        # Store auxiliary outputs for loss computation
        self.auxiliary_outputs = {}
    
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
        Forward pass through the classification network with PhysicsAttention.
        
        Args:
            x: Input TOF matrix [batch_size, 1, 32, 32]
            
        Returns:
            prob_map: Binary probability map [batch_size, 1, 128, 128]
                     Values in [0, 1] indicating probability of SOS > threshold
        """
        # Store original TOF input for attention
        tof_input = x
        
        # Encoder path
        enc1 = self.enc_conv1(x)      # 32x32x32 (base_filters=32)
        enc2 = self.enc_conv2(enc1)   # 16x16x64
        enc3 = self.enc_conv3(enc2)   # 8x8x128
        enc4 = self.enc_conv4(enc3)   # 4x4x256
        
        # Bottleneck processing with PhysicsAttention
        if self.use_attention:
            # Apply physics-informed attention
            attended_bottleneck, aux_outputs = self.bottleneck_physics_attention(enc4, tof_input)
            # Store auxiliary outputs for loss computation
            self.auxiliary_outputs = aux_outputs
            # Apply residual blocks
            bottleneck = self.bottleneck(attended_bottleneck)  # 4x4x256
        else:
            # Standard bottleneck processing
            bottleneck = self.bottleneck(enc4)  # 4x4x256
            self.auxiliary_outputs = {}
        
        # Decoder path
        dec1 = self.dec_conv1(bottleneck)  # 8x8x128
        dec2 = self.dec_conv2(dec1)        # 16x16x64
        dec3 = self.dec_conv3(dec2)        # 32x32x32
        dec4 = self.dec_conv4(dec3)        # 64x64x16
        dec5 = self.dec_conv5(dec4)        # 128x128x8
        
        # Final binary classification prediction
        prob_map = self.final_conv(dec5)   # 128x128x1
        
        return prob_map
    
    def get_auxiliary_outputs(self):
        """
        Get auxiliary outputs from the PhysicsAttention layer for loss computation.
        
        Returns:
            auxiliary_outputs: dict with attention maps and physics residuals
        """
        return self.auxiliary_outputs
    
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
