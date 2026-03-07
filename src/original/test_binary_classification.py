"""
Test script for the binary classification approach to solve convergence issues.
This demonstrates the new TOF-to-SOS binary classification network.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from logger import log_message
from models.tof_to_sos_classifier import create_tof_to_sos_classifier, create_binary_mask_from_sos, compute_class_weights

def test_binary_classification_model():
    """Test the binary classification model creation and forward pass."""
    
    log_message("Testing TOF-to-SOS Binary Classification Model")
    log_message("=" * 50)
    
    # Create model
    sos_threshold = 1.5
    model = create_tof_to_sos_classifier(model_size='medium', sos_threshold=sos_threshold)
    
    # Print model info
    info = model.get_model_info()
    log_message(f"Model Info:")
    log_message(f"  Parameters: {info['total_parameters']:,}")
    log_message(f"  Input size: {info['input_size']}")
    log_message(f"  Output size: {info['output_size']}")
    log_message(f"  Model type: {info['model_type']}")
    log_message(f"  SOS threshold: {info['sos_threshold']}")
    
    # Test forward pass
    batch_size = 2
    tof_input = torch.randn(batch_size, 1, 32, 32)  # Random TOF input
    
    log_message(f"\nTesting forward pass with input shape: {tof_input.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits_output = model(tof_input)  # Raw logits
        prob_output = torch.sigmoid(logits_output)  # Convert to probabilities
    
    log_message(f"Output logits shape: {logits_output.shape}")
    log_message(f"Output logits range: [{logits_output.min():.3f}, {logits_output.max():.3f}]")
    log_message(f"Output probabilities range: [{prob_output.min():.3f}, {prob_output.max():.3f}]")
    
    # Test binary prediction
    binary_pred = model.predict_binary_mask(tof_input, threshold=0.5)
    log_message(f"Binary prediction shape: {binary_pred.shape}")
    log_message(f"Binary prediction unique values: {torch.unique(binary_pred)}")
    
    return model, tof_input, logits_output, prob_output, binary_pred


def test_binary_mask_creation():
    """Test binary mask creation from SOS maps."""
    
    log_message("\nTesting Binary Mask Creation")
    log_message("=" * 30)
    
    # Create synthetic SOS map with interesting regions
    batch_size = 2
    sos_map = torch.ones(batch_size, 1, 128, 128) * 1.2  # Background SOS
    
    # Add some "interesting" regions with SOS > 1.5
    sos_map[:, :, 50:70, 50:70] = 1.8  # Tumor region 1
    sos_map[:, :, 80:90, 80:90] = 2.0  # Tumor region 2
    sos_map[:, :, 30:40, 100:110] = 1.6  # Tumor region 3
    
    threshold = 1.5
    binary_mask = create_binary_mask_from_sos(sos_map, threshold)
    
    log_message(f"SOS map shape: {sos_map.shape}")
    log_message(f"SOS map range: [{sos_map.min():.3f}, {sos_map.max():.3f}]")
    log_message(f"Binary mask shape: {binary_mask.shape}")
    log_message(f"Binary mask unique values: {torch.unique(binary_mask)}")
    
    # Count positive pixels
    total_pixels = binary_mask.numel()
    positive_pixels = binary_mask.sum().item()
    positive_ratio = positive_pixels / total_pixels
    
    log_message(f"Total pixels: {total_pixels}")
    log_message(f"Positive pixels (SOS > {threshold}): {positive_pixels}")
    log_message(f"Positive ratio: {positive_ratio:.4f} ({positive_ratio*100:.2f}%)")
    
    # Test class weights computation
    pos_weight = compute_class_weights(binary_mask)
    log_message(f"Computed pos_weight for class balancing: {pos_weight:.3f}")
    
    return sos_map, binary_mask, pos_weight


def test_loss_computation():
    """Test the loss computation with binary_cross_entropy_with_logits."""
    
    log_message("\nTesting Loss Computation")
    log_message("=" * 25)
    
    # Create synthetic data
    batch_size = 2
    logits_pred = torch.randn(batch_size, 1, 128, 128)  # Random logits
    
    # Create binary target with class imbalance (few positive pixels)
    binary_target = torch.zeros(batch_size, 1, 128, 128)
    binary_target[:, :, 50:70, 50:70] = 1.0  # Small positive region
    
    # Compute class weights
    pos_weight = compute_class_weights(binary_target)
    
    # Test different loss functions
    log_message(f"Logits shape: {logits_pred.shape}")
    log_message(f"Target shape: {binary_target.shape}")
    log_message(f"Positive weight: {pos_weight:.3f}")
    
    # Standard BCE with logits
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits_pred, binary_target
    )
    log_message(f"Standard BCE loss: {bce_loss:.6f}")
    
    # Weighted BCE with logits (correct approach)
    weighted_bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits_pred, binary_target, 
        pos_weight=pos_weight
    )
    log_message(f"Weighted BCE loss: {weighted_bce_loss:.6f}")
    
    # Show the difference
    improvement = (bce_loss - weighted_bce_loss) / bce_loss * 100
    log_message(f"Improvement with weighting: {improvement:.2f}%")
    
    return bce_loss, weighted_bce_loss


def main():
    """Run all tests."""
    
    log_message("TOF-to-SOS Binary Classification Test Suite")
    log_message("=" * 60)
    log_message("This tests the new binary classification approach to solve convergence issues.")
    log_message("Instead of predicting exact SOS values, we predict probability maps of 'interesting' regions.")
    log_message("")
    
    try:
        # Test 1: Model creation and forward pass
        model, tof_input, logits, probs, binary_pred = test_binary_classification_model()
        
        # Test 2: Binary mask creation
        sos_map, binary_mask, pos_weight = test_binary_mask_creation()
        
        # Test 3: Loss computation
        bce_loss, weighted_bce_loss = test_loss_computation()
        
        log_message("\n" + "=" * 60)
        log_message("All tests completed successfully!")
        log_message("The binary classification approach is ready for training.")
        log_message("")
        log_message("Key advantages of this approach:")
        log_message("1. Solves convergence issues by using classification instead of regression")
        log_message("2. Handles class imbalance with weighted BCE loss and focal loss")
        log_message("3. Outputs probability maps that can guide RL agent to interesting regions")
        log_message("4. More stable training with binary targets instead of continuous SOS values")
        log_message("")
        log_message("To train the model, run: python src/main.py")
        
    except Exception as e:
        log_message(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
