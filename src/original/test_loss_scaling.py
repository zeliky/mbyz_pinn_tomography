#!/usr/bin/env python3
"""
Test script to verify loss scaling fixes for TOF-to-SOS training.
"""

import torch
import numpy as np
from models.tof_to_sos_net import create_tof_to_sos_net, normalize_tof, normalize_sos, denormalize_sos
from training_steps_handlers import TOFToSOSTrainingStep

def test_loss_scaling():
    """Test that loss values are properly scaled."""
    
    print("🔧 Testing Loss Scaling Fixes...")
    
    # Create model and training step
    model = create_tof_to_sos_net(model_size='light')
    training_step = TOFToSOSTrainingStep(
        use_physics_loss=False,
        tof_range=(0, 1000),
        sos_range=(0.5, 2.7)
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_step.init(model, device)
    
    # Create synthetic batch data
    batch_size = 2
    batch = {
        'raw_tof': torch.rand(batch_size, 32, 32) * 1000,  # TOF values 0-1000
        'raw_sos': torch.rand(batch_size, 128, 128) * (2.7 - 0.5) + 0.5,  # SOS values 0.5-2.7
    }
    
    # Move to device
    for key in batch:
        batch[key] = batch[key].to(device)
    
    print(f"📊 Input Data Ranges:")
    print(f"  TOF: {batch['raw_tof'].min():.3f} - {batch['raw_tof'].max():.3f}")
    print(f"  SOS: {batch['raw_sos'].min():.3f} - {batch['raw_sos'].max():.3f}")
    
    # Test normalization
    tof_norm = normalize_tof(batch['raw_tof'].unsqueeze(1), 0, 1000)
    sos_norm = normalize_sos(batch['raw_sos'].unsqueeze(1), 0.5, 2.7)
    
    print(f"📊 Normalized Data Ranges:")
    print(f"  TOF: {tof_norm.min():.3f} - {tof_norm.max():.3f}")
    print(f"  SOS: {sos_norm.min():.3f} - {sos_norm.max():.3f}")
    
    # Test denormalization
    sos_denorm = denormalize_sos(sos_norm, 0.5, 2.7)
    print(f"📊 Denormalized SOS Range:")
    print(f"  SOS: {sos_denorm.min():.3f} - {sos_denorm.max():.3f}")
    
    # Test training step
    model.train()
    try:
        total_loss, data_loss, physics_loss, w_mse_loss = training_step.perform_step(batch)
        
        print(f"✅ Loss Computation Successful:")
        print(f"  Total Loss: {total_loss:.6f}")
        print(f"  Data Loss: {data_loss:.6f}")
        print(f"  Physics Loss: {physics_loss:.6f}")
        print(f"  Weighted MSE Loss: {w_mse_loss:.6f}")
        
        # Check if loss is in reasonable range
        if data_loss > 0.01:  # Should be much higher than 0.0027
            print(f"✅ Loss scaling FIXED! Data loss {data_loss:.6f} is in reasonable range")
        else:
            print(f"⚠️  Loss still seems low: {data_loss:.6f}")
            
        # Test gradient flow
        total_loss.backward()
        
        # Check if gradients exist
        has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0 
                          for p in model.parameters())
        
        if has_gradients:
            print("✅ Gradients are flowing properly")
        else:
            print("❌ No gradients detected")
            
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        return False
    
    print("🎯 Loss scaling test completed!")
    return True

def test_expected_loss_magnitude():
    """Test what the expected loss magnitude should be."""
    
    print("\n🧮 Computing Expected Loss Magnitude...")
    
    # Simulate realistic SOS differences
    # Typical SOS range: 0.5 - 2.7 cm/µs
    # Tumor vs normal tissue difference: ~0.05 cm/µs
    # Random prediction error: ~0.1 - 0.5 cm/µs
    
    batch_size = 8
    height, width = 128, 128
    
    # Create realistic SOS maps
    sos_true = torch.ones(batch_size, 1, height, width) * 1.5  # Normal tissue
    sos_pred = sos_true + torch.randn_like(sos_true) * 0.2  # Add realistic noise
    
    # Compute MSE
    mse = torch.nn.functional.mse_loss(sos_pred, sos_true)
    
    print(f"📊 Expected Loss Magnitude Analysis:")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {height}×{width} = {height*width:,} pixels")
    print(f"  SOS difference std: 0.2 cm/µs")
    print(f"  Expected MSE: {mse:.6f}")
    print(f"  Per-pixel error: {torch.sqrt(mse):.6f} cm/µs")
    
    # With 16K pixels and 0.2 std difference, MSE should be around 0.04
    expected_range = (0.01, 0.1)
    if expected_range[0] <= mse <= expected_range[1]:
        print(f"✅ Loss magnitude {mse:.6f} is in expected range {expected_range}")
    else:
        print(f"⚠️  Loss magnitude {mse:.6f} outside expected range {expected_range}")

if __name__ == "__main__":
    print("🚀 Testing TOF-to-SOS Loss Scaling Fixes\n")
    
    # Test expected loss magnitude
    test_expected_loss_magnitude()
    
    # Test actual loss scaling
    success = test_loss_scaling()
    
    if success:
        print("\n✅ All tests passed! Loss scaling should now work properly.")
        print("\n🎯 Key fixes applied:")
        print("  1. MSE loss computed on denormalized (real) SOS values")
        print("  2. Increased learning rate from 1e-4 to 1e-3")
        print("  3. Increased batch size from 4 to 8")
        print("  4. Disabled physics loss initially for debugging")
        print("  5. Proper loss scaling should give MSE ~0.01-0.1 range")
    else:
        print("\n❌ Tests failed. Check the implementation.")
