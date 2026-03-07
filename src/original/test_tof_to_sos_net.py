"""
Simple test script for TOF-to-SOS super-resolution network.
"""

import torch
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from models.tof_to_sos_net import create_tof_to_sos_net, normalize_tof, denormalize_sos
    print("✅ Successfully imported TOF-to-SOS network modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_network():
    """Test the TOF-to-SOS network with dummy data."""
    print("\n🧪 Testing TOF-to-SOS Super-Resolution Network...")
    
    # Test different model sizes
    for size in ['light', 'medium']:
        print(f"\n=== Testing {size.upper()} model ===")
        
        try:
            # Create model
            model = create_tof_to_sos_net(model_size=size)
            info = model.get_model_info()
            
            print(f"Parameters: {info['total_parameters']:,}")
            print(f"Input size: {info['input_size']}")
            print(f"Output size: {info['output_size']}")
            print(f"Upsampling factor: {info['upsampling_factor']}x")
            
            # Test forward pass
            batch_size = 2
            dummy_tof = torch.randn(batch_size, 1, 32, 32)  # Batch of TOF matrices
            
            # Normalize input (simulate real TOF values 700-900)
            dummy_tof = dummy_tof * 100 + 800  # Scale to 700-900 range
            tof_normalized = normalize_tof(dummy_tof, 700, 900)
            
            print(f"Input shape: {tof_normalized.shape}")
            print(f"Input range: [{tof_normalized.min():.3f}, {tof_normalized.max():.3f}]")
            
            # Forward pass
            model.eval()
            with torch.no_grad():
                sos_pred_normalized = model(tof_normalized)
            
            print(f"Output shape: {sos_pred_normalized.shape}")
            print(f"Output range: [{sos_pred_normalized.min():.3f}, {sos_pred_normalized.max():.3f}]")
            
            # Denormalize output
            sos_pred = denormalize_sos(sos_pred_normalized, 0.8, 2.1)
            print(f"Denormalized SOS range: [{sos_pred.min():.3f}, {sos_pred.max():.3f}]")
            
            # Memory usage estimate
            param_memory = info['total_parameters'] * 4 / (1024**2)  # MB
            print(f"Estimated parameter memory: {param_memory:.1f} MB")
            
            print(f"✅ {size.upper()} model test passed!")
            
        except Exception as e:
            print(f"❌ {size.upper()} model test failed: {e}")
            return False
    
    return True

def test_training_step():
    """Test the training step handler."""
    print("\n🧪 Testing TOF-to-SOS Training Step Handler...")
    
    try:
        from training_steps_handlers import TOFToSOSTrainingStep
        
        # Create training step handler
        training_step = TOFToSOSTrainingStep(
            use_physics_loss=False,
            tof_range=(700, 900),
            sos_range=(0.8, 2.1)
        )
        
        print("✅ Training step handler created successfully")
        
        # Test data preparation
        batch_size = 2
        dummy_batch = {
            'raw_tof': torch.randn(batch_size, 32, 32) * 100 + 800,  # 700-900 range
            'raw_sos': torch.randn(batch_size, 128, 128) * 0.65 + 1.45  # 0.8-2.1 range
        }
        
        # Test input data extraction
        device = torch.device('cpu')
        training_step.init(create_tof_to_sos_net('light'), device)
        
        tof_input = training_step.get_model_input_data(dummy_batch)
        print(f"Training input shape: {tof_input.shape}")
        print(f"Training input range: [{tof_input.min():.3f}, {tof_input.max():.3f}]")
        
        print("✅ Training step handler test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Training step handler test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting TOF-to-SOS Network Tests...")
    
    # Test network architecture
    network_ok = test_network()
    
    # Test training components
    training_ok = test_training_step()
    
    # Summary
    print("\n" + "="*50)
    if network_ok and training_ok:
        print("🎉 All tests passed! TOF-to-SOS implementation is ready.")
        print("\nNext steps:")
        print("1. Ensure your dataset has 'raw_tof' and 'raw_sos' fields")
        print("2. Adjust TOF/SOS ranges in train_tof_to_sos_super_res() if needed")
        print("3. Run: train_tof_to_sos_super_res() in main.py")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    print("="*50)
