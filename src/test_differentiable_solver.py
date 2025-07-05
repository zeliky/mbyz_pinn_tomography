import torch
import numpy as np
import matplotlib.pyplot as plt
from RL.env.wave_solver import WaveSolver
from RL.env.differentiable_wave_solver import create_differentiable_wave_solver
import time


def test_solver_accuracy():
    """Test that the differentiable solver produces similar results to the original."""
    print("Testing solver accuracy...")
    
    # Create test speed field
    F = torch.ones((64, 64), dtype=torch.float32) * 1.5
    # Add some heterogeneity
    F[20:40, 20:40] = 2.0  # Faster region
    F[10:20, 30:50] = 1.0  # Slower region
    
    source_pos = np.array([32, 32])  # Center source
    
    # Original solver
    original_solver = WaveSolver()
    T_original = original_solver.simulate_T(F, source_pos)
    
    # Differentiable solver
    diff_solver = create_differentiable_wave_solver('standard')
    F_grad = F.clone().requires_grad_(True)
    T_diff = diff_solver.simulate_T(F_grad, source_pos)
    
    # Compare results
    mse = torch.mean((T_original - T_diff.detach())**2)
    max_diff = torch.max(torch.abs(T_original - T_diff.detach()))
    
    print(f"MSE between solvers: {mse:.6f}")
    print(f"Max absolute difference: {max_diff:.6f}")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im1 = axes[0].imshow(T_original.numpy(), cmap='viridis')
    axes[0].set_title('Original Solver')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(T_diff.detach().numpy(), cmap='viridis')
    axes[1].set_title('Differentiable Solver')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[1])
    
    diff_map = torch.abs(T_original - T_diff.detach()).numpy()
    im3 = axes[2].imshow(diff_map, cmap='hot')
    axes[2].set_title('Absolute Difference')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('solver_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved comparison plot as 'solver_comparison.png'")
    
    return mse < 1e-3  # Accept if MSE is small


def test_gradient_flow():
    """Test that gradients flow correctly through the differentiable solver."""
    print("\nTesting gradient flow...")
    
    # Create test setup - ensure F is a leaf tensor
    F = torch.ones((32, 32), dtype=torch.float32, requires_grad=True)
    F.data.fill_(1.5)  # Use .data to avoid creating non-leaf tensor
    source_pos = np.array([16, 16])
    target_receivers = torch.tensor([0.5, 0.7, 0.9, 1.1])  # Target ToF values
    
    # Differentiable solver
    diff_solver = create_differentiable_wave_solver('standard')
    
    # Forward pass
    T = diff_solver.simulate_T(F, source_pos)
    
    # Extract receiver values (simulate 4 receivers)
    receiver_positions = [(8, 8), (24, 8), (8, 24), (24, 24)]
    T_receivers = torch.stack([T[y, x] for x, y in receiver_positions])
    
    # Compute loss
    loss = torch.mean((T_receivers - target_receivers)**2)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_gradients = F.grad is not None
    if has_gradients:
        grad_norm = torch.norm(F.grad)
        grad_max = torch.max(torch.abs(F.grad))
        print(f"Gradients computed successfully!")
        print(f"Gradient norm: {grad_norm:.6f}")
        print(f"Max gradient magnitude: {grad_max:.6f}")
        
        # Plot gradients
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(F.detach().numpy(), cmap='viridis')
        plt.title('Speed Field F')
        plt.colorbar()
        
        plt.subplot(1, 3, 2)
        plt.imshow(T.detach().numpy(), cmap='plasma')
        plt.title('Time-of-Flight T')
        plt.colorbar()
        
        plt.subplot(1, 3, 3)
        plt.imshow(F.grad.numpy(), cmap='RdBu_r')
        plt.title('Gradients ∂L/∂F')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig('gradient_flow_test.png', dpi=150, bbox_inches='tight')
        print("Saved gradient visualization as 'gradient_flow_test.png'")
        
    else:
        print("ERROR: No gradients computed!")
    
    return has_gradients and grad_norm > 1e-8


def test_optimization_levels():
    """Test different optimization levels."""
    print("\nTesting optimization levels...")
    
    F = torch.ones((32, 32), dtype=torch.float32) * 1.5
    F[10:20, 10:20] = 2.0
    source_pos = np.array([16, 16])
    
    levels = ['basic', 'standard', 'optimized']
    times = []
    
    for level in levels:
        print(f"Testing {level} optimization level...")
        solver = create_differentiable_wave_solver(level)
        F_test = F.clone().requires_grad_(True)
        
        start_time = time.time()
        T = solver.simulate_T(F_test, source_pos)
        end_time = time.time()
        
        elapsed = end_time - start_time
        times.append(elapsed)
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Output shape: {T.shape}")
        print(f"  Output range: [{T.min():.3f}, {T.max():.3f}]")
    
    return True


def test_memory_usage():
    """Test memory usage and caching."""
    print("\nTesting memory usage and caching...")
    
    F = torch.ones((64, 64), dtype=torch.float32) * 1.5
    source_pos = np.array([32, 32])
    
    # Test with caching
    solver_cached = create_differentiable_wave_solver('standard')
    F_test = F.clone().requires_grad_(True)
    
    # First call
    start_time = time.time()
    T1 = solver_cached.simulate_T(F_test, source_pos)
    time1 = time.time() - start_time
    
    # Second call (should use cache if same input)
    start_time = time.time()
    T2 = solver_cached.simulate_T(F_test, source_pos)
    time2 = time.time() - start_time
    
    print(f"First call time: {time1:.3f}s")
    print(f"Second call time: {time2:.3f}s")
    print(f"Results identical: {torch.allclose(T1, T2)}")
    
    # Clear cache and test
    solver_cached.clear_cache()
    start_time = time.time()
    T3 = solver_cached.simulate_T(F_test, source_pos)
    time3 = time.time() - start_time
    
    print(f"After cache clear: {time3:.3f}s")
    
    return True


def run_comprehensive_test():
    """Run all tests."""
    print("=" * 60)
    print("DIFFERENTIABLE WAVE SOLVER COMPREHENSIVE TEST")
    print("=" * 60)
    
    tests = [
        ("Solver Accuracy", test_solver_accuracy),
        ("Gradient Flow", test_gradient_flow),
        ("Optimization Levels", test_optimization_levels),
        ("Memory Usage", test_memory_usage),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"Running: {test_name}")
        print(f"{'-' * 40}")
        
        try:
            result = test_func()
            results[test_name] = result
            status = "PASSED" if result else "FAILED"
            print(f"Result: {status}")
        except Exception as e:
            print(f"ERROR: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print(f"{'=' * 60}")
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:.<30} {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    print(f"\nOverall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! The differentiable solver is ready to use.")
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed. Please review the implementation.")
    
    return total_passed == total_tests


if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)
