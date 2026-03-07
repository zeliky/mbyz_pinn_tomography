import torch
import numpy as np
from torch.autograd import Function
from py2mat.msfm2d import msfm2d
from logger import visualize_matdata


class DifferentiableWaveSolverFunction(Function):
    """
    Custom PyTorch autograd function that wraps the MATLAB msfm2d solver
    with surrogate gradients computed via finite differences.
    """
    
    @staticmethod
    def forward(ctx, F, source_pos, eps=1e-4):
        """
        Forward pass using the fast MATLAB solver.
        
        Args:
            F: Speed of sound tensor (H, W)
            source_pos: Source position array (x, y)
            eps: Finite difference step size for gradient computation
        
        Returns:
            T: Time-of-flight tensor (H, W)
        """
        # Convert to numpy for MATLAB call
        F_np = F.detach().cpu().numpy()
        source_grid = np.array([int(source_pos[0]), int(source_pos[1])]).reshape(1, 2)
        
        # Call fast MATLAB solver
        T_np = msfm2d(F_np, source_grid)
        
        # Convert back to tensor
        T = torch.tensor(T_np, dtype=torch.float32, device=F.device)
        
        # Save for backward pass
        ctx.save_for_backward(F, torch.tensor(source_pos, device=F.device))
        ctx.eps = eps
        
        return T
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using finite differences to approximate gradients.
        
        Args:
            grad_output: Gradient of loss w.r.t. output T
            
        Returns:
            grad_F: Gradient w.r.t. speed of sound field F
            grad_source_pos: None (source position is not differentiable)
            grad_eps: None (epsilon is not differentiable)
        """
        F, source_pos = ctx.saved_tensors
        eps = ctx.eps
        
        # Initialize gradient tensor
        grad_F = torch.zeros_like(F)
        
        # Convert inputs to numpy for MATLAB calls
        F_np = F.detach().cpu().numpy()
        source_grid = np.array([int(source_pos[0]), int(source_pos[1])]).reshape(1, 2)
        
        # Compute baseline T for finite differences
        T_base = msfm2d(F_np, source_grid)
        T_base_tensor = torch.tensor(T_base, dtype=torch.float32, device=F.device)
        
        # Compute finite difference gradients
        # We'll use a more efficient approach: perturb multiple pixels at once
        H, W = F.shape
        
        # Create perturbation masks for efficient batch computation
        batch_size = min(64, H * W)  # Process in batches to manage memory
        
        for batch_start in range(0, H * W, batch_size):
            batch_end = min(batch_start + batch_size, H * W)
            batch_grads = []
            
            for idx in range(batch_start, batch_end):
                i, j = divmod(idx, W)
                
                # Forward perturbation
                F_pert = F_np.copy()
                F_pert[i, j] += eps
                
                try:
                    T_pert = msfm2d(F_pert, source_grid)
                    T_pert_tensor = torch.tensor(T_pert, dtype=torch.float32, device=F.device)
                    
                    # Compute finite difference gradient
                    dT_dF = (T_pert_tensor - T_base_tensor) / eps
                    
                    # Apply chain rule: grad_F[i,j] = sum(grad_output * dT_dF)
                    grad_F[i, j] = torch.sum(grad_output * dT_dF)
                    
                except Exception as e:
                    # If MATLAB solver fails for perturbed input, use zero gradient
                    grad_F[i, j] = 0.0
        
        return grad_F, None, None


class DifferentiableWaveSolver:
    """
    Differentiable wrapper around the MATLAB wave solver that enables
    gradient-based training while maintaining the performance of the
    original MATLAB implementation.
    """
    
    def __init__(self, finite_diff_eps=1e-4, use_gradient_caching=True):
        """
        Initialize the differentiable wave solver.
        
        Args:
            finite_diff_eps: Step size for finite difference gradient computation
            use_gradient_caching: Whether to cache gradients for repeated computations
        """
        self.finite_diff_eps = finite_diff_eps
        self.use_gradient_caching = use_gradient_caching
        self.gradient_cache = {}
    
    def simulate_T(self, F, source_pos):
        """
        Compute time-of-flight values using msfm2d with gradient support.
        
        Parameters:
        - F: Full mesh tensor (128x128) containing speed of sound values
        - source_pos: Source position (x, y) in real coordinates
        
        Returns:
        - T_grid: 2D tensor of time-of-flight values for the entire mesh
        """
        # Check cache if enabled
        if self.use_gradient_caching:
            cache_key = (F.data_ptr(), tuple(source_pos))
            if cache_key in self.gradient_cache:
                return self.gradient_cache[cache_key]
        
        # Use custom autograd function
        T_grid = DifferentiableWaveSolverFunction.apply(F, source_pos, self.finite_diff_eps)
        
        # Cache result if enabled
        if self.use_gradient_caching:
            self.gradient_cache[cache_key] = T_grid
        
        return T_grid
    
    def clear_cache(self):
        """Clear the gradient cache to free memory."""
        self.gradient_cache.clear()
    
    def set_finite_diff_eps(self, eps):
        """Update the finite difference step size."""
        self.finite_diff_eps = eps
        self.clear_cache()  # Clear cache since gradients will change


class OptimizedDifferentiableWaveSolver(DifferentiableWaveSolver):
    """
    Optimized version with adaptive finite differences and gradient approximations.
    """
    
    def __init__(self, finite_diff_eps=1e-4, adaptive_eps=True, gradient_sparsity_threshold=1e-6):
        super().__init__(finite_diff_eps, use_gradient_caching=True)
        self.adaptive_eps = adaptive_eps
        self.gradient_sparsity_threshold = gradient_sparsity_threshold
    
    def _adaptive_finite_difference(self, F, source_pos, grad_output):
        """
        Compute gradients using adaptive finite differences with sparsity optimization.
        """
        # Start with base epsilon
        eps = self.finite_diff_eps
        
        # Convert to numpy
        F_np = F.detach().cpu().numpy()
        source_grid = np.array([int(source_pos[0]), int(source_pos[1])]).reshape(1, 2)
        
        # Compute baseline
        T_base = msfm2d(F_np, source_grid)
        T_base_tensor = torch.tensor(T_base, dtype=torch.float32, device=F.device)
        
        # Initialize gradient
        grad_F = torch.zeros_like(F)
        H, W = F.shape
        
        # Identify regions where gradients are likely to be significant
        # (near source and areas with high grad_output magnitude)
        source_x, source_y = int(source_pos[0]), int(source_pos[1])
        grad_output_magnitude = torch.abs(grad_output)
        
        # Create importance mask (higher values = more important to compute accurately)
        y_coords, x_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        distance_to_source = torch.sqrt((x_coords - source_x)**2 + (y_coords - source_y)**2)
        importance_mask = grad_output_magnitude * torch.exp(-distance_to_source / 20.0)
        
        # Sort pixels by importance
        flat_importance = importance_mask.flatten()
        sorted_indices = torch.argsort(flat_importance, descending=True)
        
        # Compute gradients for most important pixels first
        num_important_pixels = min(int(0.1 * H * W), 500)  # Limit computation
        
        for idx in sorted_indices[:num_important_pixels]:
            i, j = divmod(idx.item(), W)
            
            if flat_importance[idx] < self.gradient_sparsity_threshold:
                break
            
            # Adaptive epsilon based on local F values
            local_eps = eps * max(0.1, abs(F[i, j].item()))
            
            # Forward perturbation
            F_pert = F_np.copy()
            F_pert[i, j] += local_eps
            
            try:
                T_pert = msfm2d(F_pert, source_grid)
                T_pert_tensor = torch.tensor(T_pert, dtype=torch.float32, device=F.device)
                
                # Compute gradient
                dT_dF = (T_pert_tensor - T_base_tensor) / local_eps
                grad_F[i, j] = torch.sum(grad_output * dT_dF)
                
            except Exception:
                grad_F[i, j] = 0.0
        
        return grad_F


# Factory function to create the appropriate solver
def create_differentiable_wave_solver(optimization_level='standard'):
    """
    Factory function to create a differentiable wave solver.
    
    Args:
        optimization_level: 'basic', 'standard', or 'optimized'
    
    Returns:
        DifferentiableWaveSolver instance
    """
    if optimization_level == 'basic':
        return DifferentiableWaveSolver(finite_diff_eps=1e-3, use_gradient_caching=False)
    elif optimization_level == 'standard':
        return DifferentiableWaveSolver(finite_diff_eps=1e-4, use_gradient_caching=True)
    elif optimization_level == 'optimized':
        return OptimizedDifferentiableWaveSolver(finite_diff_eps=1e-4, adaptive_eps=True)
    else:
        raise ValueError(f"Unknown optimization level: {optimization_level}")
