"""
Comprehensive evaluation framework comparing GNN policy vs traditional solvers
for the inverse problem of estimating SOS maps from TOF measurements.
"""

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
from tqdm import tqdm

# Import your existing modules
from RL.policy.gnn_policy import GNNPolicy
from RL.policy.enhanced_gnn_policy import create_enhanced_gnn_policy
from RL.env.acoustic_env import AcousticEnv
from RL.env.differentiable_wave_solver import create_differentiable_wave_solver
from graph.network import GraphDataset
from py2mat.msfm2d import msfm2d


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    sos_mse: float
    sos_mae: float
    tof_consistency_mse: float
    tof_consistency_mae: float
    inference_time: float
    reconstruction_accuracy: float
    relative_error: float


class TraditionalInverseSolver:
    """
    Traditional iterative solver for the inverse problem using gradient descent
    with physics-informed constraints.
    """
    
    def __init__(self, max_iterations=100, learning_rate=0.01, tolerance=1e-6):
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.solver = create_differentiable_wave_solver('standard')
    
    def solve(self, tof_measurements, sources_positions, receivers_positions, 
              initial_sos=None, domain_shape=(128, 128)):
        """
        Solve the inverse problem using iterative optimization.
        
        Args:
            tof_measurements: Measured TOF values (S, R)
            sources_positions: Source positions (S, 2)
            receivers_positions: Receiver positions (R, 2)
            initial_sos: Initial SOS guess (optional)
            domain_shape: Shape of the SOS domain
            
        Returns:
            estimated_sos: Estimated SOS map
            convergence_info: Information about convergence
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize SOS map
        if initial_sos is None:
            sos_map = torch.ones(domain_shape, device=device, requires_grad=True) * 1.5
        else:
            sos_map = initial_sos.clone().detach().requires_grad_(True)
        
        # Convert inputs to tensors
        tof_target = torch.tensor(tof_measurements, device=device, dtype=torch.float32)
        
        # Optimizer
        optimizer = torch.optim.Adam([sos_map], lr=self.learning_rate)
        
        # Convergence tracking
        losses = []
        start_time = time.time()
        
        for iteration in range(self.max_iterations):
            optimizer.zero_grad()
            
            total_loss = 0.0
            
            # Compute forward simulation for each source
            for s_idx, source_pos in enumerate(sources_positions):
                # Simulate wave propagation
                T_grid = self.solver.simulate_T(sos_map, source_pos)
                
                # Extract TOF at receiver positions
                T_receivers = self._extract_receiver_values(T_grid, receivers_positions)
                
                # Data fitting loss
                data_loss = torch.nn.functional.mse_loss(T_receivers, tof_target[s_idx])
                total_loss += data_loss
            
            # Regularization (smoothness prior)
            smoothness_loss = self._compute_smoothness_loss(sos_map)
            total_loss += 0.01 * smoothness_loss
            
            # Physics constraint (positive SOS values)
            physics_loss = torch.relu(-sos_map + 0.1).mean()  # Penalize SOS < 0.1
            total_loss += 0.1 * physics_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Clamp SOS values to reasonable range
            with torch.no_grad():
                sos_map.clamp_(0.1, 3.0)
            
            losses.append(total_loss.item())
            
            # Check convergence
            if iteration > 10 and abs(losses[-1] - losses[-2]) < self.tolerance:
                break
        
        inference_time = time.time() - start_time
        
        convergence_info = {
            'iterations': iteration + 1,
            'final_loss': losses[-1],
            'converged': iteration < self.max_iterations - 1,
            'inference_time': inference_time,
            'loss_history': losses
        }
        
        return sos_map.detach(), convergence_info
    
    def _extract_receiver_values(self, T_grid, receiver_positions):
        """Extract TOF values at receiver positions."""
        T_receivers = []
        for x, y in receiver_positions:
            x_idx = int(x)
            y_idx = int(y)
            T_receivers.append(T_grid[y_idx, x_idx])
        return torch.stack(T_receivers)
    
    def _compute_smoothness_loss(self, sos_map):
        """Compute smoothness regularization loss."""
        dx = torch.diff(sos_map, dim=1)
        dy = torch.diff(sos_map, dim=0)
        return torch.mean(dx**2) + torch.mean(dy**2)


class GNNInverseSolver:
    """
    GNN-based solver for the inverse problem.
    """
    
    def __init__(self, policy_path, config, graph_dataset, model_type='original'):
        self.config = config
        self.graph_dataset = graph_dataset
        self.model_type = model_type
        
        # Load trained policy
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_type == 'enhanced':
            self.policy = create_enhanced_gnn_policy(
                num_sensor_nodes=len(config['sources_positions']) + len(config['receivers_positions']),
                model_size='medium'
            )
        else:
            self.policy = GNNPolicy(
                num_sensor_nodes=len(config['sources_positions']) + len(config['receivers_positions'])
            )
        
        # Load trained weights if available
        if Path(policy_path).exists():
            checkpoint = torch.load(policy_path, map_location=device)
            if 'policy_state_dict' in checkpoint:
                self.policy.load_state_dict(checkpoint['policy_state_dict'])
            else:
                self.policy.load_state_dict(checkpoint)
        
        self.policy.to(device)
        self.policy.eval()
        
        # Create environment for evaluation
        self.env = AcousticEnv(
            config, graph_dataset,
            use_differentiable_solver=True,
            solver_optimization_level='standard'
        )
    
    def solve(self, tof_measurements, sources_positions, receivers_positions, 
              max_steps=50, domain_shape=(128, 128)):
        """
        Solve the inverse problem using the trained GNN policy.
        
        Args:
            tof_measurements: Measured TOF values (S, R)
            sources_positions: Source positions (S, 2)
            receivers_positions: Receiver positions (R, 2)
            max_steps: Maximum number of policy steps
            domain_shape: Shape of the SOS domain
            
        Returns:
            estimated_sos: Estimated SOS map
            inference_info: Information about the inference process
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Update config with new TOF measurements
        self.config['tof_matrix'] = torch.tensor(tof_measurements, device=device)
        
        # Reset environment
        observation = self.env.reset()
        
        start_time = time.time()
        
        # Run policy for multiple steps
        for step in range(max_steps):
            with torch.no_grad():
                action_dist, value = self.policy(observation)
                action = action_dist.mean  # Use mean action (deterministic)
            
            # Take step in environment
            observation, reward, done, info = self.env.step(action)
            
            if done:
                break
        
        inference_time = time.time() - start_time
        
        # Extract final SOS map
        estimated_sos = self.env.full_mesh.clone()
        
        inference_info = {
            'steps': step + 1,
            'final_reward': reward,
            'inference_time': inference_time,
            'converged': done
        }
        
        return estimated_sos, inference_info


class EvaluationFramework:
    """
    Comprehensive evaluation framework for comparing GNN vs traditional solvers.
    """
    
    def __init__(self, config, graph_dataset):
        self.config = config
        self.graph_dataset = graph_dataset
        self.results = []
    
    def generate_test_cases(self, num_cases=20, noise_level=0.0):
        """
        Generate test cases with known ground truth SOS maps and corresponding TOF measurements.
        
        Args:
            num_cases: Number of test cases to generate
            noise_level: Noise level to add to TOF measurements (0.0 = no noise)
            
        Returns:
            test_cases: List of test case dictionaries
        """
        test_cases = []
        solver = create_differentiable_wave_solver('standard')
        
        for case_id in range(num_cases):
            # Generate random SOS map with realistic structures
            sos_map = self._generate_realistic_sos_map()
            
            # Compute corresponding TOF measurements
            tof_matrix = []
            for source_pos in self.config['sources_positions']:
                T_grid = solver.simulate_T(sos_map, source_pos)
                T_receivers = self._extract_receiver_values(T_grid, self.config['receivers_positions'])
                
                # Add noise if specified
                if noise_level > 0:
                    noise = torch.randn_like(T_receivers) * noise_level * T_receivers.mean()
                    T_receivers += noise
                
                tof_matrix.append(T_receivers.cpu().numpy())
            
            test_case = {
                'case_id': case_id,
                'ground_truth_sos': sos_map.cpu().numpy(),
                'tof_measurements': np.array(tof_matrix),
                'sources_positions': self.config['sources_positions'],
                'receivers_positions': self.config['receivers_positions'],
                'noise_level': noise_level
            }
            
            test_cases.append(test_case)
        
        return test_cases
    
    def _generate_realistic_sos_map(self, domain_shape=(128, 128)):
        """Generate realistic SOS map with anatomical-like structures."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Base SOS value
        sos_map = torch.ones(domain_shape, device=device) * 1.5
        
        # Add circular inclusions (simulating organs/tissues)
        centers = [(40, 40), (80, 80), (60, 30), (30, 90)]
        radii = [15, 20, 12, 18]
        sos_values = [1.2, 1.8, 1.1, 2.0]
        
        y_coords, x_coords = torch.meshgrid(torch.arange(domain_shape[0]), 
                                          torch.arange(domain_shape[1]), indexing='ij')
        y_coords = y_coords.to(device)
        x_coords = x_coords.to(device)
        
        for (cx, cy), radius, sos_val in zip(centers, radii, sos_values):
            distance = torch.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
            mask = distance <= radius
            sos_map[mask] = sos_val
        
        # Add some smooth background variation
        x_norm = x_coords / domain_shape[1]
        y_norm = y_coords / domain_shape[0]
        background_variation = 0.1 * torch.sin(2 * np.pi * x_norm) * torch.cos(2 * np.pi * y_norm)
        sos_map += background_variation
        
        # Ensure reasonable SOS range
        sos_map = torch.clamp(sos_map, 0.8, 2.5)
        
        return sos_map
    
    def _extract_receiver_values(self, T_grid, receiver_positions):
        """Extract TOF values at receiver positions."""
        T_receivers = []
        for x, y in receiver_positions:
            x_idx = int(x)
            y_idx = int(y)
            T_receivers.append(T_grid[y_idx, x_idx])
        return torch.stack(T_receivers)
    
    def evaluate_solver(self, solver, test_cases, solver_name):
        """
        Evaluate a solver on test cases.
        
        Args:
            solver: Solver instance (GNNInverseSolver or TraditionalInverseSolver)
            test_cases: List of test cases
            solver_name: Name of the solver for logging
            
        Returns:
            results: List of evaluation results
        """
        results = []
        
        print(f"\n=== Evaluating {solver_name} ===")
        
        for test_case in tqdm(test_cases, desc=f"Testing {solver_name}"):
            try:
                # Solve inverse problem
                estimated_sos, solver_info = solver.solve(
                    test_case['tof_measurements'],
                    test_case['sources_positions'],
                    test_case['receivers_positions']
                )
                
                # Compute metrics
                metrics = self._compute_metrics(
                    estimated_sos, 
                    test_case['ground_truth_sos'],
                    test_case['tof_measurements'],
                    test_case['sources_positions'],
                    test_case['receivers_positions'],
                    solver_info['inference_time']
                )
                
                result = {
                    'case_id': test_case['case_id'],
                    'solver_name': solver_name,
                    'metrics': metrics,
                    'solver_info': solver_info,
                    'success': True
                }
                
            except Exception as e:
                print(f"Error in case {test_case['case_id']}: {str(e)}")
                result = {
                    'case_id': test_case['case_id'],
                    'solver_name': solver_name,
                    'metrics': None,
                    'solver_info': None,
                    'success': False,
                    'error': str(e)
                }
            
            results.append(result)
        
        return results
    
    def _compute_metrics(self, estimated_sos, ground_truth_sos, tof_measurements,
                        sources_positions, receivers_positions, inference_time):
        """Compute comprehensive evaluation metrics."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert to tensors
        if not isinstance(estimated_sos, torch.Tensor):
            estimated_sos = torch.tensor(estimated_sos, device=device)
        if not isinstance(ground_truth_sos, torch.Tensor):
            ground_truth_sos = torch.tensor(ground_truth_sos, device=device)
        
        # SOS reconstruction metrics
        sos_mse = torch.nn.functional.mse_loss(estimated_sos, ground_truth_sos).item()
        sos_mae = torch.nn.functional.l1_loss(estimated_sos, ground_truth_sos).item()
        
        # Relative error
        relative_error = (torch.abs(estimated_sos - ground_truth_sos) / 
                         (ground_truth_sos + 1e-6)).mean().item()
        
        # TOF consistency (forward simulation with estimated SOS)
        solver = create_differentiable_wave_solver('standard')
        tof_consistency_errors = []
        
        for s_idx, source_pos in enumerate(sources_positions):
            # Simulate with estimated SOS
            T_grid = solver.simulate_T(estimated_sos, source_pos)
            T_receivers_est = self._extract_receiver_values(T_grid, receivers_positions)
            
            # Compare with measurements
            T_receivers_measured = torch.tensor(tof_measurements[s_idx], device=device)
            
            mse_error = torch.nn.functional.mse_loss(T_receivers_est, T_receivers_measured).item()
            mae_error = torch.nn.functional.l1_loss(T_receivers_est, T_receivers_measured).item()
            
            tof_consistency_errors.append((mse_error, mae_error))
        
        # Average TOF consistency
        tof_consistency_mse = np.mean([err[0] for err in tof_consistency_errors])
        tof_consistency_mae = np.mean([err[1] for err in tof_consistency_errors])
        
        # Reconstruction accuracy (percentage of pixels within 10% error)
        pixel_errors = torch.abs(estimated_sos - ground_truth_sos) / (ground_truth_sos + 1e-6)
        reconstruction_accuracy = (pixel_errors < 0.1).float().mean().item()
        
        return EvaluationMetrics(
            sos_mse=sos_mse,
            sos_mae=sos_mae,
            tof_consistency_mse=tof_consistency_mse,
            tof_consistency_mae=tof_consistency_mae,
            inference_time=inference_time,
            reconstruction_accuracy=reconstruction_accuracy,
            relative_error=relative_error
        )
    
    def run_comprehensive_evaluation(self, gnn_policy_path, num_test_cases=20, 
                                   noise_levels=[0.0, 0.05, 0.1]):
        """
        Run comprehensive evaluation comparing GNN vs traditional solvers.
        
        Args:
            gnn_policy_path: Path to trained GNN policy
            num_test_cases: Number of test cases per noise level
            noise_levels: List of noise levels to test
            
        Returns:
            comprehensive_results: Dictionary with all results and analysis
        """
        all_results = []
        
        for noise_level in noise_levels:
            print(f"\n{'='*50}")
            print(f"Testing with noise level: {noise_level}")
            print(f"{'='*50}")
            
            # Generate test cases for this noise level
            test_cases = self.generate_test_cases(num_test_cases, noise_level)
            
            # Test traditional solver
            traditional_solver = TraditionalInverseSolver()
            traditional_results = self.evaluate_solver(
                traditional_solver, test_cases, f"Traditional (noise={noise_level})"
            )
            
            # Test original GNN solver
            try:
                gnn_solver = GNNInverseSolver(
                    gnn_policy_path, self.config, self.graph_dataset, model_type='original'
                )
                gnn_results = self.evaluate_solver(
                    gnn_solver, test_cases, f"GNN Original (noise={noise_level})"
                )
            except Exception as e:
                print(f"Could not load original GNN policy: {e}")
                gnn_results = []
            
            # Test enhanced GNN solver
            try:
                enhanced_gnn_solver = GNNInverseSolver(
                    gnn_policy_path, self.config, self.graph_dataset, model_type='enhanced'
                )
                enhanced_gnn_results = self.evaluate_solver(
                    enhanced_gnn_solver, test_cases, f"GNN Enhanced (noise={noise_level})"
                )
            except Exception as e:
                print(f"Could not load enhanced GNN policy: {e}")
                enhanced_gnn_results = []
            
            # Combine results for this noise level
            noise_results = {
                'noise_level': noise_level,
                'traditional': traditional_results,
                'gnn_original': gnn_results,
                'gnn_enhanced': enhanced_gnn_results,
                'test_cases': test_cases
            }
            
            all_results.append(noise_results)
        
        # Analyze and summarize results
        comprehensive_results = self._analyze_results(all_results)
        
        # Save results
        self._save_results(comprehensive_results)
        
        # Generate plots
        self._generate_plots(comprehensive_results)
        
        return comprehensive_results
    
    def _analyze_results(self, all_results):
        """Analyze and summarize evaluation results."""
        analysis = {
            'summary': {},
            'detailed_results': all_results,
            'statistical_analysis': {}
        }
        
        # Compute summary statistics for each solver and noise level
        for noise_result in all_results:
            noise_level = noise_result['noise_level']
            
            for solver_type in ['traditional', 'gnn_original', 'gnn_enhanced']:
                results = noise_result[solver_type]
                if not results:
                    continue
                
                # Extract successful results
                successful_results = [r for r in results if r['success']]
                if not successful_results:
                    continue
                
                # Compute statistics
                metrics_list = [r['metrics'] for r in successful_results]
                
                stats = {
                    'success_rate': len(successful_results) / len(results),
                    'avg_sos_mse': np.mean([m.sos_mse for m in metrics_list]),
                    'std_sos_mse': np.std([m.sos_mse for m in metrics_list]),
                    'avg_sos_mae': np.mean([m.sos_mae for m in metrics_list]),
                    'avg_tof_consistency_mse': np.mean([m.tof_consistency_mse for m in metrics_list]),
                    'avg_inference_time': np.mean([m.inference_time for m in metrics_list]),
                    'avg_reconstruction_accuracy': np.mean([m.reconstruction_accuracy for m in metrics_list]),
                    'avg_relative_error': np.mean([m.relative_error for m in metrics_list])
                }
                
                key = f"{solver_type}_noise_{noise_level}"
                analysis['summary'][key] = stats
        
        return analysis
    
    def _save_results(self, results, filename='evaluation_results.json'):
        """Save evaluation results to file."""
        # Convert dataclass objects to dictionaries for JSON serialization
        def convert_metrics(obj):
            if isinstance(obj, EvaluationMetrics):
                return obj.__dict__
            return obj
        
        # Create a JSON-serializable version
        json_results = json.loads(json.dumps(results, default=convert_metrics))
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to {filename}")
    
    def _generate_plots(self, results):
        """Generate comparison plots."""
        # Extract data for plotting
        noise_levels = []
        traditional_mse = []
        gnn_original_mse = []
        gnn_enhanced_mse = []
        traditional_time = []
        gnn_original_time = []
        gnn_enhanced_time = []
        
        for key, stats in results['summary'].items():
            if 'traditional_noise_' in key:
                noise_level = float(key.split('_')[-1])
                noise_levels.append(noise_level)
                traditional_mse.append(stats['avg_sos_mse'])
                traditional_time.append(stats['avg_inference_time'])
            elif 'gnn_original_noise_' in key:
                gnn_original_mse.append(stats['avg_sos_mse'])
                gnn_original_time.append(stats['avg_inference_time'])
            elif 'gnn_enhanced_noise_' in key:
                gnn_enhanced_mse.append(stats['avg_sos_mse'])
                gnn_enhanced_time.append(stats['avg_inference_time'])
        
        # Create comparison plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: SOS MSE vs Noise Level
        ax1.plot(noise_levels, traditional_mse, 'o-', label='Traditional', linewidth=2)
        if gnn_original_mse:
            ax1.plot(noise_levels, gnn_original_mse, 's-', label='GNN Original', linewidth=2)
        if gnn_enhanced_mse:
            ax1.plot(noise_levels, gnn_enhanced_mse, '^-', label='GNN Enhanced', linewidth=2)
        ax1.set_xlabel('Noise Level')
        ax1.set_ylabel('SOS MSE')
        ax1.set_title('Reconstruction Accuracy vs Noise')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Inference Time vs Noise Level
        ax2.plot(noise_levels, traditional_time, 'o-', label='Traditional', linewidth=2)
        if gnn_original_time:
            ax2.plot(noise_levels, gnn_original_time, 's-', label='GNN Original', linewidth=2)
        if gnn_enhanced_time:
            ax2.plot(noise_levels, gnn_enhanced_time, '^-', label='GNN Enhanced', linewidth=2)
        ax2.set_xlabel('Noise Level')
        ax2.set_ylabel('Inference Time (s)')
        ax2.set_title('Inference Speed vs Noise')
        ax2.legend()
        ax2.grid(True)
        ax2.set_yscale('log')
        
        # Plot 3: Accuracy vs Speed Trade-off
        if traditional_mse and traditional_time:
            ax3.scatter(traditional_time, traditional_mse, s=100, label='Traditional', alpha=0.7)
        if gnn_original_mse and gnn_original_time:
            ax3.scatter(gnn_original_time, gnn_original_mse, s=100, label='GNN Original', alpha=0.7)
        if gnn_enhanced_mse and gnn_enhanced_time:
            ax3.scatter(gnn_enhanced_time, gnn_enhanced_mse, s=100, label='GNN Enhanced', alpha=0.7)
        ax3.set_xlabel('Inference Time (s)')
        ax3.set_ylabel('SOS MSE')
        ax3.set_title('Accuracy vs Speed Trade-off')
        ax3.legend()
        ax3.grid(True)
        ax3.set_xscale('log')
        
        # Plot 4: Success Rate Comparison
        solver_names = []
        success_rates = []
        for key, stats in results['summary'].items():
            if 'noise_0.0' in key:  # Only show clean data results
                solver_name = key.replace('_noise_0.0', '').replace('_', ' ').title()
                solver_names.append(solver_name)
                success_rates.append(stats['success_rate'] * 100)
        
        ax4.bar(solver_names, success_rates, alpha=0.7)
        ax4.set_ylabel('Success Rate (%)')
        ax4.set_title('Solver Success Rates (Clean Data)')
        ax4.set_ylim(0, 105)
        for i, v in enumerate(success_rates):
            ax4.text(i, v + 1, f'{v:.1f}%', ha='center')
        
        plt.tight_layout()
        plt.savefig('solver_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Comparison plots saved as 'solver_comparison.png'")
    
    def print_summary(self, results):
        """Print a summary of the evaluation results."""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        for key, stats in results['summary'].items():
            solver_type = key.split('_noise_')[0].replace('_', ' ').title()
            noise_level = key.split('_noise_')[1]
            
            print(f"\n{solver_type} (Noise: {noise_level})")
            print("-" * 40)
            print(f"Success Rate: {stats['success_rate']*100:.1f}%")
            print(f"SOS MSE: {stats['avg_sos_mse']:.6f} ± {stats['std_sos_mse']:.6f}")
            print(f"SOS MAE: {stats['avg_sos_mae']:.6f}")
            print(f"TOF Consistency MSE: {stats['avg_tof_consistency_mse']:.6f}")
            print(f"Reconstruction Accuracy: {stats['avg_reconstruction_accuracy']*100:.1f}%")
            print(f"Relative Error: {stats['avg_relative_error']*100:.1f}%")
            print(f"Inference Time: {stats['avg_inference_time']:.3f}s")


def main():
    """Main function to run the evaluation."""
    # Example configuration (you'll need to adapt this to your setup)
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'c_init': 1.5,
        'sources_positions': np.random.rand(16, 2) * 128,  # 16 random sources
        'receivers_positions': np.random.rand(32, 2) * 128,  # 32 random receivers
        'full_mesh_resolution': (128, 128),
        'selected_sources': list(range(16))
    }
    
    # Create graph dataset
    graph_config = {
        'c_init': 1.5,
        't_init': 0.0,
        'x_range': (10, 118),
        'y_range': (10, 118),
        'nx': 4,
        'ny': 4,
        'mesh_node_k': 9,
        'sensor_k': 5,
        'num_source_nodes': 16,
        'num_receiver_nodes': 32
    }
    
    graph_dataset = GraphDataset(**graph_config)
    
    # Create evaluation framework
    evaluator = EvaluationFramework(config, graph_dataset)
    
    # Run comprehensive evaluation
    # Note: You'll need to provide the path to your trained GNN policy
    gnn_policy_path = "path/to/your/trained/policy.pth"
    
    results = evaluator.run_comprehensive_evaluation(
        gnn_policy_path=gnn_policy_path,
        num_test_cases=10,  # Start with fewer cases for testing
        noise_levels=[0.0, 0.05]  # Test clean and noisy data
    )
    
    # Print summary
    evaluator.print_summary(results)


if __name__ == "__main__":
    main()
