import os
from dataset import TofDataset
from train import PINNTrainer
from models.pinn_combined import CombinedSosTofModel
from models.resnet_ltsm import   TofToSosUNetModel
from training_steps_handlers import   CombinedSosTofTrainingStep, TofToSosUNetTrainingStep
from models.eikonal_solver import EikonalSolverMultiLayer

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
checkpoint_path = 'pinn_tof-predictor_model.2025_01_07_14_41_45_862989-0.pth'

solver = EikonalSolverMultiLayer(num_layers=3, speed_of_sound=1450, domain_size=0.128, grid_resolution=128)
solver.to('cpu')
trainer = PINNTrainer(model=TofToSosUNetModel(),
                          training_step_handler=TofToSosUNetTrainingStep(solver),
                          train_dataset=TofDataset(['train']),
                          val_dataset=TofDataset(['test'])
    )

trainer.load_checkpoint(checkpoint_path)
trainer.visualize_predictions( num_samples=10)
