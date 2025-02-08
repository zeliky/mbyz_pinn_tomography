import os
from dataset import TofDataset
from train import PINNTrainer
from models.pinn_combined import CombinedSosTofModel
from models.resnet_ltsm import   TofToSosUNetModel
from training_steps_handlers import   CombinedSosTofTrainingStep, TofToSosUNetTrainingStep
from models.eikonal_solver import EikonalSolverMultiLayer

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
checkpoint_path = 'pinn_tof-predictor_model.2025_02_09_00_36_38_771232-2.pth'


trainer = PINNTrainer(model=TofToSosUNetModel(),
                          training_step_handler=TofToSosUNetTrainingStep(),
                          train_dataset=TofDataset(['train']),
                          val_dataset=TofDataset(['test'])
    )

trainer.load_checkpoint(checkpoint_path)
trainer.visualize_predictions( num_samples=20)
