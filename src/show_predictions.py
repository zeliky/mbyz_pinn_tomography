import os
from dataset import TofDataset
from settings import app_settings
from train import PINNTrainer
from models.pinn_linear import TOFtoSOSPINNLinerModel
from models.pinn_combined import CombinedSosTofModel
from models.resnet_ltsm import   TofToSosUNetModel
from training_steps_handlers import   CombinedSosTofTrainingStep, TofToSosUNetTrainingStep, TOFtoSOSPINNLinerTrainingStep
from models.eikonal_solver import EikonalSolverMultiLayer

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
checkpoint_path = 'TOFtoSOSPINNLinerModel.2025_02_16_16_53_07_588042-1.pth'


trainer = PINNTrainer(model=TOFtoSOSPINNLinerModel(app_settings.sources_amount),
                          training_step_handler=TOFtoSOSPINNLinerTrainingStep(),
                          train_dataset=TofDataset(['train']),
                          val_dataset=TofDataset(['test'])
    )

trainer.load_checkpoint(checkpoint_path)
trainer.visualize_predictions( num_samples=20)
