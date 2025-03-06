import os
from dataset import TofDataset
from settings import app_settings
from train import PINNTrainer
from models.pinn_linear import TOFtoSOSPINNLinerModel
from models.pinn_combined import CombinedSosTofModel
from models.resnet_ltsm import   TofToSosUNetModel
from models.resnet_ltsm import   TofToSosUNetModel
from models.gat import DualHeadGATModel
from training_steps_handlers import   CombinedSosTofTrainingStep, TofToSosUNetTrainingStep, TOFtoSOSPINNLinerTrainingStep ,DualHeadGATTrainingStep
from models.eikonal_solver import EikonalSolverMultiLayer

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
checkpoint_path = 'DualHeadGATModel.2025_03_06_23_46_59_997478-0.pth'


trainer = PINNTrainer(model=DualHeadGATModel(),
                          training_step_handler=DualHeadGATTrainingStep(),
                          train_dataset=TofDataset(['train']),
                          val_dataset=TofDataset(['test'])
    )

#trainer.load_checkpoint(checkpoint_path)
#trainer.visualize_predictions( num_samples=20)

trainer.visualize_tof( num_samples=32)
