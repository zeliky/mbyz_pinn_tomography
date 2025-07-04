import os
from dataset import TofDataset
from settings import app_settings
from train import PINNTrainer
from models.pinn_linear import TOFtoSOSPINNLinerModel
from models.pinn_combined import CombinedSosTofModel
from models.resnet_ltsm import   TofToSosUNetModel
from models.resnet_ltsm import   TofToSosUNetModel
from models.gat import DualHeadGATModel
from models.tof_to_sos_net import TOFToSOSSuperResNet, create_tof_to_sos_net
from training_steps_handlers import   CombinedSosTofTrainingStep, TofToSosUNetTrainingStep, TOFtoSOSPINNLinerTrainingStep ,DualHeadGATTrainingStep, TOFToSOSTrainingStep
from models.eikonal_solver import EikonalSolverMultiLayer

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
checkpoint_path = 'TOFToSOSSuperResNet.2025_07_04_01_20_15_489740-49.pth'

model = create_tof_to_sos_net(model_size='medium')

trainer = PINNTrainer(model=model,
                          training_step_handler=TOFToSOSTrainingStep(
                            use_physics_loss=False,  
                            tof_range=(0, 800),    
                            sos_range=(0.08, 2.2)    
                          ),
                          train_dataset=TofDataset(['train']),
                          val_dataset=TofDataset(['test'])
    )

#trainer.load_checkpoint(checkpoint_path)
trainer.visualize_predictions( num_samples=20)

trainer.visualize_tof( num_samples=32)


