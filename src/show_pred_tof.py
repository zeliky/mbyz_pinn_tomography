
from dataset import TofDataset
from train import PINNTrainer
from models.pinn_combined import CombinedSosTofModel
from training_steps_handlers import  CombinedSosTofTrainingStep



checkpoint_path = 'pinn_tof-predictor_model.2024_12_29_01_15_19_800618-0.pth'
trainer = PINNTrainer(model=CombinedSosTofModel(),
                      training_step_handler=CombinedSosTofTrainingStep(),
                      batch_size=3,
                      train_dataset=TofDataset(['train']),
                      val_dataset=TofDataset(['test'])
    )

trainer.load_checkpoint(checkpoint_path)
trainer.check_tof( 128,128)
