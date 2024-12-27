
from dataset import TofDataset
from train import PINNTrainer
from models.pinn_unet import   MultiSourceTOFModel
from training_steps_handlers import   TofPredictorTrainingStep



checkpoint_path = 'pinn_tof-predictor_model.2024_12_27_03_31_12_966505-49.pth'
trainer = PINNTrainer(model=MultiSourceTOFModel(in_channels=1, n_src=32, base_filters=32),
                      training_step_handler=TofPredictorTrainingStep(),
                      train_dataset=TofDataset(['train']),
                      val_dataset=TofDataset(['test'])
    )

trainer.load_checkpoint(checkpoint_path)
trainer.check_tof( 128,128)
