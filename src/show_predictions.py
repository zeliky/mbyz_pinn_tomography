
from dataset import TofDataset
from train import PINNTrainer
from models.resnet_ltsm import TofToSosUNetModel
from training_steps_handlers import TofToSosUNetTrainingStep

checkpoint_path = 'pinn_tof-sos_model.2tumors.pth'
trainer = PINNTrainer(model=TofToSosUNetModel(),
                        training_step_handler=TofToSosUNetTrainingStep(),
                          train_dataset=TofDataset(['train']),
                          val_dataset=TofDataset(['test'])
    )

trainer.load_checkpoint(checkpoint_path)
trainer.visualize_predictions( )
