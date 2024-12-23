
from dataset import TofDataset
from train import PINNTrainer
from models.resnet_ltsm import TofToSosUNetModel

checkpoint_path = 'pinn_tof-sos_model.5tumors_w_noise.pth'
trainer = PINNTrainer(model=TofToSosUNetModel(),
                          train_dataset=TofDataset(['train']),
                          val_dataset=TofDataset(['test'])
    )

trainer.load_checkpoint(checkpoint_path)
trainer.visualize_predictions( )
