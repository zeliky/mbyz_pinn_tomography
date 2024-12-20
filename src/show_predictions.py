
from dataset import TofDataset
from train import PINNTrainer
from model_pinn_resnet import TofToSosUNetModel

checkpoint_path = 'pinn_tof-sos_model.2024_12_20_13_47_29_333979-6.pth'
trainer = PINNTrainer(model=TofToSosUNetModel(),
                          train_dataset=TofDataset(['train']),
                          val_dataset=TofDataset(['test'])
    )

trainer.load_checkpoint(checkpoint_path)
trainer.visualize_predictions( )
