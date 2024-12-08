
from dataset import TofDataset
from train import PINNTrainer
from model_pinn_unet import PhysicsInformedUNet

checkpoint_path = 'pinn_tof-sos_model.2024_12_08_19_16_45_306530-2.pth'
trainer = PINNTrainer(model=PhysicsInformedUNet(),
                          train_dataset=TofDataset(['train']),
                          val_dataset=TofDataset(['test'])
    )

trainer.load_checkpoint(checkpoint_path)
trainer.visualize_predictions( num_samples=10)
