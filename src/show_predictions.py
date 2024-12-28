
from dataset import TofDataset
from train import PINNTrainer
from models.pinn_combined import CombinedSosTofModel
from training_steps_handlers import   CombinedSosTofTrainingStep

checkpoint_path = 'pinn_tof-predictor_model.2024_12_28_15_35_20_705674-2.pth'
trainer = PINNTrainer(model=CombinedSosTofModel(),
                        training_step_handler=CombinedSosTofTrainingStep(),
                          train_dataset=TofDataset(['train']),
                          val_dataset=TofDataset(['test'])
    )

trainer.load_checkpoint(checkpoint_path)
trainer.visualize_predictions( num_samples=10)
