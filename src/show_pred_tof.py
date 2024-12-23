
from dataset import TofDataset
from train import PINNTrainer
from models.resnet_ltsm import   TofPredictorModel
from training_steps_handlers import   TofPredictorTrainingStep



checkpoint_path = 'pinn_tof-predictor_model.2024_12_23_23_31_34_459849-3.pth'
trainer = PINNTrainer(model=TofPredictorModel(),
                      training_step_handler=TofPredictorTrainingStep(),
                      train_dataset=TofDataset(['train']),
                      val_dataset=TofDataset(['test'])
    )

trainer.load_checkpoint(checkpoint_path)
trainer.check_tof( )
