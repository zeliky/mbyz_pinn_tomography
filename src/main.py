from logger import log_message

from Terminal_and_HTML_Code.Terminal_and_HTML import terminal_html
from report_dataset_info import report_dataset_info
from dataset import TofDataset
from train import PINNTrainer
from models.resnet_ltsm import   TofToSosUNetModel
from models.pinn_unet import MultiSourceTOFModel
from models.pinn_combined import CombinedSosTofModel
from training_steps_handlers import TofToSosUNetTrainingStep, TofPredictorTrainingStep, CombinedSosTofTrainingStep
import os


#sos_checkpoint_path = 'pinn_tof-sos_model.5tumors_w_noise.pth'
#tof_checkpoint_path = 'pinn_tof-predictor_model.sources_only.pth'
sos_checkpoint_path = None
tof_checkpoint_path = None
combined_checkpoint_path = None



def train_sos_predictor():
    global sos_checkpoint_path
    epochs = 50
    trainer = PINNTrainer(model=TofToSosUNetModel(),
                          training_step_handler=TofToSosUNetTrainingStep(),
                          batch_size=2,
                          train_dataset=TofDataset(['train']),
                          val_dataset=TofDataset(['validation']),
                          epochs=epochs,
                          lr=1e-5
                          )
    if sos_checkpoint_path is not None:
        trainer.load_checkpoint(sos_checkpoint_path)
    trainer.train_model()
    log_message(' ')

    log_message("[main.py] Training pipeline complete.")



def train_tof_predictor():
    global tof_checkpoint_path
    epochs = 50
    trainer = PINNTrainer(model=MultiSourceTOFModel(in_channels=1, n_src=32, base_filters=32),
                          training_step_handler=TofPredictorTrainingStep(),
                          batch_size=1,
                          train_dataset=TofDataset(['train']),
                          val_dataset=TofDataset(['validation']),
                          epochs=epochs,
                          lr=1e-4
                          )
    if tof_checkpoint_path is not None:
        trainer.load_checkpoint(tof_checkpoint_path)
    trainer.train_model()
    log_message(' ')

    log_message("[main.py] Training pipeline complete.")


def train_combined_model():
    global combined_checkpoint_path
    epochs = 50
    trainer = PINNTrainer(model=CombinedSosTofModel(),
                          training_step_handler=CombinedSosTofTrainingStep(),
                          batch_size=5,
                          train_dataset=TofDataset(['train']),
                          val_dataset=TofDataset(['validation']),
                          epochs=epochs,
                          lr=1e-5
                          )
    if combined_checkpoint_path is not None:
        trainer.load_checkpoint(combined_checkpoint_path)
    trainer.train_model()
    log_message(' ')

    log_message("[main.py] Training pipeline complete.")


if __name__ == "__main__":
    # define the terminal_html folder and initiate the corresponding class 'terminal_html'
    log_message(' ')
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    log_message("[main.py] Starting PINN training pipeline...")

    #train_sos_predictor()
    #train_tof_predictor()
    train_combined_model()


