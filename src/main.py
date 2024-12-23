from logger import log_message

from Terminal_and_HTML_Code.Terminal_and_HTML import terminal_html
from report_dataset_info import report_dataset_info
from dataset import TofDataset
from train import PINNTrainer
from models.resnet_ltsm import TofToSosUNetModel, TofPredictorModel
from training_steps_handlers import TofToSosUNetTrainingStep, TofPredictorTrainingStep
import os


sos_checkpoint_path = 'pinn_tof-sos_model.5tumors_w_noise.pth'
tof_checkpoint_path = 'pinn_tof-predictor_model.sources_only.pth'


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
    trainer = PINNTrainer(model=TofPredictorModel(),
                          training_step_handler=TofPredictorTrainingStep(),
                          batch_size=20,
                          train_dataset=TofDataset(['train']),
                          val_dataset=TofDataset(['validation']),
                          epochs=epochs,
                          lr=1e-3
                          )
    if tof_checkpoint_path is not None:
        trainer.load_checkpoint(tof_checkpoint_path)
    trainer.train_model()
    log_message(' ')

    log_message("[main.py] Training pipeline complete.")


def train_combined_model():
    global checkpoint_path
    epochs = 50
    trainer = PINNTrainer(model=TofToSosUNetModel(),
                          training_step_handler=TofPredictorTrainingStep(),
                          batch_size=1,
                          train_dataset=TofDataset(['train']),
                          val_dataset=TofDataset(['validation']),
                          epochs=epochs,
                          lr=1e-5
                          )
    if tof_checkpoint_path is not None:
        trainer.load_checkpoint(checkpoint_path)
    trainer.train_model()
    log_message(' ')

    log_message("[main.py] Training pipeline complete.")


if __name__ == "__main__":
    # define the terminal_html folder and initiate the corresponding class 'terminal_html'
    log_message(' ')
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    log_message("[main.py] Starting PINN training pipeline...")

    #train_sos_predictor()
    train_tof_predictor()
    #train_combined_model()


