from logger import log_message
from settings import  app_settings
import torch
from Terminal_and_HTML_Code.Terminal_and_HTML import terminal_html
from report_dataset_info import report_dataset_info
from dataset import TofDataset
from train import PINNTrainer
from models.resnet_ltsm import TofToSosUNetModel
from models.pinn_linear import TOFtoSOSPINNLinerModel
from models.pinn_unet import MultiSourceTOFModel
from models.pinn_combined import CombinedSosTofModel
from training_steps_handlers import (TofToSosUNetTrainingStep, TofPredictorTrainingStep, CombinedSosTofTrainingStep,
                                     TOFtoSOSPINNLinerTrainingStep)
import os
import time
from TimeMeasurement.time_measurement import convert
from models.eikonal_solver import EikonalSolverMultiLayer

#sos_checkpoint_path = 'pinn_tof-sos_model.5tumors_w_noise.pth'
#tof_checkpoint_path = 'pinn_tof-predictor_model.sources_only.pth'
#sos_checkpoint_path = 'pinn_tof-predictor_model.pth'
sos_checkpoint_path = None
tof_checkpoint_path = None
multi_tof_checkpoint_path = None
combined_checkpoint_path = None



def train_multitof_to_sos_predictor():
    global multi_tof_checkpoint_path
    epochs = 30
    trainer = PINNTrainer(model=TOFtoSOSPINNLinerModel(app_settings.sources_amount),
                          training_step_handler=TOFtoSOSPINNLinerTrainingStep(),
                          batch_size=1,
                          train_dataset=TofDataset(['train']),
                          val_dataset=TofDataset(['validation']),
                          epochs=epochs,
                          lr=1e-4
                          )
    if multi_tof_checkpoint_path is not None:
        trainer.load_checkpoint(multi_tof_checkpoint_path)
    trainer.train_model()
    log_message(' ')

    trainer.visualize_training_and_validation()

    log_message("[main.py] Training pipeline complete.")


def train_sos_predictor():
    global sos_checkpoint_path
    epochs = 30
    trainer = PINNTrainer(model=TofToSosUNetModel(),
                          training_step_handler=TofToSosUNetTrainingStep(),
                          batch_size=1,
                          train_dataset=TofDataset(['train']),
                          val_dataset=TofDataset(['validation']),
                          epochs=epochs,
                          lr=1e-3,
                          scheduler_step_size=4
                          )
    if sos_checkpoint_path is not None:
        trainer.load_checkpoint(sos_checkpoint_path)
    trainer.train_model()
    log_message(' ')

    trainer.visualize_training_and_validation()

    log_message("[main.py] Training pipeline complete.")



def train_tof_predictor():
    global tof_checkpoint_path
    epochs = 2
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
    epochs = 20
    trainer = PINNTrainer(model=CombinedSosTofModel(),
                          training_step_handler=CombinedSosTofTrainingStep(),
                          batch_size=2,
                          train_dataset=TofDataset(['train']),
                          val_dataset=TofDataset(['validation']),
                          epochs=epochs,
                          lr=1e-4
                          )
    if combined_checkpoint_path is not None:
        trainer.load_checkpoint(combined_checkpoint_path)
    trainer.train_model()
    log_message(' ')

    log_message("[main.py] Training pipeline complete.")


if __name__ == "__main__":
    # define the terminal_html folder and initiate the corresponding class 'terminal_html'
    st = time.process_time()
    log_message(' ')
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    log_message("[main.py] Starting PINN training pipeline...")
    print('started')
    #train_sos_predictor()
    #train_tof_predictor()
    #train_combined_model()
    train_multitof_to_sos_predictor()
    # Measure time
    et = time.process_time()
    res = et - st
    hours, minutes, seconds = convert(res)
    log_message(" ")
    log_message('CPU Execution time: {} hours, {} Minutes, {} seconds'.format(int(hours), int(minutes), int(seconds)))


