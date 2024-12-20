from logger import log_message

from Terminal_and_HTML_Code.Terminal_and_HTML import terminal_html
from report_dataset_info import report_dataset_info
from dataset import TofDataset
from train import PINNTrainer
from model_pinn_resnet import TofToSosUNetModel
import os


checkpoint_path = 'pinn_tof-sos_model.2024_12_20_13_47_29_333979-6.pth'


def train_sos_predictor():
    global checkpoint_path
    epochs = 50
    trainer = PINNTrainer(model=TofToSosUNetModel(),
                          batch_size=2,
                          train_dataset=TofDataset(['train']),
                          val_dataset=TofDataset(['validation']),
                          epochs=epochs,
                          lr=1e-5
                          )

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


