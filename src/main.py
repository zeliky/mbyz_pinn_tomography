from logger import log_message

from Terminal_and_HTML_Code.Terminal_and_HTML import terminal_html
from report_dataset_info import report_dataset_info
from dataset import TofDataset
from train import PINNTrainer
from model_pinn_unet import PhysicsInformedUNet
import os
if __name__ == "__main__":
    # define the terminal_html folder and initiate the corresponding class 'terminal_html'
    log_message(' ')
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    log_message("[main.py] Starting PINN training pipeline...")
    epochs = 50
    trainer = PINNTrainer(model=PhysicsInformedUNet(),
                          train_dataset=TofDataset(['train']),
                          val_dataset=TofDataset(['validation']),
                          epochs=epochs,
                          lr=1e-3
    )
    #trainer.load_checkpoint('pinn_tof-sos_model.2024_12_07_23_19_18_097306-2.pth')
    trainer.train_model()
    log_message(' ')


    log_message("[main.py] Training pipeline complete.")
