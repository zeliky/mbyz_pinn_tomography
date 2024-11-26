from train import train_model
from logger import log_message
import os
from Terminal_and_HTML_Code.Terminal_and_HTML import terminal_html
from report_dataset_info import report_dataset_info
from dataset import TofDataset
from settings import app_settings
if __name__ == "__main__":
    # define the terminal_html folder and initiate the corresponding class 'terminal_html'


    log_message(' ')
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    log_message("[main.py] Starting PINN training pipeline...")
    log_message(' ')
    epochs = 3 #200

    dataset = TofDataset(['train', 'validation'])
    train_model(dataset, num_epochs=epochs, batch_size=16, learning_rate=1e-3)
    log_message("[main.py] Training pipeline complete.")
