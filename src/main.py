from train import train_model

import os
from Terminal_and_HTML_Code.Terminal_and_HTML import terminal_html
from report_dataset_info import report_dataset_info
from dataset import TofDataset
if __name__ == "__main__":
    # define the terminal_html folder and initiate the corresponding class 'terminal_html'
    folder = "../myOutputs/"
    cached_dataset = 'train_validation.pcl'
    formatted_datetime, p = terminal_html(folder)
    p.print(f"[main.py] terminal_html folder = {folder}")
    p.print(' ')
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    p.print("[main.py] Starting PINN training pipeline...")
    p.print(' ')
    epochs = 3 #200

    dataset = TofDataset.load_dataset(cached_dataset)
    train_model(dataset, num_epochs=epochs, batch_size=1, learning_rate=1e-3, p=p)
    p.print("[main.py] Training pipeline complete.")
