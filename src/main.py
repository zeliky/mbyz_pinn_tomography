from train import train_model
import os
from Terminal_and_HTML_Code.Terminal_and_HTML import terminal_html
 
if __name__ == "__main__":
    # define the terminal_html folder and initiate the corresponding class 'terminal_html'
    folder = "../myOutputs/"
    formatted_datetime, p = terminal_html(folder)
    p.print(f"[main.py] terminal_html folder = {folder}")
    p.print(' ')
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    p.print("[main.py] Starting PINN training pipeline...")
    p.print(' ')
    epochs = 3 #200
    
    train_model(num_epochs=epochs, batch_size=1, learning_rate=1e-3, p=p)
    p.print("[main.py] Training pipeline complete.")
