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
from models.gat import DualHeadGATModel, SosEstimator
from models.tof_to_sos_net import TOFToSOSSuperResNet, create_tof_to_sos_net
from models.tof_to_sos_classifier import create_tof_to_sos_classifier
from RL.policy.gnn_policy import GNNPolicy
from training_steps_handlers import (RLAgetTrainingStep,TofToSosUNetTrainingStep, TofPredictorTrainingStep, CombinedSosTofTrainingStep,
                                     TOFtoSOSPINNLinerTrainingStep, DualHeadGATTrainingStep, TOFToSOSTrainingStep, TOFToSOSClassificationTrainingStep)
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
#gat_tof_sos_checkpoint_path = 'bc
# _ready.pth'
gat_tof_sos_checkpoint_path = None
rl_gat_tof_sos_checkpoint_path = None


def train_gat_rl_agent_gat_policy():
    global rl_gat_tof_sos_checkpoint_path
    epochs = 30
    trainer = PINNTrainer(model=GNNPolicy(num_sensor_nodes=app_settings.sources_amount + app_settings.receivers_amount),
                          training_step_handler=RLAgetTrainingStep(),
                          batch_size=1,
                          train_dataset=TofDataset(['train']),
                          val_dataset=TofDataset(['validation']),
                          epochs=epochs,
                          lr=1e-4
                          )
    if rl_gat_tof_sos_checkpoint_path is not None:
        trainer.load_checkpoint(rl_gat_tof_sos_checkpoint_path)
    trainer.train_model()
    log_message(' ')

    trainer.visualize_training_and_validation()

    log_message("[main.py] Training pipeline complete.")
def train_gat_tof_sos_predictor():
    global gat_tof_sos_checkpoint_path
    epochs = 30
    grid_res = 8
    mesh_node_connections = 20
    estimator = SosEstimator(num_nodes=grid_res*grid_res+64, init_value=0.15, fmm_iterations=10)
    trainer = PINNTrainer(model=DualHeadGATModel(),
                          training_step_handler=DualHeadGATTrainingStep(estimator, nx=grid_res, ny=grid_res, mnc=mesh_node_connections),
                          batch_size=1,
                          train_dataset=TofDataset(['train']),
                          val_dataset=TofDataset(['validation']),
                          epochs=epochs,
                          lr=1e-4
                          )
    if gat_tof_sos_checkpoint_path is not None:
        trainer.load_checkpoint(gat_tof_sos_checkpoint_path)
    trainer.train_model()
    log_message(' ')

    trainer.visualize_training_and_validation()

    log_message("[main.py] Training pipeline complete.")

def train_multitof_to_sos_predictor():
    global multi_tof_checkpoint_path
    epochs = 30
    trainer = PINNTrainer(model=TOFtoSOSPINNLinerModel(app_settings.sources_amount),
                          training_step_handler=TOFtoSOSPINNLinerTrainingStep(),
                          batch_size=1,
                          train_dataset=TofDataset(['train']),
                          val_dataset=TofDataset(['validation']),
                          epochs=epochs,
                          lr=1e-3
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
                          lr=1e-1,
                          scheduler_step_size=4
                          )
    if sos_checkpoint_path is not None:
        trainer.load_checkpoint(sos_checkpoint_path)
    trainer.train_model()
    log_message(' ')

    trainer.visualize_training_and_validation()

    log_message("[main.py] Training pipeline complete.")


def train_tof_to_sos_super_res():
    """
    Train the TOF-to-SOS super-resolution network.
    Converts 32x32 TOF matrix to 128x128 SOS map.
    """
    global sos_checkpoint_path
    epochs = 50
    
    # Create model - you can choose 'light', 'medium', or 'heavy'
    model = create_tof_to_sos_net(model_size='medium')
    
    # Print model information
    info = model.get_model_info()
    log_message(f"TOF-to-SOS Model Info:")
    log_message(f"  Parameters: {info['total_parameters']:,}")
    log_message(f"  Input size: {info['input_size']}")
    log_message(f"  Output size: {info['output_size']}")
    log_message(f"  Upsampling factor: {info['upsampling_factor']}x")
    log_message(f"  Use attention: {info['use_attention']}")
    log_message(f"  Use residual: {info['use_residual']}")
    
    # Create trainer
    trainer = PINNTrainer(
        model=model,
        training_step_handler=TOFToSOSTrainingStep(
            use_physics_loss=False,  # Disable physics loss initially for debugging
            tof_range=(0, 800),    
            sos_range=(0.08, 2.2)    
        ),
        batch_size=15,  # Increase batch size for better gradient estimates
        train_dataset=TofDataset(['train']),
        val_dataset=TofDataset(['validation']),
        epochs=epochs,
        lr=1e-3  # Increase learning rate        
    )
    
    # Load checkpoint if available
    if sos_checkpoint_path is not None:
        trainer.load_checkpoint(sos_checkpoint_path)
        log_message(f"Loaded checkpoint: {sos_checkpoint_path}")
    
    # Train the model
    log_message("Starting TOF-to-SOS super-resolution training...")
    trainer.train_model()
    log_message(' ')

    # Visualize training progress
    trainer.visualize_training_and_validation()

    log_message("[main.py] TOF-to-SOS super-resolution training complete.")



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


def train_tof_to_sos_classifier():
    """
    Train the TOF-to-SOS binary classification network.
    Converts 32x32 TOF matrix to 128x128 binary probability map indicating "interesting" SOS regions (>1.5).
    This approach should solve the convergence issues by focusing on binary classification instead of regression.
    """
    global sos_checkpoint_path
    epochs = 50
    sos_threshold = 1.1  # Threshold for "interesting" regions
    
    # Create classification model - you can choose 'light', 'medium', or 'heavy'
    model = create_tof_to_sos_classifier(model_size='medium', sos_threshold=sos_threshold)
    
    # Print model information
    info = model.get_model_info()
    log_message(f"TOF-to-SOS Classifier Info:")
    log_message(f"  Parameters: {info['total_parameters']:,}")
    log_message(f"  Input size: {info['input_size']}")
    log_message(f"  Output size: {info['output_size']}")
    log_message(f"  Model type: {info['model_type']}")
    log_message(f"  SOS threshold: {info['sos_threshold']}")
    log_message(f"  Output type: {info['output_type']}")
    log_message(f"  Use attention: {info['use_attention']}")
    log_message(f"  Use residual: {info['use_residual']}")
    
    # Create trainer with classification training step
    trainer = PINNTrainer(
        model=model,
        training_step_handler=TOFToSOSClassificationTrainingStep(
            sos_threshold=sos_threshold,
            tof_range=(0, 800),    
            sos_range=(0.08, 2.2),
            use_focal_loss=False,  # Disable focal loss initially to prevent explosion
            use_physics_loss=True,  # Enable eikonal physics constraint |∇T| = 1/c
            use_morphological_loss=True,  # Enable spatial coherence constraints
            use_precision_loss=True,  # Enable precision-focused F-beta loss
            use_hard_negative_mining=True  # Enable hard negative mining for false positives
        ),
        batch_size=5,  # Smaller batch size for more stable training
        train_dataset=TofDataset(['train']),
        val_dataset=TofDataset(['validation']),
        epochs=epochs,
        lr=1e-4  # Lower learning rate to prevent gradient explosion
    )
    
    # Load checkpoint if available
    if sos_checkpoint_path is not None:
        trainer.load_checkpoint(sos_checkpoint_path)
        log_message(f"Loaded checkpoint: {sos_checkpoint_path}")
    
    # Train the model
    log_message("Starting TOF-to-SOS binary classification training...")
    log_message(f"Target: Detect regions with SOS > {sos_threshold}")
    log_message("This should solve convergence issues by using binary classification instead of regression.")
    trainer.train_model()
    log_message(' ')

    # Visualize training progress
    trainer.visualize_training_and_validation()

    log_message("[main.py] TOF-to-SOS binary classification training complete.")
    log_message("Next step: Use RL agent to refine exact SOS values in detected regions.")


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
    #train_multitof_to_sos_predictor()
    #train_gat_tof_sos_predictor()
    # train_gat_rl_agent_gat_policy()
    #train_tof_to_sos_super_res()
    train_tof_to_sos_classifier()
    # Measure time
    et = time.process_time()
    res = et - st
    hours, minutes, seconds = convert(res)
    log_message(" ")
    log_message('CPU Execution time: {} hours, {} Minutes, {} seconds'.format(int(hours), int(minutes), int(seconds)))
