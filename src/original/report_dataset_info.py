from logger import log_message
def report_dataset_info(dataset, p):
    log_message(' ')
    log_message('[report_dataset_info.py] ================== report dataset structure ==========================')
    log_message(' ')
    # Number of items in the dataset
    log_message(f"Dataset size: {len(dataset)}")

    # Access ToF images
    tof_image = dataset.tof_images[0]
    log_message(f"ToF Image shape: {tof_image.shape}")

    # Access Anatomy images
    anatomy_image = dataset.anatomy_images[0]
    log_message(f"Anatomy Image shape: {anatomy_image.shape}")

    # Access ToF data
    tof_data = dataset.tof_data[0]
    log_message(f"ToF Data keys: {tof_data.keys()}")
    log_message(f"Coords shape: {tof_data['coords'].shape}")

    # Access SoS values
    sos_values = dataset.sos_data[0]
    log_message(f"SoS Values shape: {sos_values.shape}")

    log_message(' ')
    log_message('[report_dataset_info.py] ================== end report dataset structure ==========================')
    log_message(' ')