def report_dataset_info(dataset, p):
    p.print(' ')
    p.print('[report_dataset_info.py] ================== report dataset structure ==========================')
    p.print(' ')
    # Number of items in the dataset
    p.print(f"Dataset size: {len(dataset)}")

    # Access ToF images
    tof_image = dataset.tof_images[0]
    p.print(f"ToF Image shape: {tof_image.shape}")

    # Access Anatomy images
    anatomy_image = dataset.anatomy_images[0]
    p.print(f"Anatomy Image shape: {anatomy_image.shape}")

    # Access ToF data
    tof_data = dataset.tof_data[0]
    p.print(f"ToF Data keys: {tof_data.keys()}")
    p.print(f"Coords shape: {tof_data['coords'].shape}")

    # Access SoS values
    sos_values = dataset.sos_data[0]
    p.print(f"SoS Values shape: {sos_values.shape}")

    p.print(' ')
    p.print('[report_dataset_info.py] ================== end report dataset structure ==========================')
    p.print(' ')