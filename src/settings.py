class AppSettings:
    output_folder: str = "../myOutputs/"
    input_folder: str = "../inputData"
    cached_dataset: str = 'train_validation.pcl'
    train_path: str = f'{input_folder}/ForLearning/'
    validation_path: str = f'{input_folder}/ForValidation/'
    test_path: str = f'{input_folder}/ForTest/'
    tof_path: str = f'{input_folder}/TimeOfFlightData/'

    anatomy_width: int = 128
    anatomy_height: int= 128
    min_sos: float=0.14
    max_sos: float=0.145
    min_tof: float=0.0
    max_tof: float=100.0
    pixel_to_mm = 1e-3



app_settings = AppSettings()
