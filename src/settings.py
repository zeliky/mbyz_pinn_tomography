class AppSettings:
    output_folder: str = "../myOutputs/"
    input_folder: str = "../inputData"
    cached_dataset: str = 'train_validation.pcl'
    train_path: str = f'{input_folder}/ForLearning/'
    validation_path: str = f'{input_folder}/ForValidation/'
    test_path: str = f'{input_folder}/ForTest/'
    tof_path: str = f'{input_folder}/TimeOfFlightData/'




# Usage
app_settings = AppSettings()
