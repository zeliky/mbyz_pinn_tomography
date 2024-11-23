from dataset import TofDataset
from report_dataset_info import report_dataset_info
from Terminal_and_HTML_Code.Terminal_and_HTML import terminal_html


folder = "../myOutputs/"
formatted_datetime, p = terminal_html(folder)
dataset = TofDataset(['train', 'validation'], p)
dataset.store_dataset('train_validation.pcl')
report_dataset_info(dataset, p)


# Visualize the first ToF image
visualize_tof_image(dataset, idx=0, p=p)
visualize_anatomy_image(dataset, idx=0, p=p)
visualize_sources_and_receivers(dataset, idx=0, p=p)
