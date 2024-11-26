from dataset import TofDataset
from report_dataset_info import report_dataset_info
from Terminal_and_HTML_Code.Terminal_and_HTML import terminal_html
from torch.utils.data import DataLoader


folder = "../myOutputs/"
formatted_datetime, p = terminal_html(folder)
dataset = TofDataset(['train', 'validation'])

data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

for batch in data_loader:
    tof_input = batch['tof_input']  # Shape: [batch_size, 1, 128, 128]
    sos_image = batch['sos_image']  # Shape: [batch_size, 1, 128, 128]
    tof_data = batch['tof_data']  # Shape: [batch_size, 1, 128, 128]
    print(f"tof: {tof_input.shape}")
    print(f"sos: {sos_image.shape}")
    print(f"source: {tof_data['x_s'].shape}")
    print(f"receivers: {tof_data['x_r'].shape}")
    print(f"observed_tof: {tof_data['x_o'].shape}")
    break

