import torch
from pathlib import Path
from normalizer_utils import artificial_highres, artificial_highres_with_onehot

PREPROC_DIR = Path('./src/preprocessing')
data_path = PREPROC_DIR / 'merged_normalized.pt'
output_path = PREPROC_DIR / 'merged_normalized_upscaled.pt'

data = torch.load(data_path, map_location='cpu')
x_raw = data['x']
y_raw = data['y']

scale_factor = 2.0

# Channels 5, 6, 7 are one-hot encoded and need special handling
onehot_channels = [5, 6, 7]
print("Upscaling x with one-hot preservation...")
upscaled_x_data = artificial_highres_with_onehot(x_raw, scale_factor, onehot_channels)
print("Upscaling y...")
upscaled_y_data = artificial_highres(y_raw, scale_factor)
print("Upscaling complete.")

print(f"  Raw input shape:  {tuple(x_raw.shape)}")
print(f"  Raw output shape: {tuple(y_raw.shape)}")
print(f"  Upscaled input shape: {tuple(upscaled_x_data.shape)}")
print(f"  Upscaled output shape: {tuple(upscaled_y_data.shape)}")

torch.save({
    'x': upscaled_x_data,
    'y': upscaled_y_data
}, output_path)

