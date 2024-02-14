import nibabel as nib
import numpy as np
import os
from PIL import Image

# Function to save metric data to CSV and convert to PNG
def save_metric_data(metric_data, csv_folder, png_folder, file_base_name, metric_name):
    csv_path = os.path.join(csv_folder, f"{file_base_name}_{metric_name}.csv")
    png_path = os.path.join(png_folder, f"{file_base_name}_{metric_name}.png")
    
    # Save CSV
    np.savetxt(csv_path, metric_data, delimiter=",", fmt="%d")
    
    # Normalize and save PNG
    normalized_data = (255 * (metric_data - np.min(metric_data)) / np.ptp(metric_data)).astype(np.uint8)
    img = Image.fromarray(normalized_data)
    img.save(png_path)

# Directories for NIfTI files and output folders
nii_dir = 'D:\\A\\Data\\Online data\\Plantar pressure\\CAD_WALK\\HealthyControls\\C33'
base_dir = 'D:\\A\\1 Codes\\Python\\CAD_Walk\\HealthyControls\\38_5_C33'
folders = {
    "max": ("csv_max", "png_max"),
    "mean": ("csv_mean", "png_mean"),
    "absspeed": ("csv_absspeed", "png_absspeed"),
    "integral": ("csv_integral", "png_integral"),
    "zo":("csv_01", "png_01"),
}

# Ensure output directories exist
for key in folders.values():
    os.makedirs(os.path.join(base_dir, key[0]), exist_ok=True)
    os.makedirs(os.path.join(base_dir, key[1]), exist_ok=True)

# Function to process each NIfTI file
def process_nii_file(nii_path, base_name):
    # Load the NIfTI file
    nii = nib.load(nii_path)
    data = nii.get_fdata()
    
    # Compute metrics
    metrics = {
        "max": np.max(data, axis=2),
        "mean": np.mean(data, axis=2),
        "absspeed": np.max(np.abs(np.diff(data, axis=2)), axis=2),
        "integral": np.trapz(data, axis=2),
        "zo": np.max(np.where(data != 0, 255, 0), axis=2),
    }
    
    # Save metrics
    for metric_name, metric_data in metrics.items():
        csv_folder, png_folder = folders[metric_name]
        save_metric_data(metric_data, os.path.join(base_dir, csv_folder), os.path.join(base_dir, png_folder), base_name, metric_name)

# Process each NIfTI file in the directory
for filename in os.listdir(nii_dir):
    if filename.lower().endswith('.nii'):
        process_nii_file(os.path.join(nii_dir, filename), os.path.splitext(filename)[0])
