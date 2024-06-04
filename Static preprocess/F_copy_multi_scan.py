import os
import shutil
import csv

# Define the paths
side = "left"
main_folder = "D:\\A\\A_Process_data\\3DScan"
csv_path = f"D:\\A\\A_Process_data\\WATMat\\{side}_maga.csv"
source_dir = os.path.join(main_folder, "4Processed", f"png_{side}")
destination_dir = os.path.join(main_folder, "5Collect")

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Read the CSV file and store the data in a dictionary
subfolder_counts = {}
with open(csv_path, mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for row in reader:
        subfolder_name, n = row
        subfolder_counts[subfolder_name] = int(n)

# Iterate over each subfolder in the source directory
for subdir, _, files in os.walk(source_dir):
    subfolder_name = os.path.basename(subdir)
    
    if subfolder_name in subfolder_counts:
        n = subfolder_counts[subfolder_name]
        png_files = [file for file in files if file.endswith('.png')]
        
        # Copy each PNG file to the destination directory with the new name
        for i in range(n):
            for idx, file in enumerate(png_files, start=1):
                src_file_path = os.path.join(subdir, file)
                new_file_name = f"{side}_{subfolder_name}_{i * len(png_files) + idx}.png"
                dst_file_path = os.path.join(destination_dir, new_file_name)
                
                shutil.copy(src_file_path, dst_file_path)
                print(f"Copied {src_file_path} to {dst_file_path}")

print("All files have been copied successfully.")
