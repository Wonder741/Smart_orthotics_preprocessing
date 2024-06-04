import os
import shutil
import csv

# Define the source and destination directories
side = "left"
main_folder = "D:\\A\\A_Process_data\\WATMat"
source_dir = os.path.join(main_folder, "4Processed", f"png_{side}")
destination_dir = os.path.join(main_folder, "5Collect")
csv_file = os.path.join(main_folder, f"{side}_maga.csv")

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# List to store subfolder names and the count of PNG files copied
subfolder_data = []

# Iterate over each subfolder in the source directory
for subdir, _, files in os.walk(source_dir):
    subfolder_name = os.path.basename(subdir)
    png_files = [file for file in files if file.endswith('.png')]
    
    # Copy each PNG file to the destination directory with the new name
    for idx, file in enumerate(png_files, start=1):
        src_file_path = os.path.join(subdir, file)
        new_file_name = f"{side}_{subfolder_name}_{idx}.png"
        dst_file_path = os.path.join(destination_dir, new_file_name)
        
        shutil.copy(src_file_path, dst_file_path)
        print(f"Copied {src_file_path} to {dst_file_path}")
    
    # Append subfolder name and count of PNG files copied to the list
    subfolder_data.append([subfolder_name, len(png_files)])

# Write the subfolder data to the CSV file
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Subfolder Name", "PNG Files Count"])
    writer.writerows(subfolder_data)

print(f"Subfolder data has been saved to {csv_file}")