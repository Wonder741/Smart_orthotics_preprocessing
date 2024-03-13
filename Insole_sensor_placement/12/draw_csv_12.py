import os
from PIL import Image
import numpy as np
import csv

def process_image(image_path, csv_path):
    # Open the image and convert it to a numpy array
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    pixels = np.array(image)

    # Convert white (255) to 0 and black (0) to 1
    pixels = np.where(pixels == 255, 0, 1)

    # Initialize an empty array for the CSV data
    csv_data = np.zeros((64, 64), dtype=np.uint8)

    # Process each 10x10 block
    for i in range(0, 640, 10):
        for j in range(0, 640, 10):
            # Extract the 10x10 block
            block = pixels[i:i+10, j:j+10]

            # Find the maximum value in the block (1 if any pixel is black, 0 otherwise)
            max_value = np.max(block)

            # Assign the max value to the corresponding cell in the CSV data array
            csv_data[i//10, j//10] = max_value

    # Write the CSV data to a file
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in csv_data:
            writer.writerow(row)

def convert_folder_to_csv(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each PNG file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.replace('.png', '.csv'))
            process_image(input_path, output_path)
            print(f'Converted {filename} to CSV.')

# Example usage
input_folder = 'D:\\A\\1 InsoleDataset\\AutoCAD\\PNG'
output_folder = 'D:\\A\\1 InsoleDataset\\AutoCAD\\CSV'
convert_folder_to_csv(input_folder, output_folder)
