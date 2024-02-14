import os
import numpy as np
import pandas as pd
from PIL import Image
import scipy.ndimage as ndimage

def process_file(csv_file, output_image_dir, output_image_l_dir, output_csv_dir, output_csv_l_dir):
    # 1. Load CSV as an image
    data = pd.read_csv(csv_file, header=None).values
    original_shape = data.shape

    # 2. Expand image to 51x51
    expanded_image = np.zeros((51, 51))
    x_offset = (51 - original_shape[0]) // 2
    y_offset = (51 - original_shape[1]) // 2
    expanded_image[x_offset:x_offset+original_shape[0], y_offset:y_offset+original_shape[1]] = data

    # 3. Find the centroid of the object in the expanded image
    object_pixels = expanded_image > 0
    centroid = ndimage.center_of_mass(object_pixels)

    # 4. Move the object so that the centroid is at pixel (26, 26)
    center_of_image = (26, 26)
    shift = (center_of_image[0] - centroid[0], center_of_image[1] - centroid[1])
    shifted_image = ndimage.shift(expanded_image, shift, order=0, mode='constant', cval=0)

    # Resize the image to 1020x1020
    resized_image = np.kron(shifted_image, np.ones((20,20)))

    # 5. Convert the image and resized image back to CSV
    new_csv_file = os.path.join(output_csv_dir, os.path.basename(csv_file))
    pd.DataFrame(shifted_image).to_csv(new_csv_file, header=None, index=False)
    new_csv_file_l = os.path.join(output_csv_l_dir, os.path.basename(csv_file))
    pd.DataFrame(resized_image).to_csv(new_csv_file_l, header=None, index=False)

    # 6. Save the original and resized images
    new_image_file = os.path.join(output_image_dir, os.path.splitext(os.path.basename(csv_file))[0] + '.png')
    new_image_file_l = os.path.join(output_image_l_dir, os.path.splitext(os.path.basename(csv_file))[0] + '_l.png')
    Image.fromarray(np.uint8(shifted_image * 255)).save(new_image_file)  # Save original size image
    Image.fromarray(np.uint8(resized_image * 255)).save(new_image_file_l)  # Save resized image

# Paths
input_dir = 'D:\\A\\1 Codes\\Python\\CWH\\Test'
output_image_dir = 'D:\\A\\1 Codes\\Python\\CWH\\image'
output_image_l_dir = 'D:\\A\\1 Codes\\Python\\CWH\\image_l'
output_csv_dir = 'D:\\A\\1 Codes\\Python\\CWH\\csv'
output_csv_l_dir = 'D:\\A\\1 Codes\\Python\\CWH\\csv_l'

# Create output directories if they don't exist
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_image_l_dir, exist_ok=True)
os.makedirs(output_csv_dir, exist_ok=True)
os.makedirs(output_csv_l_dir, exist_ok=True)

# Process each CSV file in the input directory
for file in os.listdir(input_dir):
    if file.endswith(".csv"):
        process_file(os.path.join(input_dir, file), output_image_dir, output_image_l_dir, output_csv_dir, output_csv_l_dir)
