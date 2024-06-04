import os
import shutil
import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom
import csv

def load_stl_and_properties(file_path):
    # Load the STL file and return mesh data
    return mesh.Mesh.from_file(file_path)

def append_to_csv(csv_path, file_index, max_z):
    try:
        # Convert max_z to an integer
        max_z_integer = int(max_z)
        
        # Create a single row with the file name and max_z_integer
        csv_data = [os.path.basename(file_index), max_z_integer]
        
        # Append the row to the CSV file
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_data)
        
        print(f"Data save successfully to {csv_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def generate_grayscale_image(point_cloud_path, image_output_path):
    # Load point cloud
    points = np.loadtxt(point_cloud_path, skiprows=1)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Determine bounds for the image and create grid
    x_min, x_max, y_min, y_max = x.min(), x.max(), y.min(), y.max()
    x_bins = int((x_max - x_min) * 1)  # Increased resolution by a factor of 4
    y_bins = int((y_max - y_min) * 1)  # Increased resolution by a factor of 4

    # Calculate pixel size
    pixel_width = (x_max - x_min) / x_bins
    pixel_height = (y_max - y_min) / y_bins

    # Threshold for determining if a point contributes to a pixel
    inclusion_threshold = min(pixel_width, pixel_height)  # the smaller pixel dimension

    # Create an empty grid for the image and a count grid for averaging
    image_grid = np.zeros((y_bins, x_bins))
    count_grid = np.zeros((y_bins, x_bins))

    # Populate the grid
    for xi, yi, zi in zip(x, y, z):
        x_index = int((xi - x_min) * (x_bins - 1) / (x_max - x_min))
        y_index = int((yi - y_min) * (y_bins - 1) / (y_max - y_min))

        # Calculate the center of the current grid cell
        cell_center_x = x_min + (x_index + 0.5) * pixel_width
        cell_center_y = y_min + (y_index + 0.5) * pixel_height

        # Check if point is within the inclusion threshold of the cell center
        if abs(xi - cell_center_x) <= inclusion_threshold and abs(yi - cell_center_y) <= inclusion_threshold:
            image_grid[y_index, x_index] += zi
            count_grid[y_index, x_index] += 1

    # Normalize only where counts are non-zero
    valid_mask = count_grid > 0
    image_grid[valid_mask] /= count_grid[valid_mask]
    image_grid[valid_mask] *= -1
    min_z, max_z = image_grid[valid_mask].min(), image_grid[valid_mask].max()
    #print(max_z)
    if max_z > 40:
        max_z = 40
    image_grid[valid_mask] = np.interp(image_grid[valid_mask], (0, 40), (255, 0))

    # Ensure background is black where there were no counts
    image_grid[~valid_mask] = 0

    # Apply Gaussian smoothing to the grid
    smoothed_image = gaussian_filter(image_grid, sigma=2)

    # Since the original resolution was increased by 4x, dividing by 2 gives us an effective 2x resolution
    reduced_image = zoom(smoothed_image, 0.2, order=3)  # Resize by half

    # Ensure the reduced image fits within a 640x640 frame
    final_image = np.zeros((64, 64), dtype=np.uint8)  # Create a 640x640 black background

    # Calculate dimensions to center the reduced image
    reduced_height, reduced_width = reduced_image.shape
    start_x = (64 - reduced_height) // 2
    start_y = (64 - reduced_width) // 2

    # Place the reduced image on the black background
    final_image[start_x:start_x + reduced_height, start_y:start_y + reduced_width] = reduced_image

    # Save the grayscale image
    plt.imsave(image_output_path, final_image, cmap='gray', origin='lower')
    return final_image, max_z

def process_files_in_directory(index, file_dir, csv_name):
    input_directory = os.path.join(file_dir, '1Raw', index)
    extract_directory = os.path.join(file_dir, '1Extract', index)
    csv_path = os.path.join(file_dir, csv_name)
    if not os.path.exists(extract_directory):
        os.makedirs(extract_directory)

    output_directory_left = os.path.join(file_dir, '2Greyscale', 'png_left', index)
    output_directory_right = os.path.join(file_dir, '2Greyscale', 'png_right', index)
    if not os.path.exists(output_directory_left):
        os.makedirs(output_directory_left)
    if not os.path.exists(output_directory_right):
        os.makedirs(output_directory_right)

    for filename in os.listdir(input_directory):
        if filename.endswith('.stl'):
            # Determine input and output paths
            file_path = os.path.join(input_directory, filename)
            # Determine output subdirectory based on 'Left' or 'Right' in filename
            if '_L' in filename:
                new_file_name = 'left_' + os.path.basename(index) + '.stl' 
            else: 
                new_file_name = 'right_' + os.path.basename(index) + '.stl'
            stl_extract_path = os.path.join(extract_directory, new_file_name)
            # Copy the STL file to the new directory
            shutil.copy(file_path, stl_extract_path)

            png_processed_dir = output_directory_left if '_L' in filename else output_directory_right
            mesh_data = load_stl_and_properties(file_path)

            # Prepare output paths for point cloud and image
            point_cloud_path = os.path.join(png_processed_dir, new_file_name.replace('.stl', '_point_cloud.txt'))
            image_output_path = os.path.join(png_processed_dir, new_file_name.replace('.stl', '.png'))

            # Process STL to point cloud and to grayscale image
            stl_to_point_cloud_1(mesh_data, point_cloud_path)
            processed_image, max_z = generate_grayscale_image(point_cloud_path, image_output_path)
            print(f"Processed {filename} and saved image to {image_output_path}")
            append_to_csv(csv_path, os.path.splitext(new_file_name)[0], max_z)


def sample_points_from_triangle(triangle, num_points,):
    """Sample points from a triangle using barycentric coordinates."""
    points = []
    for _ in range(num_points):
        s, t = sorted([np.random.rand(), np.random.rand()])
        f = lambda i: s * triangle[0][i] + (t-s) * triangle[1][i] + (1-t) * triangle[2][i]
        points.append([f(0), f(1), f(2)])
    return points

def stl_to_point_cloud_1(mesh_data, output_file_path, points_per_unit_area=10):
    # Load the STL mesh
    
    # Compute the area of each triangle and determine number of points to sample from each
    areas = mesh_data.areas
    total_points = (areas * points_per_unit_area).astype(int)
    
    # Sample points from each triangle
    all_points = []
    for triangle, num_points in zip(mesh_data.vectors, total_points):
        all_points.extend(sample_points_from_triangle(triangle, int(num_points)))
    
    # Save the points to the output file
    with open(output_file_path, 'w') as out_file:
        for point in all_points:
            out_file.write(f"{point[0]} {point[1]} {point[2]}\n")
    
    print(f"Saved point cloud with {len(all_points)} points to {output_file_path}")

# Define input and output directories
index = "35wengyu"
file_directory = 'D:\\A\\A_Process_data\\3DScan'
csv_file_name = "scan_process_result.csv"

# Process all STL files in the input directory
process_files_in_directory(index, file_directory, csv_file_name)
