import cv2
import numpy as np
import os
import csv
import pandas as pd

def calculate_centroid_of_white_mask(binary):
    # Calculate the moments of the binary image
    M = cv2.moments(binary)

    # Calculate the centroid of the white mask
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    centroid = [cx, cy]
    return centroid

# Use the minimum frame method to locate the point image center, and move the center to the image center (320, 320)
def align_centers(points_mask):
    frame_center = calculate_centroid_of_white_mask(points_mask)

    # Calculate the center of the object mask
    object_center = (32, 32)

    # Calculate the translation needed to move the frame center to the object center
    translation = np.array(object_center) - np.array(frame_center)

    # Apply the translation to the points mask
    rows, cols = points_mask.shape
    M = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
    aligned_points_mask = cv2.warpAffine(points_mask, M, (cols, rows))

    return aligned_points_mask, translation, object_center

# Rotated the aligned point image based on image center, find the optimal angle that have the maximum overlap with the object mask
def test_rotation(object_mask, aligned_points_mask, rotation_center):
    rows, cols = aligned_points_mask.shape
    max_overlap = 0
    best_angle = 0
    optimal_rotate_image = None

    for angle in range(0, 360, 1):  # Rotate from 0 to 180 degrees in steps of 10
        M = cv2.getRotationMatrix2D(rotation_center, angle, 1)
        rotated_points = cv2.warpAffine(aligned_points_mask, M, (cols, rows))
        overlap = np.sum((object_mask & rotated_points) > 0)

        if max_overlap < overlap:
            max_overlap = overlap
            best_angle = angle
            optimal_rotate_image = rotated_points
    return optimal_rotate_image, max_overlap, best_angle

# Align the roated point image in a short range x(-bias_range ,bias_range) and y(-bias_range, bias_range) to further optimal the image alignment
def test_bias(object_mask, points_mask, bias_range=10):
    rows, cols = points_mask.shape
    max_overlap = 0
    optimal_bias = (0, 0)
    optimal_moved_image = None

    # Test different biases
    for dx in range(-bias_range, bias_range + 1):
        for dy in range(-bias_range, bias_range + 1):
            # Apply the bias
            M_translate = np.float32([[1, 0, dx], [0, 1, dy]])
            translated_points = cv2.warpAffine(points_mask, M_translate, (cols, rows))

            # Calculate the overlap
            overlap = np.sum((object_mask & translated_points) > 0)

            # Update the maximum overlap and optimal bias
            if overlap > max_overlap:
                max_overlap = overlap
                optimal_bias = (dx, dy)
                optimal_moved_image = translated_points

    return optimal_moved_image, max_overlap, optimal_bias

def main():
    index1 = 'left'
    id_index = '05zhengkaiwen'
    size_index = '42m.png'
    side_index = index1 + '_mask'
    mask_dir = 'D:\\A\\A_Process_data\\AutoCAD'
    object_image_path = os.path.join(mask_dir, side_index, size_index)
    csv_index = 'csv_' + index1
    png_index = 'png_' + index1
    #png_index = 'png_right'
    point_dir = 'D:\\A\\A_Process_data\\WATMat\\2Average'
    points_image_folder = os.path.join(point_dir, png_index, id_index)
    debug_dir = 'D:\\A\\A_Process_data\\WATMat\\3Debug'
    modified_image_folder = os.path.join(debug_dir, png_index, id_index)
    modified_csv_folder = os.path.join(debug_dir, csv_index, id_index)

    # Define the path for the processed points images
    processed_dir = 'D:\\A\\A_Process_data\\WATMat\\4Processed'
    processed_points_image_folder = os.path.join(processed_dir, png_index, id_index)

    # Ensure the processed points image folder exists
    if not os.path.exists(processed_points_image_folder):
        os.makedirs(processed_points_image_folder)

    # Ensure the modified image folder exists
    if not os.path.exists(modified_image_folder):
        os.makedirs(modified_image_folder)

    # Ensure the modified image folder exists
    if not os.path.exists(modified_csv_folder):
        os.makedirs(modified_csv_folder)

    # CSV file path
    csv_file_path = os.path.join(modified_image_folder, 'results.csv')

    # Load the object mask
    object_image = cv2.imread(object_image_path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(object_image, (64, 64), interpolation=cv2.INTER_AREA)
    object_edges = cv2.Canny(resized_image, 100, 200)
    
    object_mask = cv2.threshold(resized_image, 1, 255, cv2.THRESH_BINARY)[1]
    object_mask_edge = cv2.threshold(object_edges, 1, 255, cv2.THRESH_BINARY)[1]

    # Open the CSV file for writing
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Image File Name', 'Optimal Angle', 'dx', 'dy'])

        # Process each points image in the folder
        for filename in os.listdir(points_image_folder):
            if filename.endswith('.png'):
                points_image_path = os.path.join(points_image_folder, filename)
                modified_image_path = os.path.join(modified_image_folder, 'modified_' + filename)
                processed_points_image_path = os.path.join(processed_points_image_folder, 'processed_' + filename)  # Path for the processed points image

                # Load and preprocess the points image
                points_image = cv2.imread(points_image_path, cv2.IMREAD_GRAYSCALE)
                points_mask = cv2.threshold(points_image, 1, 255, cv2.THRESH_BINARY)[1]

                # Align centers, rotate, and find optimal bias
                aligned_points_mask, translation, centroid = align_centers(points_mask)
                # Apply the same transformations to points_image
                rows, cols = points_image.shape
                M = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
                aligned_points_image = cv2.warpAffine(points_image, M, (cols, rows))  # Alignment
                optimal_angle = 0
                optimal_translation = [0, 0]
                current_overlap = 0
                max_overlap = 10

                if(current_overlap < max_overlap):
                    current_overlap = max_overlap
                    optimal_rotate_image, max_overlap, best_angle = test_rotation(object_mask, aligned_points_mask, centroid)
                    
                    if 181 <= best_angle <= 359:
                        best_angle = best_angle - 360
                    optimal_angle += best_angle

                    M_rotate = cv2.getRotationMatrix2D(centroid, best_angle, 1)  # Rotation
                    rotated_points_image = cv2.warpAffine(aligned_points_image, M_rotate, (cols, rows))

                    aligned_points_mask, max_overlap, optimal_bias = test_bias(object_mask, optimal_rotate_image)
                    optimal_translation[0] += optimal_bias[0]
                    optimal_translation[1] += optimal_bias[1]

                    M_translate = np.float32([[1, 0, optimal_bias[0]], [0, 1, optimal_bias[1]]])  # Bias
                    processed_points_image = cv2.warpAffine(rotated_points_image, M_translate, (cols, rows))

                    centroid = calculate_centroid_of_white_mask(aligned_points_mask)

                print(max_overlap, optimal_angle, optimal_translation)

                # Overlay the object mask and the modified points mask for debugging
                debug_image = cv2.cvtColor(object_mask_edge, cv2.COLOR_GRAY2BGR)
                debug_image[:, :, 1] = np.maximum(debug_image[:, :, 1], aligned_points_mask)

                # Save the debug image
                cv2.imwrite(modified_image_path, debug_image)
                # Save the processed points image
                cv2.imwrite(processed_points_image_path, processed_points_image)

                # Convert the processed image to a DataFrame
                df = pd.DataFrame(processed_points_image)
                # Save the DataFrame to a CSV file
                csv_path = os.path.join(modified_csv_folder, f"{filename}.csv")
                df.to_csv(csv_path, index=False, header=False)

                print(f"Processed {points_image_path}")
                
                # Write the results to the CSV file
                csv_writer.writerow([filename, optimal_angle, optimal_translation[0], optimal_translation[1]])


if __name__ == '__main__':
    main()