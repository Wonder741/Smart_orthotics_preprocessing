import cv2
import numpy as np
import os
import csv

# Use the minimum frame method to locate the point image center, and move the center to the image center (320, 320)
def align_centers(points_mask):
    # Find contours in the points mask
    contours, _ = cv2.findContours(points_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the bounding rectangle for all contours
    x, y, w, h = cv2.boundingRect(np.vstack(contours))

    # Calculate the center of the bounding rectangle
    frame_center = (int(x + w / 2), int(y + h / 2))

    # Calculate the center of the object mask
    object_center = (320, 320)

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

# Align the roated point image in a short range x(-20 ,20) and y(-20, 20) to further optimal the image alignment
def test_bias(object_mask, points_mask, bias_range=40):
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
    size_index = 'PNG_L//8ol-Model.png'
    folder_index = 'LeftPNG//009//'
    object_image_path = 'D://A//1 InsoleDataset//AutoCAD//' + size_index
    points_image_folder = 'D://A//1 InsoleDataset//WMT//Averaged//' + folder_index
    modified_image_folder = 'D://A//1 InsoleDataset//WMT//Modified//' + folder_index

    # Ensure the modified image folder exists
    if not os.path.exists(modified_image_folder):
        os.makedirs(modified_image_folder)
    
    # CSV file path
    csv_file_path = os.path.join(modified_image_folder, 'results.csv')

    # Load the object mask
    object_image = cv2.imread(object_image_path, cv2.IMREAD_GRAYSCALE)
    object_edges = cv2.Canny(object_image, 100, 200)
    object_mask = cv2.threshold(object_image, 1, 255, cv2.THRESH_BINARY)[1]
    object_mask1 = cv2.threshold(object_edges, 1, 255, cv2.THRESH_BINARY)[1]

    # Open the CSV file for writing
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Image File Name', 'Optimal Angle', 'dx', 'dy'])

        # Process each points image in the folder
        for filename in os.listdir(points_image_folder):
            if filename.endswith('.png'):
                points_image_path = os.path.join(points_image_folder, filename)
                modified_image_path = os.path.join(modified_image_folder, 'modified_' + filename)

                # Load and preprocess the points image
                points_image = cv2.imread(points_image_path, cv2.IMREAD_GRAYSCALE)
                points_mask = cv2.threshold(points_image, 1, 255, cv2.THRESH_BINARY)[1]

                # Align centers, rotate, and find optimal bias
                aligned_points_mask, translation, rotation_center = align_centers(points_mask)
                optimal_rotate_image, max_overlap, best_angle_1 = test_rotation(object_mask, aligned_points_mask, rotation_center)
                optimal_moved_image, max_overlap, optimal_bias_1 = test_bias(object_mask, optimal_rotate_image)

                optimal_rotate_image, max_overlap, best_angle_2 = test_rotation(object_mask, optimal_moved_image, rotation_center)
                optimal_moved_image, max_overlap, optimal_bias_2 = test_bias(object_mask, optimal_rotate_image)


                # Overlay the object mask and the modified points mask for debugging
                debug_image = cv2.cvtColor(object_mask1, cv2.COLOR_GRAY2BGR)
                debug_image[:, :, 1] = np.maximum(debug_image[:, :, 1], optimal_moved_image)

                # Save the debug image
                cv2.imwrite(modified_image_path, debug_image)

                print(f"Processed {points_image_path}, saved to {modified_image_path}")

                if 181 <= best_angle_1 <= 359:
                    best_angle_1 = best_angle_1 - 360
                if 181 <= best_angle_2 <= 359:
                    best_angle_2 = best_angle_2 - 360

                optimal_angle = best_angle_1 + best_angle_2
                optimal_translation = translation + optimal_bias_1 + optimal_bias_2
                print(max_overlap)
                
                # Write the results to the CSV file
                csv_writer.writerow([filename, optimal_angle, optimal_translation[0], optimal_translation[1]])


if __name__ == '__main__':
    main()