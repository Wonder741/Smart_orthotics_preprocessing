import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import csv
import math

def calculate_angle(point1, point2, coord_weight_x, coord_weight_y):
    # Calculate the angle between the line connecting point1 and point2 and the horizontal axis
    dx = point2[0] / coord_weight_x - point1[0]
    dy = point2[1] / coord_weight_y - point1[1]
    angle = math.atan2(dy, dx) * 180 / math.pi
    return angle if angle >= 0 else 360 + angle

def apply_mask_and_cluster(input_folder, mask_path, output_csv, output_images_folder, coord_weight_x, coord_weight_y, value_weight):
    # Read the mask image as a grayscale image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Calculate the centroid of the mask
    mask_points = np.column_stack(np.where(mask > 0))
    mask_centroid = np.mean(mask_points, axis=0)
    mask_centroid = [mask_centroid[1], mask_centroid[0]]

    # Open the CSV file for writing
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['file name', 'point 1 x', 'point 1 y', 'point 2 x', 'point 2 y', 'point 3 x', 'point 3 y', 'point 4 x', 'point 4 y', 'point 5 x', 'point 5 y'])

        # Iterate over the images in the input folder
        for filename in os.listdir(input_folder):
            if filename.endswith('.png'):
                # Read the image as a grayscale image
                img = cv2.imread(os.path.join(input_folder, filename), cv2.IMREAD_GRAYSCALE)

                # Apply the mask to the image
                filtered_img = cv2.bitwise_and(img, img, mask=mask)

                # Find the non-black points and their corresponding pixel values
                y_coords, x_coords = np.where(filtered_img>0)
                pixel_values = filtered_img[y_coords, x_coords]
                features = np.column_stack((x_coords * coord_weight_x, y_coords * coord_weight_y, pixel_values * value_weight))
            
                # Specify the initial centroids
                #init_centroids = np.array([[47, 297, 0], [143, 297, 0], [47, 367, 0], [143, 367, 0]])
                init_centroids = np.array([[319, 384, 0], [319, 384, 0], [319, 384, 0], [319, 384, 0], [319, 384, 0]])

                # Perform k-means clustering with 4 clusters and the specified initial centroids
                kmeans = KMeans(n_clusters=5, init=init_centroids, n_init=1, random_state=0).fit(features)
                cluster_centers = kmeans.cluster_centers_

                # Calculate angles and sort clusters counter-clockwise
                angles = [calculate_angle(mask_centroid, center, coord_weight_x, coord_weight_y) for center in cluster_centers]
                print(angles)
                sorted_indices = np.argsort(angles) 
                sorted_centers = cluster_centers[sorted_indices]

                """ # Mark the centroid points on the masked grayscale image
                for center in sorted_centers:
                    cv2.circle(filtered_img, (int(center[0] / coord_weight), int(center[1] / coord_weight)), 2, (0, 0, 0), -1) """
                # Define a list of colors for marking the centroids BGRK
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0), (0, 255, 255), (255, 0, 255)]

                # Convert the grayscale image to BGR (3-channel) format to allow color drawing
                filtered_img_bgr = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)

                # Mark the centroid points on the masked BGR image with different colors
                for idx, center in enumerate(sorted_centers):
                    cv2.circle(filtered_img_bgr, (int(center[0] / coord_weight_x), int(center[1] / coord_weight_y)), 2, colors[idx], -1)

                cv2.circle(filtered_img_bgr, (int(mask_centroid[0]), int(mask_centroid[1])), 2, colors[5], -1)
                # Save the marked image to the output images folder
                cv2.imwrite(os.path.join(output_images_folder, filename), filtered_img_bgr)

                # Write the file name and sorted cluster coordinates to the CSV file
                writer.writerow([filename] + [coord for center in sorted_centers for coord in [center[0] / coord_weight_x, center[1] / coord_weight_y]])

    print('Process completed. The results have been saved to', output_csv)

if __name__ == '__main__':
    input_folder = 'D://A//1 InsoleDataset//WMT//Processed_Points//LeftPNG//001//'
    output_images_folder = 'D://A//1 InsoleDataset//WMT//Kmean_Mask//LeftPNG//001//sorted_keypoint_3//'
    mask_path = 'D://A//1 InsoleDataset//WMT//Kmean_Mask//LeftPNG//001//combined_mask3.png'
    output_csv = 'D://A//1 InsoleDataset//WMT//Kmean_Mask//LeftPNG//001//output3.csv'
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_images_folder):
        os.makedirs(output_images_folder)
    coord_weight_x = 2  # Adjust the coordinate weight here
    coord_weight_y = 1  # Adjust the coordinate weight here
    value_weight = 0.5  # Adjust the value weight here
    apply_mask_and_cluster(input_folder, mask_path, output_csv, output_images_folder, coord_weight_x, coord_weight_y, value_weight)
