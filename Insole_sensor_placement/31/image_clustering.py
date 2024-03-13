import cv2
import numpy as np
import os
import csv
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances_argmin_min

def extract_features(image):
    feature_vectors = []

    # Compute gradients using Sobel operator
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute gradient magnitude and direction (in degrees)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_direction = np.arctan2(sobel_y, sobel_x) * (180 / np.pi)

    # Iterate over each pixel and extract features
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            intensity = image[y, x]
            grad_mag = gradient_magnitude[y, x]
            grad_dir = gradient_direction[y, x]
            feature_vectors.append([intensity, x, y, grad_mag, grad_dir])

    # Convert the list of feature vectors to a NumPy array
    feature_matrix = np.array(feature_vectors)
    return feature_matrix

def select_representative_points(cluster_labels, feature_matrix):
    unique_labels = set(cluster_labels)
    centroids = []
    average_intensities = []
    
    for label in unique_labels:
        if label != -1:  # Ignore noise points
            points_in_cluster = feature_matrix[cluster_labels == label]
            centroid = np.mean(points_in_cluster[:, 1:3], axis=0)  # Calculate centroid using x, y coordinates
            average_intensity = np.mean(points_in_cluster[:, 0])  # Calculate average intensity
            centroids.append(centroid)
            average_intensities.append(average_intensity)
            
    print(len(centroids))
    # If there are more than 16 clusters, select the 16 centroids of clusters with the largest average intensity
    if len(centroids) > 16:

        indices = np.argsort(average_intensities)[::-1][:16]  # Get indices of top 16 clusters based on intensity
        selected_centroids = np.array(centroids)[indices]
    else:
        selected_centroids = np.array(centroids)
    
    return selected_centroids

from scipy.spatial.distance import cdist

def merge_close_clusters(centroids, cluster_labels, threshold=20):
    # Calculate pairwise distances between centroids
    distances = cdist(centroids, centroids)

    # Find pairs of centroids closer than the threshold
    close_pairs = np.where((distances < threshold) & (distances > 0))

    # Create a mapping from old cluster labels to new cluster labels
    label_mapping = {label: label for label in set(cluster_labels)}

    for i, j in zip(close_pairs[0], close_pairs[1]):
        if i < j:  # Avoid double counting pairs
            # Merge clusters by updating the label mapping
            smaller_label = min(label_mapping[i], label_mapping[j])
            label_mapping[i] = smaller_label
            label_mapping[j] = smaller_label

    # Update cluster labels based on the mapping
    new_labels = [label_mapping[label] for label in cluster_labels]

    return new_labels



def process_images(input_folder, output_folder, csv_filename):
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Image', 'Point', 'X', 'Y'])  # Header
        
        for filename in os.listdir(input_folder):
            if filename.endswith('.png'):
                image_path = os.path.join(input_folder, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                # Extract features and apply DBSCAN
                feature_matrix = extract_features(image)
                dbscan = DBSCAN(eps=30, min_samples=50)  # Adjust parameters
                cluster_labels = dbscan.fit_predict(feature_matrix)
                unique_labels = set(cluster_labels)
                centroids = []
                for label in unique_labels:
                    if label != -1:  # Ignore noise points
                        points_in_cluster = feature_matrix[cluster_labels == label]
                        centroid = np.mean(points_in_cluster[:, 1:3], axis=0)  # Calculate centroid using x, y coordinates
                        average_intensity = np.mean(points_in_cluster[:, 0])  # Calculate average intensity
                        centroids.append(centroid)
                # Merge close clusters and update labels
                cluster_labels = merge_close_clusters(centroids, cluster_labels, threshold=20)
                
                # Select 16 representative points
                points = select_representative_points(cluster_labels, feature_matrix)
                
                # Overlay points on the original image and save
                for idx, point in enumerate(points):
                    cv2.circle(image, (int(point[0]), int(point[1])), radius=5, color=(255, 255, 255), thickness=-1)
                    csvwriter.writerow([filename, idx] + point.tolist())

                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, image)

# Example usage
folder_index = 'LeftPNG//001//'
input_folder = 'D://A//1 InsoleDataset//WMT//Processed_Points//' + folder_index
output_folder = 'D://A//1 InsoleDataset//WMT//Clustering//' + folder_index
csv_filename = output_folder + 'coordinates.csv'
# Ensure the processed points image folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
process_images(input_folder, output_folder, csv_filename)
