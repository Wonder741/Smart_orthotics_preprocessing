import numpy as np
from skimage import io, filters
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Load the grayscale image and the mask
output_path = "D://A//1 InsoleDataset//WMT//Kmean_Mask//LeftPNG//001//"
image_path = "D://A//1 InsoleDataset//WMT//Processed_Points//LeftPNG//001//"
image_files = sorted(os.listdir(image_path))
mask_path = "D://A//1 InsoleDataset//AutoCAD//PNG_L//12ol-Model.png"
mask = io.imread(mask_path, as_gray=True)

# Output directories for sorted cluster masks
output_dirs = ["sorted_cluster_1", "sorted_cluster_2", "sorted_cluster_3", "sorted_cluster_4"]
for output_dir in output_dirs:
    os.makedirs(output_path + output_dir, exist_ok=True)

# Iterate through the images in the folder
for image_file in image_files:
    image = io.imread(os.path.join(image_path, image_file), as_gray=True)

    # Compute the gradient of the image using Sobel filters
    gradient = np.sqrt(filters.sobel_h(image)**2 + filters.sobel_v(image)**2)

    # Find indices of points where the mask is white (assuming white is close to 255)
    mask_indices = np.where(mask > 0)  # Adjust the threshold as needed

    # Combine pixel values, their coordinates, and gradient for clustering
    features = np.column_stack((mask_indices[0], mask_indices[1], image[mask_indices], gradient[mask_indices]))

    # Feature scaling and weighting
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    weights = np.array([1, 15, 5, 1])  # Adjust the weights as needed
    weighted_features = scaled_features * weights

    # Perform K-means clustering with 4 clusters on weighted features
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=1)
    kmeans.fit(weighted_features)

    # Initialize a label array with -1 for all points
    label_array = -1 * np.ones(image.shape, dtype=int)

    # Assign the cluster labels to the corresponding positions in the label array
    for i, (row, col) in enumerate(zip(mask_indices[0], mask_indices[1])):
        label_array[row, col] = kmeans.labels_[i]

    # Sort the clusters based on their centroid x-coordinates
    sorted_cluster_indices = np.argsort(kmeans.cluster_centers_[:, 1])

    # Save the sorted cluster masks as separate images
    for i, cluster_id in enumerate(sorted_cluster_indices):
        mask_image = np.zeros(image.shape, dtype=np.uint8)
        mask_image[label_array == cluster_id] = 255
        output_file = os.path.join(output_path, output_dirs[i], f"sorted_mask_{image_file}")
        io.imsave(output_file, mask_image)

    print(f"Processed and saved sorted masks for {image_file}")
