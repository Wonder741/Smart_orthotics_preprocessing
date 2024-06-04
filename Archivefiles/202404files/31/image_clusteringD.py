import numpy as np
from skimage import io, filters
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the grayscale image and the mask
image_path = "D://A//1 InsoleDataset//WMT//Processed_Points//LeftPNG//001//processed_001_02_Data_left.png"
mask_path = "D://A//1 InsoleDataset//AutoCAD//PNG_L//12ol-Model.png"
image = io.imread(image_path, as_gray=True)
mask = io.imread(mask_path, as_gray=True)

# Compute the gradient of the image using Sobel filters
gradient = np.sqrt(filters.sobel_h(image)**2 + filters.sobel_v(image)**2)

# Find indices of points where the mask is white (assuming white is close to 255)
mask_indices = np.where(mask > 0)  # Adjust the threshold as needed
# Find indices of non-black (non-zero) pixels
non_black_indices = np.where(image > 0)

# Combine pixel values, their coordinates, and gradient for clustering
features = np.column_stack((mask_indices[0], mask_indices[1], image[mask_indices], gradient[mask_indices]))

# Feature scaling and weighting
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
weights = np.array([1.5, 20, 5, 2])  # Adjust the weights as needed
weighted_features = scaled_features * weights

# Perform K-means clustering with 4 clusters on weighted features
kmeans = KMeans(n_clusters=4, n_init=10, random_state=1)
kmeans.fit(weighted_features)

# Initialize a label array with -1 for all points
label_array = -1 * np.ones(image.shape, dtype=int)

# Assign the cluster labels to the corresponding positions in the label array
for i, (row, col) in enumerate(zip(mask_indices[0], mask_indices[1])):
    label_array[row, col] = kmeans.labels_[i]

# Create a colored image with the same shape as the original but with 3 channels for RGB
colored_image = np.zeros((*image.shape, 3), dtype=np.uint8)

# Map each cluster to a different color and apply it to the non-black areas of the image
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Red, Green, Blue, Yellow
for i in range(4):
    cluster_indices = non_black_indices[0][label_array[non_black_indices] == i], non_black_indices[1][label_array[non_black_indices] == i]
    colored_image[cluster_indices] = colors[i]

# Display the original and colored clustered images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(colored_image)
plt.title('Clustered Image with Centroids')
plt.axis('off')

# Mark the centroids on the clustered image
for centroid in kmeans.cluster_centers_:
    # Inverse transform to get the original scale for centroids
    original_centroid = scaler.inverse_transform(centroid.reshape(1, -1) / weights)
    plt.scatter(original_centroid[0][1], original_centroid[0][0], color='white', marker='x', s=100)

plt.show()
