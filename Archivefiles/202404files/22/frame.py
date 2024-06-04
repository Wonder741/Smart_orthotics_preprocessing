import cv2
import numpy as np

# Load the image
image_path = 'D://A//1 InsoleDataset//WMT//Averaged//LeftPNG//001//001_11_Data_left.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Threshold the image to create a binary mask
_, binary_mask = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

# Find contours in the binary mask
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the bounding rectangle for all contours
x, y, w, h = cv2.boundingRect(np.vstack(contours))

# Calculate the centers
image_center = (image.shape[1] // 2, image.shape[0] // 2)
frame_center = (x + w // 2, y + h // 2)

# Draw the bounding rectangle and centers on the original image
result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for color drawing
cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.circle(result_image, image_center, 5, (255, 0, 0), -1)  # Image center in blue
cv2.circle(result_image, frame_center, 5, (0, 0, 255), -1)  # Frame center in red

# Display the result
cv2.imshow('Minimum Frame with Centers', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
