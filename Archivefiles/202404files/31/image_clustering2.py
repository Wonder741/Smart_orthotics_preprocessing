import cv2
import os
import numpy as np

def process_masks(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize an empty array to store the combined mask
    combined_mask = None

    # Iterate over the images in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            # Read the image as a grayscale image
            img = cv2.imread(os.path.join(input_folder, filename), cv2.IMREAD_GRAYSCALE)

            # Initialize the combined mask with the first image
            if combined_mask is None:
                combined_mask = img
            else:
                # Combine the current mask with the previous masks using the "OR" operation
                combined_mask = cv2.bitwise_or(combined_mask, img)

    # Find the largest contour (area) in the combined mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a blank image with the same dimensions as the mask
    output_img = np.zeros_like(combined_mask)

    # Draw the largest contour on the blank image
    cv2.drawContours(output_img, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    # Save the result to the output folder
    cv2.imwrite(os.path.join(output_folder, output_file_name), output_img)

    print('Process completed. The result has been saved to', output_folder)

if __name__ == '__main__':
    output_file_name = 'combined_mask4.png'
    input_folder = 'D://A//1 InsoleDataset//WMT//Kmean_Mask//LeftPNG//001//sorted_cluster_4'
    output_folder = 'D://A//1 InsoleDataset//WMT//Kmean_Mask//LeftPNG//001'
    process_masks(input_folder, output_folder)
