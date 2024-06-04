import cv2
import numpy as np

def load_and_preprocess_images(object_image_path, points_image_path):
    # Load images
    object_image = cv2.imread(object_image_path, cv2.IMREAD_GRAYSCALE)
    points_image = cv2.imread(points_image_path, cv2.IMREAD_GRAYSCALE)

    # Preprocess object image
    object_edges = cv2.Canny(object_image, 100, 200)
    object_mask = cv2.threshold(object_image, 1, 255, cv2.THRESH_BINARY)[1]
    object_mask1 = cv2.threshold(object_edges, 1, 255, cv2.THRESH_BINARY)[1]

    # Preprocess points image
    points_mask = cv2.threshold(points_image, 1, 255, cv2.THRESH_BINARY)[1]

    return object_mask, points_mask, object_mask1

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

def test_rotation(object_mask, aligned_points_mask, rotation_center):
    rows, cols = aligned_points_mask.shape
    overlap_before = 0
    best_angle = 0
    optimal_rotate_image = None

    for angle in range(0, 360, 1):  # Rotate from 0 to 180 degrees in steps of 10
        M = cv2.getRotationMatrix2D(rotation_center, angle, 1)
        rotated_points = cv2.warpAffine(aligned_points_mask, M, (cols, rows))
        overlap = np.sum((object_mask & rotated_points) > 0)
        if overlap_before < overlap:
            overlap_before = overlap
            best_angle = angle
            optimal_rotate_image = rotated_points
    return optimal_rotate_image, overlap_before, best_angle

def test_bias(object_mask, points_mask, bias_range=20):
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
    object_image_path = 'D://A//1 InsoleDataset//AutoCAD//PNG_L//9ol-Model.png'
    points_image_path = 'D://A//1 InsoleDataset//WMT//Averaged//LeftPNG//001//001_01_Data_left.png'
    modified_image_path = 'D://A//1 InsoleDataset//WMT//Modified//modified.png'

    object_mask, points_mask, object_mask1 = load_and_preprocess_images(object_image_path, points_image_path)
    aligned_points_mask, translation, rotation_center= align_centers(points_mask)

    optimal_rotate_image, overlap_before, best_angle = test_rotation(object_mask, aligned_points_mask, rotation_center)
    print("Testing rotation angles:")
    print(overlap_before, best_angle)


    print(f"Translation: {translation}")

    print(f"Optimal Rotation Angle: {best_angle} degrees")

    optimal_moved_image, max_overlap, optimal_bias = test_bias(object_mask, optimal_rotate_image)
    print(f"Translation1: {optimal_bias}")

    # Overlay the object mask and the modified points mask for debugging
    debug_image = cv2.cvtColor(object_mask1, cv2.COLOR_GRAY2BGR)  # Convert to BGR for color
    debug_image[:, :, 1] = np.maximum(debug_image[:, :, 1], optimal_moved_image)  # Overlay in green channel

    # Save or display the debug image
    cv2.imwrite(modified_image_path, debug_image)
    cv2.imshow('Debug Image', debug_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

if __name__ == '__main__':
    main()