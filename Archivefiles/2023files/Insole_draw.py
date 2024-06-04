from PIL import Image
import os
import csv

# Directories
source_dir = 'D:\\A\\Designs\\CAD\\Insole_png'
csv_dir = 'D:\\A\\Designs\\CAD\\Insole_csv'

# Ensure the CSV directory exists
os.makedirs(csv_dir, exist_ok=True)

# Function to process each PNG file
def process_file(png_path, csv_path):
    # Open the PNG file
    img = Image.open(png_path)
    
    # Convert the image to grayscale ('L') to work with a single value for each pixel
    img = img.convert('L')
    
    # Get the size of the image
    width, height = img.size
    
    # Initialize an empty list to hold the CSV data
    csv_data = []
    
    # Iterate over each pixel in the image
    for y in range(height):
        row = []
        for x in range(width):
            # Get the pixel value (0 to 255)
            pixel_value = img.getpixel((x, y))
            
            # Convert pixel value to CSV format (0 for white, 255 for black)
            csv_value = 255 if pixel_value < 128 else 0
            row.append(csv_value)
        csv_data.append(row)
    
    # Write the CSV file
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)

# Process each PNG file in the source directory
for filename in os.listdir(source_dir):
    if filename.lower().endswith('.png'):
        base_name = os.path.splitext(filename)[0]
        png_path = os.path.join(source_dir, filename)
        csv_path = os.path.join(csv_dir, f'{base_name}.csv')
        process_file(png_path, csv_path)
