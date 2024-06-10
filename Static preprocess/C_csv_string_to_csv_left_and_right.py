import os
import csv
import pandas as pd
import numpy as np
from PIL import Image

index = '33anson'

def append_to_csv(csv_path, file_index, max_z, mean_z):
    try:
        # Convert max_z to an integer
        max_z_integer = int(max_z)
        mean_z_integer = int(mean_z)
        
        # Create a single row with the file name and max_z_integer
        csv_data = [os.path.basename(file_index), max_z_integer, mean_z_integer]
        
        # Append the row to the CSV file
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_data)
        
        #print(f"Data save successfully to {csv_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def read_and_trim_csv(file_path, line_number):
    """
    Read a CSV file and delete the first line_number lines.
    """
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        # Skip first several lines
        for _ in range(line_number):
            next(reader, None)
        data = list(reader)
    return data

def process_string(string):
    """
    Process the string: split it into non-space character sequences and return as a list.
    """
    temp_list = []
    word = ''
    for char in string:
        if char != ' ':
            word += char
        else:
            if word:
                temp_list.append(word)
                word = ''
    if word:
        temp_list.append(word)
    return temp_list

def chunk_and_skip(data, chunk_size):
    return [item for i, block in enumerate([data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]) for j, item in enumerate(block) if j != 1]

""" def save_processed_data(data, original_file_path, destination_folder):
    #Save the processed data into a new CSV file in the destination folder.
    new_file_path = os.path.join(destination_folder, os.path.basename(original_file_path))
    with open(new_file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row) """

def save_processed_data(data, original_file_path, destination_folder):
    """
    Save the processed data into a new CSV file in the destination folder.
    """
    new_file_path = os.path.join(destination_folder, os.path.basename(original_file_path))
    with open(new_file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)

# Function to check if a string is numeric and convert
def convert_if_numeric(s):
    try:
        return int(s)  # Try converting to integer
    except ValueError:
        return np.nan  # If not numeric, return NaN

def frame_data(csv_data):
    df = pd.DataFrame(csv_data)
    # Change non-numeric values to 0 and convert DataFrame to numeric
    frames = []
    for start in range(0, len(df), 97):
        if start + 97 <= len(df):  # Ensure we have a full frame
            frame = df.iloc[start+1:start+97, 1:65].reset_index(drop=True)
            if len(frame) == 96:
                frames.append(frame.to_numpy())
    return frames

def calculate_average(frames):
    if frames:
        mean_frame = np.mean(frames, axis=0).astype(int)
        max_value = mean_frame.max()
        non_zero_elements = mean_frame[mean_frame != 0]
        if non_zero_elements.size > 0:
            non_zero_mean_value = non_zero_elements.mean()
        else:
            non_zero_mean_value = 0  # Handle case where there are no non-zero elements
        return mean_frame, max_value, non_zero_mean_value
    else:
        return np.array([]), None, None  # Return an empty array if no frames
    
def save_to_csv_and_png(frame, base_name, csv_dir, png_dir):
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"{base_name}.csv")
    png_path = os.path.join(png_dir, f"{base_name}.png")
    pd.DataFrame(frame).to_csv(csv_path, index=False, header=False)
    norm_frame = np.interp(frame, (0, 512), (0, 255)).astype(np.uint8)
    img = Image.fromarray(norm_frame)
    img.save(png_path)

# Main function
def main(index):
    """
    Process all CSV files in the source folder that contain "Data" in their name 
    and save them in the destination folder.
    """
    # Static=23 or dynamic=30
    line_skip = 23

    # Paths
    target_folder = 'D:\\A\\A_Process_data\\WATMat'
    csv_file_name = "pressure_process_result.csv"
    csv_result = os.path.join(target_folder, csv_file_name)
    source_folder = os.path.join(target_folder, "1Raw", index)
    #temp_dir = os.path.join(target_folder, "1Temp", index)
    left_csv_dir = os.path.join(target_folder, "2Average", "csv_left", index)
    right_csv_dir = os.path.join(target_folder, "2Average", "csv_right", index)
    left_png_dir = os.path.join(target_folder, "2Average", "png_left", index)
    right_png_dir = os.path.join(target_folder, "2Average", "png_right", index)

    # Create output directories if they don't exist
    os.makedirs(source_folder, exist_ok=True)
    #os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(left_csv_dir, exist_ok=True)
    os.makedirs(right_csv_dir, exist_ok=True)
    os.makedirs(left_png_dir, exist_ok=True)
    os.makedirs(right_png_dir, exist_ok=True)

    frame_width = 98
    frame_height = 64
    file_number = 1
    for filename in os.listdir(source_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(source_folder, filename)
            basename = index + f'_{file_number}'
            data = read_and_trim_csv(file_path, line_skip)
            data_first_column = [process_string(row[0]) for row in data]
            data_skip_line = chunk_and_skip(data_first_column, frame_width)
            # Applying the conversion to each element in each row of data_skip_line
            cleaned_data = [[convert_if_numeric(item) for item in row] for row in data_skip_line]
            frames = frame_data(cleaned_data)
            avg_frame, max_value, mean_value = calculate_average(frames)
            if avg_frame.size > 0:
                left_frame = avg_frame[:48, :]
                right_frame = avg_frame[48:, :]
                left_frame = np.vstack((left_frame, np.zeros((16, left_frame.shape[1]))))
                right_frame = np.vstack((np.zeros((16, right_frame.shape[1])), right_frame))
                append_to_csv(csv_result, f"left_{basename}", max_value, mean_value)
                append_to_csv(csv_result, f"right_{basename}", max_value, mean_value)
                save_to_csv_and_png(left_frame, f"left_{basename}", left_csv_dir, left_png_dir)
                save_to_csv_and_png(right_frame, f"right_{basename}", right_csv_dir, right_png_dir)
            file_number += 1


if __name__ == '__main__':
    main(index)
    print("C process complete")