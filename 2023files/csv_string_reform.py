import os
import csv

def process_csv_files(source_folder, destination_folder):
    """
    Process all CSV files in the source folder that contain "Data" in their name 
    and save them in the destination folder.
    """
    for filename in os.listdir(source_folder):
        if "Data" in filename and filename.endswith('.csv'):
            process_file(os.path.join(source_folder, filename), destination_folder)

def process_file(file_path, destination_folder):
    """
    Process a single CSV file: delete the first 23 lines, extract and process strings, 
    and save the edited file in the destination folder.
    """
    data = read_and_trim_csv(file_path)
    processed_data = [process_string(row[0]) for row in data]
    processed_data1 = chunk_and_skip(processed_data, 98)
    save_processed_data(processed_data1, file_path, destination_folder)

def chunk_and_skip(data, chunk_size):
    return [item for i, block in enumerate([data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]) for j, item in enumerate(block) if j != 0]

def read_and_trim_csv(file_path):
    """
    Read a CSV file and delete the first 23 lines.
    """
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        # Skip first 23 lines
        for _ in range(23):
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

def save_processed_data(data, original_file_path, destination_folder):
    """
    Save the processed data into a new CSV file in the destination folder.
    """
    new_file_path = os.path.join(destination_folder, os.path.basename(original_file_path))
    with open(new_file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)


# Paths
input_dir = 'D:\\A\\Data\\Experiment data\\0123\\Converted\\New folder\\Converted data'
output_dir = 'D:\\A\\Data\\Experiment data\\0123\\Processed'

# Create output directories if they don't exist
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Example usage
process_csv_files(input_dir, output_dir)


