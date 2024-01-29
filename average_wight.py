import os
import pandas as pd
import numpy as np

def process_file(file_path, output_folder):
    # Read the CSV file, skipping the first row
    data = pd.read_csv(file_path, header=None, skiprows=1)

    # Convert all data to numeric, setting errors='coerce' to replace non-numeric values with NaN
    data = data.apply(pd.to_numeric, errors='coerce')

    # Split data into frames (every 98 lines)
    frames = [data.iloc[i:i+98] for i in range(0, len(data), 98)]

    # Filter out frames that do not have the expected number of rows and columns (98 rows x 64 columns)
    filtered_frames = [frame for frame in frames if frame.shape == (98, 65)]

    # Calculate average across all filtered frames
    if filtered_frames:
        average_data = np.nanmean([frame.values for frame in filtered_frames], axis=0)

        # Save the averaged data to a new CSV file
        output_file = os.path.join(output_folder, os.path.basename(file_path))
        pd.DataFrame(average_data).to_csv(output_file, header=False, index=False)
    else:
        print(f"No valid frames found in {file_path}")

def main():
    input_folder = 'D:\\A\\Data\\Experiment data\\Statics\\C'
    output_folder = 'D:\\A\\Data\\Experiment data\\Statics\\D'

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each CSV file in the input folder
    for file in os.listdir(input_folder):
        if file.endswith('.csv'):
            file_path = os.path.join(input_folder, file)
            process_file(file_path, output_folder)

    print("Processing complete.")

if __name__ == "__main__":
    main()
