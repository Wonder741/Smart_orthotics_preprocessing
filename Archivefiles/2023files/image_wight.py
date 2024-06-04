import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def process_file(file_path, output_folder):
    # Read the CSV file
    data = pd.read_csv(file_path, header=None)

    # Remove the first row and first column
    modified_data = data.iloc[1:, 1:]

    # Sum up the values of all cells
    total_sum = modified_data.to_numpy().astype(float).sum()
    print(f"Total sum of cells in {os.path.basename(file_path)}: {total_sum}")

    # Convert data to a numpy array for plotting
    data_array = modified_data.to_numpy()

    # Generate a false color map image
    plt.figure(figsize=(10, 8))
    plt.imshow(data_array, cmap='viridis')
    plt.colorbar(label='Scale')
    plt.title(f"Data Visualization: {os.path.basename(file_path)}")

    # Save the image to the output folder
    output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(file_path))[0] + '.png')
    plt.savefig(output_file)
    plt.close()

def main():
    input_folder = 'D:\\A\\Data\\Experiment data\\Statics\\D'
    output_folder = 'D:\\A\\Data\\Experiment data\\Statics\\E'

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
