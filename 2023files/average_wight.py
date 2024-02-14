import pandas as pd
import numpy as np
from PIL import Image
import os

def process_csv_file(file_path):
    df = pd.read_csv(file_path, header=None, dtype=str)
    # Change non-numeric values to 0 and convert DataFrame to numeric
    df = df.applymap(lambda x: 0 if not x.isdigit() else int(x))
    frames = []
    for start in range(0, len(df), 97):
        if start + 97 <= len(df):  # Ensure we have a full frame
            frame = df.iloc[start+1:start+97, 1:65].reset_index(drop=True)
            if len(frame) == 96:
                frames.append(frame.to_numpy())
    return frames

def calculate_average(frames):
    if frames:
        return np.mean(frames, axis=0)
    else:
        return np.array([])  # Return an empty array if no frames

def save_to_csv_and_png(frame, base_name, csv_dir, png_dir):
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"{base_name}.csv")
    png_path = os.path.join(png_dir, f"{base_name}.png")
    pd.DataFrame(frame).to_csv(csv_path, index=False, header=False)
    norm_frame = np.interp(frame, (frame.min(), frame.max()), (0, 255)).astype(np.uint8)
    img = Image.fromarray(norm_frame)
    img = img.resize((64, 48), Image.Resampling.LANCZOS)
    img.save(png_path)

def process_all_files(source_dir, left_csv_dir, right_csv_dir, left_png_dir, right_png_dir):
    for filename in os.listdir(source_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(source_dir, filename)
            base_name = os.path.splitext(filename)[0]
            frames = process_csv_file(file_path)
            avg_frame = calculate_average(frames)
            if avg_frame.size > 0:
                left_frame = avg_frame[:48, :]
                right_frame = avg_frame[48:, :]
                save_to_csv_and_png(left_frame, f"{base_name}_left", left_csv_dir, left_png_dir)
                save_to_csv_and_png(right_frame, f"{base_name}_right", right_csv_dir, right_png_dir)

# Define your directories
source_dir = 'D:\\A\\Data\\Experiment data\\0123\\Processed'
left_csv_dir = 'D:\\A\\Data\\Experiment data\\0123\\Left_average'
right_csv_dir = 'D:\\A\\Data\\Experiment data\\0123\\Right_average'
left_png_dir = 'D:\\A\\Data\\Experiment data\\0123\\Left_png'
right_png_dir = 'D:\\A\\Data\\Experiment data\\0123\\Right_png'

# Process all CSV files individually and save results
process_all_files(source_dir, left_csv_dir, right_csv_dir, left_png_dir, right_png_dir)
