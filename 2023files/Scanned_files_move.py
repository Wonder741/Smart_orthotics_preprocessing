import tkinter as tk
from tkinter import filedialog, messagebox
import os
import shutil

last_selected_directory = None  # Global variable to remember the last selected directory
predefined_path = None  # Variable for predefined path

def add_folder():
    global last_selected_directory
    initial_dir = last_selected_directory or "/"  # Use the last directory, or start from root
    folder_path = filedialog.askdirectory(initialdir=initial_dir, title="Select Folder")
    if folder_path:
        if folder_path in folder_list.get(0, tk.END):
            messagebox.showwarning("Duplicate Folder", "This folder has already been added.")
        else:
            folder_list.insert(tk.END, folder_path)
            last_selected_directory = os.path.dirname(folder_path)  # Update the last directory

def set_destination_folder():
    global predefined_path
    initial_dir = predefined_path or "/"  # Use the last directory, or start from root
    predefined_path = filedialog.askdirectory(initialdir=initial_dir, title="Select Destination Folder")
    if predefined_path:
        destination_label.config(text=f"Destination: {predefined_path}")

def get_next_folder_number():
    existing_folders = [f for f in os.listdir(predefined_path) if os.path.isdir(os.path.join(predefined_path, f))]
    numbers = [int(folder) for folder in existing_folders if folder.isdigit()]
    return max(numbers) + 1 if numbers else 1

def copy_stl_files():
    if not predefined_path:
        messagebox.showerror("No Destination", "Please set a destination folder path before copying.")
        return

    feedback_list.delete(0, tk.END)  # Clear previous feedback
    folder_number = get_next_folder_number()

    idx = 0
    while idx < folder_list.size():
        folder_path = folder_list.get(idx)
        stl_files = [file for file in os.listdir(folder_path) if file.lower().endswith(".stl")]
        csv_files = [file for file in os.listdir(folder_path) if file.lower().endswith(".csv")]

        stl_feedback = process_files(folder_path, stl_files, folder_number, 'STL', 2)
        csv_feedback = process_files(folder_path, csv_files, folder_number, 'CSV', 1)

        update_feedback_list(stl_feedback)
        update_feedback_list(csv_feedback)

        folder_list.delete(idx)
        folder_number += 1

def process_files(folder_path, files, folder_number, file_type, expected_count):
    if len(files) == 0:
        return {'message': f"Error: No {file_type} files in {folder_path}", 'color': 'red'}
    elif len(files) == expected_count:
        copy_files(folder_path, files, folder_number)
        return {'message': f"Success: {expected_count} {file_type} file(s) copied from {folder_path}", 'color': 'blue'}
    else:
        return {'message': f"Warning: {len(files)} {file_type} files in {folder_path}", 'color': 'orange'}

def copy_files(folder_path, files, folder_number):
    dest_folder = os.path.join(predefined_path, f'{folder_number:04d}')
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    for file in files:
        shutil.copy2(os.path.join(folder_path, file), os.path.join(dest_folder, file))

def update_feedback_list(feedback):
    feedback_list.insert(tk.END, feedback['message'])
    feedback_list.itemconfig(tk.END, {'fg': feedback['color']})

# GUI setup
app = tk.Tk()
app.title("STL and CSV File Organizer")

# Folder List Display
folder_list = tk.Listbox(app, width=100)
folder_list.pack(padx=10, pady=10)

# Buttons Frame
buttons_frame = tk.Frame(app)
buttons_frame.pack(padx=10, pady=10)

# Set Destination Button
destination_button = tk.Button(buttons_frame, text="Set Destination", command=set_destination_folder)
destination_button.pack(side=tk.LEFT, padx=5)

# Add Folder Button
add_button = tk.Button(buttons_frame, text="Add Folder", command=add_folder)
add_button.pack(side=tk.LEFT, padx=5)

# Copy Button
copy_button = tk.Button(buttons_frame, text="Copy STL and CSV Files", command=copy_stl_files)
copy_button.pack(side=tk.LEFT, padx=5)

# Destination Path Label
destination_label = tk.Label(app, text="Destination: Not Set", fg="red")
destination_label.pack(padx=10, pady=5)

# Feedback List Display
feedback_list = tk.Listbox(app, width=100)
feedback_list.pack(padx=10, pady=10)

app.mainloop()
