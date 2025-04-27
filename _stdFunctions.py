import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
def compute_std(video_folder, output_folder, roi, timeStart=None, timeEnd=None):
    """
    Computes and saves per-pixel standard deviation images for each video in a folder,
    after local mean subtraction within a temporal window.

    Input:
        video_folder (str): Path to the folder containing .mp4 video files.
        output_folder (str): Path to the folder where output STD images will be saved.
        roi (tuple): Region of interest as (x, y, w, h) for cropping frames.
        timeStart (float or None): Start time in seconds for frame selection (inclusive). If None, starts from beginning.
        timeEnd (float or None): End time in seconds for frame selection (exclusive). If None, goes to end.

    Output:
        std_dict (dict): Dictionary mapping video file names (without '_video.mp4' if present)
                         to a tuple (mean_std, std_img), where:
                         - mean_std (float): Mean value of the STD image.
                         - std_img (np.ndarray): The per-pixel STD image (2D array).
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    # Get list of all mp4 video files in the input folder
    video_files = [f for f in os.listdir(video_folder) if f.lower().endswith('.mp4')]
    x, y, w, h = roi
    window_size = 5  # Define the size of the local window for mean subtraction
    std_dict = {}
    for file in tqdm(video_files):
        video_path = os.path.join(video_folder, file)
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        frames = []
        # Get video properties for time/frame conversion
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Compute frame indices for timeStart and timeEnd
        if timeStart is not None:
            start_frame = int(timeStart * fps)
        else:
            start_frame = 0
        if timeEnd is not None:
            end_frame = int(timeEnd * fps)
        else:
            end_frame = total_frames
        # Clamp frame indices to valid range
        start_frame = max(0, min(start_frame, total_frames))
        end_frame = max(start_frame, min(end_frame, total_frames))
        # Set the video position to start_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Crop the frame to the region of interest (ROI)
            cropped = gray[y:y+h, x:x+w]
            # Store the cropped frame
            frames.append(cropped)
            current_frame += 1
        cap.release()
        if frames:
            # Stack all frames into a 3D numpy array (frames, height, width)
            frames_array = np.stack(frames, axis=0)
            # Step 2: Local mean subtraction
            adjusted_frames = []
            num_frames = frames_array.shape[0]

            for i in range(num_frames):
                # Define the local window around the current frame
                start = max(i - window_size, 0)
                end = min(i + window_size + 1, num_frames)
                # Exclude the current frame from the local window
                local_window = np.delete(frames_array[start:end], i - start, axis=0)  # exclude current frame
                # Compute the mean of the local window
                local_mean = np.mean(local_window, axis=0)
                # Subtract the local mean from the current frame
                adjusted = frames_array[i] - local_mean
                adjusted_frames.append(adjusted)

            # Stack adjusted frames into a 3D array
            adjusted_array = np.stack(adjusted_frames, axis=0)

            # Step 3: Compute per-pixel STD across time (after local mean removal)
            std_img = np.std(adjusted_array, axis=0)
            # Save the STD image as a PNG file
            plt.imsave(
                os.path.join(output_folder, f"{os.path.splitext(file)[0]}_std.png"),
                std_img,
                cmap='inferno'
            )
            # Compute the mean of the STD image
            mean_std = float(np.mean(std_img))
            # Remove '_video.mp4' from the key if present
            key = file
            if key.endswith('_video.mp4'):
                key = key[:-10]
            # Store the mean STD and the STD image in the dictionary
            std_dict[key] = (mean_std, std_img)
    # Return the dictionary containing results for all videos
    return std_dict

def plot_stdMatrix(testMatrix, results):
    """
    Reads a test matrix CSV file, adds a column with STD values from results,
    saves the updated CSV, and creates a scatter plot of the results.

    Input:
        testMatrix (str): Path to the CSV file containing test metadata.
        results (dict): Dictionary mapping test names to (mean_std, std_img) tuples,
                        as produced by compute_and_save_std_images.

    Output:
        None. The function saves the updated CSV and displays a scatter plot.
    """
    # Read the test matrix CSV file
    df = pd.read_csv(testMatrix)
    std_values = []
    # For each test, get the corresponding mean_std from results or "ND" if not found
    for name in df['testName']:
        if name in results:
            std_val = results[name][0]  # mean_std
        else:
            std_val = "ND"
        std_values.append(std_val)
    # Add the std values as a new column in the dataframe
    df['std'] = std_values
  
    # Save the updated dataframe back to the CSV file
    df.to_csv(testMatrix, index=False)
    # Plotting

    # Extract x, y, z values for plotting
    x = df['omegaNDExp'].values
    y = df['Z0/r'].values
    z = df['std'].values
    # Filter out points where z is "ND"
    z_numeric = pd.to_numeric(z, errors='coerce')
    mask = ~np.isnan(z_numeric)
    x = x[mask]
    y = y[mask]
    z = z_numeric[mask]
    # Scale marker sizes based on std values
    sizes = (z - z.min()) / (z.max() - z.min()) * 200 + 20  # scale to [20, 220]
    plt.figure(figsize=(8, 6))
    # Create scatter plot
    plt.scatter(
        x, y, 
        c=z, s=sizes, 
        cmap='jet', alpha=0.8, edgecolors='k', 
        marker='o'
    )
    plt.xlabel('$\\omega_f/\\omega_n$')
    plt.ylabel('$Z_0/r$')
    plt.title("Mean std per video")
    plt.colorbar()
    plt.grid(True)
    plt.show()