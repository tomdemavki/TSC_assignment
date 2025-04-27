import os
import cv2 
import rarfile

#%%
def extract_rar(rar_path, extract_folder):
    """
    Extracts the contents of a .rar archive to the specified folder.

    Args:
        rar_path (str): Path to the .rar file.
        extract_folder (str): Directory where files will be extracted.
    """
    os.makedirs(extract_folder, exist_ok=True)
    with rarfile.RarFile(rar_path) as rf:
        rf.extractall(path=extract_folder)
#%%
def select_roi(video_folder):
    """
    Opens the first .mp4 video in the folder and allows the user to select a region of interest (ROI)
    on the first frame. Returns the ROI coordinates.

    Args:
        video_folder (str): Directory containing video files.

    Returns:
        tuple: (x, y, w, h) coordinates of the selected ROI.
    """
    video_files = [f for f in os.listdir(video_folder) if f.lower().endswith('.mp4')]
    if not video_files:
        raise RuntimeError("No mp4 files found for ROI selection.")
    first_video_path = os.path.join(video_folder, video_files[0])
    cap = cv2.VideoCapture(first_video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Could not read the first frame for ROI selection.")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = cv2.selectROI("Select ROI", gray, showCrosshair=True)
    cv2.destroyAllWindows()
    x, y, w, h = roi
    return x, y, w, h