import cv2 # run pip install opencv-python
import os
from pathlib import Path

"""
Opens a video and extracts the frames as jpgs.
This can be used to extract frames from dashcam videos.
"""
# Put video in "frame_extractor/videos" folder
# UPDATE name and extension
video_title = "test_video" # don't include the extension
video_format = "mov" # mp4, mov, m4a, etc.

fps_saved = 2 # UPDATE frames per second that will be extracted

def main():

    BASE_DIR = Path(__file__).resolve().parent # directory of python file

    video_path = BASE_DIR / f"videos/{video_title}.{video_format}"
    output_folder = BASE_DIR / f"extracted_frames/{video_title}"

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps_video / fps_saved)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # save frame
            filename = os.path.join(output_folder, f"{video_title}_frame_{saved_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames ({fps_saved} per second).")
    
if __name__ == "__main__":
    main()


