import os
import pandas as pd
from tqdm import tqdm
import sys

# Define paths
dataset_path = "dataset"
labels_path = "Label.xlsx"
output_frames_path = "frames"

# Define time range (in seconds) around the seizure onset
TIME_BEFORE_SEIZURE = 5
TIME_AFTER_SEIZURE = 10

# Load labels
labels_df = pd.read_excel(labels_path)
labels_df.rename(columns={"PatID": "patient_id", "#Seizure": "video_name", "Clinical Onset": "seizure_start", "Seizure Type": "seizure_type"}, inplace=True)

def time_to_seconds(time_obj):
    """Convert time to seconds."""
    if pd.isna(time_obj):
        return None
    return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second

labels_df["seizure_start"] = labels_df["seizure_start"].apply(time_to_seconds)

def determine_label(timestamp, seizure_start, seizure_type):
    """Determine frame label based on seizure start time."""
    if seizure_start is None:
        return "No_Seizure"
    return seizure_type.replace(" ", "_") if timestamp >= seizure_start else "No_Seizure"

def get_last_extracted_frame(base_folder, patient_id, seizure_name):
    """Get the last extracted frame number from the correct folder."""
    patient_seizure_folder = os.path.join(base_folder, f"{patient_id}_{seizure_name}")
    if not os.path.exists(patient_seizure_folder):
        return -1

    frame_files = [f for f in os.listdir(patient_seizure_folder) if f.endswith(".jpg")]
    if not frame_files:
        return -1

    # Extract frame numbers from filenames
    frame_numbers = [int(f.split("_")[-2]) for f in frame_files if f.split("_")[-2].isdigit()]
    return max(frame_numbers) if frame_numbers else -1

def extract_frames(video_path, seizure_start, seizure_type, patient_id, seizure_name):
    """Extract frames within a specific time range around the seizure onset."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30  # Default frame rate if not available

    if not cap.isOpened():
        print(f"\n❌ Error: Unable to open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps

    # Define the time range for frame extraction
    start_time = max(0, seizure_start - TIME_BEFORE_SEIZURE) if seizure_start is not None else 0
    end_time = min(total_duration, seizure_start + TIME_AFTER_SEIZURE) if seizure_start is not None else total_duration

    # Convert time range to frame indices
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # Get last extracted frames
    last_frame_no_seizure = get_last_extracted_frame(os.path.join(output_frames_path, "No_Seizure"), patient_id, seizure_name)
    last_frame_seizure = get_last_extracted_frame(os.path.join(output_frames_path, seizure_type.replace(" ", "_")), patient_id, seizure_name)

    # Set the correct start frame
    last_extracted_frame = max(last_frame_no_seizure, last_frame_seizure)
    if last_extracted_frame >= start_frame:
        start_frame = last_extracted_frame + 1

    # If all frames are already extracted, skip
    if start_frame >= end_frame:
        print(f"\n✅ All frames already extracted for {patient_id} -> {seizure_name}. Skipping.")
        cap.release()
        return

    print(f"\nℹ️ Extracting frames from {start_time:.2f}s to {end_time:.2f}s for {patient_id} -> {seizure_name}")

    total_frames_to_process = end_frame - start_frame

    # Single progress bar instance
    with tqdm(
        total=total_frames_to_process,
        desc=f"Processing {os.path.basename(video_path)}",
        unit="frame",
        file=sys.stderr,
        leave=True,
        dynamic_ncols=True,
        bar_format="{l_bar}{bar} {n_fmt}/{total_fmt} frames [{elapsed}<{remaining}, {rate_fmt}]"
    ) as progress_bar:

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        while cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            timestamp = frame_idx / fps

            # Determine label based on seizure onset
            label = determine_label(timestamp, seizure_start, seizure_type)

            # Create output folder for this label
            label_folder = os.path.join(output_frames_path, label, f"{patient_id}_{seizure_name}")
            os.makedirs(label_folder, exist_ok=True)

            # Save frame
            frame_filename = f"{patient_id}_{seizure_name}_frame_{frame_idx:06d}_{label}.jpg"
            frame_path = os.path.join(label_folder, frame_filename)

            # Skip if frame already exists
            if os.path.exists(frame_path):
                progress_bar.update(1)
                continue

            # Save the frame in high quality
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            progress_bar.update(1)

    cap.release()

# Iterate through the label file
for _, row in labels_df.iterrows():
    patient_id = str(row["patient_id"]).strip()
    seizure_name = str(row["video_name"]).strip()
    seizure_start = row["seizure_start"]
    seizure_type = row["seizure_type"].strip().replace(" ", "_") if pd.notna(row["seizure_type"]) and row["seizure_type"].strip() != "" else "No_Seizure"

    patient_folder = os.path.join(dataset_path, patient_id)
    if not os.path.exists(patient_folder):
        print(f"⚠️ Patient folder not found: {patient_folder}")
        continue

    matched_videos = [file for file in os.listdir(patient_folder) if file.lower().startswith(seizure_name.lower()) and file.lower().endswith(".mp4")]

    if not matched_videos:
        print(f"⚠️ No matching video found for Patient {patient_id} -> {seizure_name}")
        continue

    video_path = os.path.join(patient_folder, matched_videos[0])
    extract_frames(video_path, seizure_start, seizure_type, patient_id, seizure_name)
