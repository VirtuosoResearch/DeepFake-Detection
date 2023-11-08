# -*- coding: utf-8 -*-
"""Untitled15.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gQgBK50vDXcbkJcgntn6F4_UBUTjMsXH

Referred from https://deepnote.com/@lulus/Load-dataset-from-kaggle-to-google-colab-411fca3f-ee05-45fd-b7e4-a28990260184#:~:text=Fire%20up%20a%20Google%20Colab,need%20to%20load%20the%20dataset.
"""

#Make directory name kaggle
! mkdir ~/.kaggle

from google.colab import files

files.upload()

#Copy the json kaggle to this directory
! cp kaggle.json ~/.kaggle/

#Allocate the required permission for this file.
! chmod 600 ~/.kaggle/kaggle.json

#Downloading competition dataset
! kaggle competitions download deepfake-detection-challenge

# Put on the same directory
from zipfile import ZipFile

# specifying the name of the zip file
file = "deepfake-detection-challenge.zip"

# open the zip file in read mode
with ZipFile(file, 'r') as zip:
    # list all the contents of the zip file
    zip.printdir()

    # extract all files
    print('extraction...')
    zip.extractall()
    print('Done!')

import json

# Path to the metadata.json file
metadata_path = 'train_sample_videos/metadata.json'

# Load the metadata into a dictionary
with open(metadata_path, 'r') as file:
    metadata = json.load(file)

metadata

len(metadata)

import os

# Create directories for real and fake videos
real_videos_dir = 'train_sample_videos/real_videos'
fake_videos_dir = 'train_sample_videos/fake_videos'

if not os.path.exists(real_videos_dir):
    os.makedirs(real_videos_dir)
if not os.path.exists(fake_videos_dir):
    os.makedirs(fake_videos_dir)

# Function to move videos to the respective directory
def move_videos(metadata, source_dir, real_dir, fake_dir):
    for filename, attributes in metadata.items():
        # Construct the path to the source video
        source_path = os.path.join(source_dir, filename)

        # Check if the source video exists
        if not os.path.exists(source_path):
            print(f"Warning: {source_path} does not exist.")
            continue

        # Move the video to the corresponding directory
        if attributes['label'] == 'REAL':
            destination_path = os.path.join(real_dir, filename)
        else:  # 'FAKE'
            destination_path = os.path.join(fake_dir, filename)

        # Move the video (if you want to keep the original, use shutil.copy2 instead)
        os.rename(source_path, destination_path)

# Path to the directory where all extracted videos are located
source_videos_dir = 'train_sample_videos/'

# Move the videos
move_videos(metadata, source_videos_dir, real_videos_dir, fake_videos_dir)

!ls train_sample_videos

# Dictionary to keep track of fake to original mappings
fake_to_original = {}

# Populate the dictionary
for filename, attributes in metadata.items():
    if attributes['label'] == 'FAKE':
        fake_to_original[filename] = attributes['original']

# Optionally, write this mapping to a file
with open('train_sample_videos/fake_to_original.json', 'w') as file:
    json.dump(fake_to_original, file, indent=4)

!ls train_sample_videos/real_videos

!ls train_sample_videos/fake_videos

# Load the metadata.json file
metadata_path = 'train_sample_videos/metadata.json'
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

# Base directory where 'REAL' and 'fake_videos' directories should be located
base_dir = 'train_sample_videos/'

# Label to directory mapping
label_to_dir = {
    'FAKE': 'fake_videos',
    # Add a similar mapping for 'REAL' if the folder name is different
    'REAL': 'real_videos'  # Change this if your 'REAL' videos are in a differently named folder
}

# Function to check the existence of video files according to metadata labels
def verify_videos(metadata, base_dir):
    errors = []
    for filename, attributes in metadata.items():
        # Translate the label to the correct directory name
        dir_name = label_to_dir.get(attributes['label'].strip(), None)
        if dir_name is None:
            errors.append((filename, 'unknown label'))
            continue

        expected_dir = os.path.join(base_dir, dir_name)
        video_path = os.path.join(expected_dir, filename.strip())

        # Check if the file exists at the expected location
        if not os.path.isfile(video_path):
            errors.append((filename, 'missing'))

    return errors

# Run the verification
errors = verify_videos(metadata, base_dir)

# Print out the results
if errors:
    print("Errors found in video file locations:")
    for filename, error_type in errors:
        print(f"File {filename} is {error_type}.")
else:
    print("All videos are in their correct directories according to the metadata.")

!pip install opencv-python

import os
import glob
import cv2

os.makedirs('frames_directory')

import os
import glob
import cv2

# Function to extract frames from a given video
def extract_frames(video_path, output_directory, every_n_frame=30):
    video_name = os.path.basename(video_path)
    video_dir = os.path.join(output_directory, os.path.splitext(video_name)[0])
    os.makedirs(video_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame if it's the nth frame
        if frame_count % every_n_frame == 0:
            print(frame_count)
            frame_path = os.path.join(video_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)

        frame_count += 1
    cap.release()

# Define paths
root_dir = 'train_sample_videos/'
real_videos_dir = os.path.join(root_dir, 'real_videos')
fake_videos_dir = os.path.join(root_dir, 'fake_videos')
frames_dir = 'frames_directory'

# Extract frames from real videos
real_videos = glob.glob(f'{real_videos_dir}/*.mp4')
c = 1
for video_path in real_videos:

    print(f"For {video_path}")
    print(f"{c}")

    extract_frames(video_path, os.path.join(frames_dir, 'real_frames'),30)
    c+=1

c=1
# Extract frames from fake videos
fake_videos = glob.glob(f'{fake_videos_dir}/*.mp4')
for video_path in fake_videos:
    print(f"For {video_path}")
    print(f"{c}")
    extract_frames(video_path, os.path.join(frames_dir, 'fake_frames'),30)
    c+=1

!ls frames_directory/real_frames/abarnvbtwb

from google.colab.patches import cv2_imshow
import cv2
import os

# Function to display frames in Google Colab
def show_frames_colab(frames_directory, display_time=1):
    frames = sorted(os.listdir(frames_directory))  # Sort the frames

    for frame_name in frames:
        frame_path = os.path.join(frames_directory, frame_name)
        frame = cv2.imread(frame_path)
        cv2_imshow(frame)  # Use cv2_imshow in Colab
        cv2.waitKey(display_time * 1000)  # Wait for specified time or until key press

# Usage example
frames_directory = 'frames_directory/real_frames/abarnvbtwb/'  # Replace with your path
show_frames_colab(frames_directory, display_time=1)  # Show each frame for 1 second

import cv2
import os

def process_frame(frame_path):
    if not os.path.exists(frame_path):
        print(f"File {frame_path} does not exist.")
        return None

    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Failed to read the image from {frame_path}.")
        return None

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # ... (rest of your processing code)
    except cv2.error as e:
        print(f"OpenCV error: {e}")
        return None

    return gray  # or whatever result you want to return

# Usage example:
frame_path = 'frames_directory/real_frames/abarnvbtwb/frame_0.jpg'
gray_frame = process_frame(frame_path)

frames_directory = 'frames_directory/real_frames/abarnvbtwb/frame_0.jpg'  # Replace with your path
show_frames_colab(frames_directory, display_time=1)