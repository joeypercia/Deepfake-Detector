import os
import random
import cv2
import ffmpeg
import numpy as np
import speech_recognition as sr
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def extract_frames(video_path, output_dir, interval=30):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
        frame_count += 1
    cap.release()

def extract_audio(video_path, audio_output_path):
    stream = ffmpeg.input(video_path)
    stream = ffmpeg.output(stream, audio_output_path)
    ffmpeg.run(stream, overwrite_output=True)

def transcribe_audio(audio_path, max_retries=5):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.record(source)
            for attempt in range(max_retries):
                try:
                    text = recognizer.recognize_google(audio_data)
                    return text
                except sr.UnknownValueError:
                    return ""
                except sr.RequestError as e:
                    if "rate limit" in str(e):
                        print(f"Rate limit hit, retrying in {2 ** attempt} seconds...")
                        time.sleep(2 ** attempt)
                    else:
                        print(f"Request error: {e}")
                        break
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return ""

def process_video(video_path, output_video_dir, output_audio_dir, output_text_dir, frame_interval=30):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    relative_path = os.path.relpath(os.path.dirname(video_path), input_directory)

    # Create output directories for the current video
    video_output_dir = os.path.join(output_video_dir, relative_path, video_name)
    audio_output_path = os.path.join(output_audio_dir, relative_path, f"{video_name}.wav")
    text_output_path = os.path.join(output_text_dir, relative_path, f"{video_name}.txt")

    # Skip processing if the text output file already exists
    if os.path.exists(text_output_path):
        print(f"Skipping {video_name}, already processed.")
        return

    os.makedirs(video_output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(audio_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(text_output_path), exist_ok=True)

    # Extract frames
    extract_frames(video_path, video_output_dir, frame_interval)

    # Extract audio
    extract_audio(video_path, audio_output_path)

    # Transcribe audio
    transcription = transcribe_audio(audio_output_path)
    with open(text_output_path, 'w') as text_file:
        text_file.write(transcription)

def preprocess_videos(input_dir, output_video_dir, output_audio_dir, output_text_dir, frame_interval=30, sample_ratio=0.1):
    video_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.mp4'):
                video_files.append(os.path.join(root, file))

    # Sample a subset of the video files
    sample_size = int(len(video_files) * sample_ratio)
    sampled_videos = random.sample(video_files, sample_size)

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_video, video, output_video_dir, output_audio_dir, output_text_dir, frame_interval) for video in sampled_videos]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing video: {e}")

# Define directories
input_directory = "LAV-DF"
output_video_dir = "LAV-DF/preprocessed_videos"
output_audio_dir = "LAV-DF/preprocessed_audio"
output_text_dir = "LAV-DF/preprocessed_text"

# Create output directories if they don't exist
os.makedirs(output_video_dir, exist_ok=True)
os.makedirs(output_audio_dir, exist_ok=True)
os.makedirs(output_text_dir, exist_ok=True)

# Preprocess videos (with a 100% sample)
preprocess_videos(input_directory, output_video_dir, output_audio_dir, output_text_dir, sample_ratio=1.0)
