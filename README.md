# Multi-Modal Deepfake Detection System

This repository contains the implementation of a multi-modal deepfake detection system, integrating video, audio, and text data to determine the authenticity of digital media.

## Overview

The project uses a robust architecture that combines the strengths of different data modalities:
- **Video Frames:** Extracted using OpenCV.
- **Audio:** Processed and extracted features using librosa.
- **Text:** Audio transcripts obtained via Google Speech-to-Text API.

The core model is built using the Hugging Face Transformers library, leveraging a RoBERTa model for text processing and custom neural network layers for processing audio and video inputs.

## Setup

### Requirements

- Python
- PyTorch
- transformers
- librosa
- OpenCV
- ffmpeg
- NumPy
- LAV-DF Dataset
- facial-landmarks-recognition

### Installation

1. Clone the repository:
    ```
    git clone https://github.com/joeypercia/Deepfake-Detector.git
    cd Deepfake-Detector
    ```

2. Install the required packages:
    ```
    pip install -r requirements.txt
    ```

3. The preprocess_videos function samples a subset of videos from the specified directory, extracts frames, audio, and transcribes the audio to text. To run preprocessing, extract LAV-DF to the project directory and execute:
    ```
    python video_processing.py
    ```

4. The training process integrates data from all modalities. To start training, configure your settings in train_model.py and run:
    ```
    python train.py
    ```
    This script handles data loading, model training, and evaluation. It optimizes the model using Optuna for hyperparameter tuning.

5. To predict if a video is a deepfake, use the predict.py script. Ensure you have a trained model and tokenizer available:
    ```
    python predict.py path_to_your_video.mp4
    ```
    The output will indicate whether the video is a deepfake or genuine.
    