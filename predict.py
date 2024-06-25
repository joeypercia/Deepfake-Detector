import os
import time
import torch
import ffmpeg
import cv2
import librosa
import numpy as np
import speech_recognition as sr
from transformers import RobertaTokenizer, RobertaModel
from model import MultiModalModel, CrossModalAttention
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_dir, tokenizer_dir, device):
    text_model = RobertaModel.from_pretrained(os.path.join(model_dir, 'text_model'))
    model = MultiModalModel(text_model, num_labels=2, class_weights=None)
    state_dict = torch.load(os.path.join(model_dir, 'model.pth'), map_location=device)
    model_state_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
    model_state_dict.update(state_dict)
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    model.eval()

    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_dir)

    return model, tokenizer


def extract_audio(video_path, audio_output_path):
    stream = ffmpeg.input(video_path)
    stream = ffmpeg.output(stream, audio_output_path)
    ffmpeg.run(stream, overwrite_output=True)


def extract_frames(video_path, output_dir, interval=30):
    logger.info(f"Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file: {video_path}")
        return 0

    frame_count = 0
    extracted_count = 0
    os.makedirs(output_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            extracted_count += 1
        frame_count += 1

    cap.release()
    logger.info(f"Extracted {extracted_count} frames from the video: {video_path}")
    return extracted_count


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


def process_video(video_path, temp_dir, frame_interval=30):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    video_output_dir = os.path.join(temp_dir, "frames", video_name)
    audio_output_path = os.path.join(temp_dir, "audio", f"{video_name}.wav")
    text_output_path = os.path.join(temp_dir, "transcription", f"{video_name}.txt")

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

    return video_output_dir, audio_output_path, text_output_path


def extract_audio_features(audio_file_path):
    y, sr = librosa.load(audio_file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    features = np.concatenate([
        np.mean(mfcc.T, axis=0),
        np.mean(chroma.T, axis=0),
        np.mean(zcr.T, axis=0),
        np.mean(spectral_contrast.T, axis=0),
        np.mean(tonnetz.T, axis=0),
        np.mean(rms.T, axis=0)
    ])
    features = np.resize(features, (1, 13, 6))  # Reshape to a 3D array
    logger.info(f"Audio features: {features}")
    return features


def extract_video_features(video_frames_dir, max_frames=10):
    video_features = []
    for frame_file in sorted(os.listdir(video_frames_dir)):
        frame_path = os.path.join(video_frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        frame = cv2.resize(frame, (224, 224))
        frame = frame.transpose((2, 0, 1))  # Convert to CHW format
        video_features.append(frame)
    video_features = np.stack(video_features)

    if len(video_features) > max_frames:
        video_features = video_features[:max_frames]
    else:
        pad_len = max_frames - len(video_features)
        padding = np.zeros((pad_len, 3, 224, 224))
        video_features = np.concatenate((video_features, padding), axis=0)

    logger.info(f"Video features shape: {video_features.shape}")
    return video_features


def extract_text_features(text, tokenizer, max_len=128):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = encoding['input_ids'].flatten()
    attention_mask = encoding['attention_mask'].flatten()
    logger.info(f"Text input ids: {input_ids}")
    logger.info(f"Text attention mask: {attention_mask}")
    return input_ids, attention_mask


def predict(video_path, model, tokenizer, device, frame_interval=30, max_frames=10, max_len=128):
    temp_dir = "temp"
    video_frames_dir, audio_path, text_path = process_video(video_path, temp_dir, frame_interval)

    with open(text_path, 'r') as text_file:
        text = text_file.read()
    input_ids, attention_mask = extract_text_features(text, tokenizer, max_len)
    audio_features = extract_audio_features(audio_path)
    video_features = extract_video_features(video_frames_dir, max_frames)

    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    audio_features = torch.tensor(audio_features, dtype=torch.float).unsqueeze(0).to(device)
    video_features = torch.tensor(video_features, dtype=torch.float).unsqueeze(0).to(device)
    text_features = torch.tensor([0, 0, 0, 0], dtype=torch.float).unsqueeze(0).to(device)  # Dummy text features for now

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, audio=audio_features, video=video_features,
                        text_features=text_features)
        logits = outputs['logits']
        probs = torch.nn.functional.softmax(logits, dim=1)
        logger.info(f"Prediction logits: {logits}")
        logger.info(f"Prediction probabilities: {probs}")
        prediction = torch.argmax(probs, dim=1).item()

    # Cleanup temporary files and directories
    for root, dirs, files in os.walk(temp_dir, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))
    os.rmdir(temp_dir)

    return "Deepfake" if prediction == 1 else "Genuine"


if __name__ == "__main__":
    import sys

    video_path = sys.argv[1]
    model_dir = "model"
    tokenizer_dir = "tokenizer"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model_and_tokenizer(model_dir, tokenizer_dir, device)
    result = predict(video_path, model, tokenizer, device)
    print(f"The video is predicted to be: {result}")
