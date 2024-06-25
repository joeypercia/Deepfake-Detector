import os
import json
import random
import numpy as np
import transformers
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaModel, Trainer, TrainingArguments, EvalPrediction
from torch.utils.data import Dataset, DataLoader
import torch
import librosa
import cv2
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
import psutil
import GPUtil
from sklearn.utils.class_weight import compute_class_weight
import logging
import optuna
from optuna.study import MaxTrialsCallback
from optuna.visualization import plot_optimization_history, plot_param_importances
from torch.cuda.amp import autocast, GradScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
import signal
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
transformers.logging.set_verbosity_error()

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
sentiment_analyzer = SentimentIntensityAnalyzer()

# Global variables for tracking progress
training_loss = []
validation_loss = []
current_trial = 0
current_epoch = 0

class MultiModalDataset(Dataset):
    def __init__(self, metadata, preprocessed_text_dir, preprocessed_audio_dir, preprocessed_video_dir, tokenizer,
                 max_len, max_frames=10):
        self.metadata = metadata
        self.preprocessed_text_dir = preprocessed_text_dir
        self.preprocessed_audio_dir = preprocessed_audio_dir
        self.preprocessed_video_dir = preprocessed_video_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_frames = max_frames

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        entry = self.metadata[index]
        video_file = entry['file']
        split = entry['split']
        label = 1 if entry['n_fakes'] > 0 else 0
        text_file_path = os.path.join(self.preprocessed_text_dir, split,
                                      os.path.splitext(os.path.basename(video_file))[0] + '.txt')
        audio_file_path = os.path.join(self.preprocessed_audio_dir, split,
                                       os.path.splitext(os.path.basename(video_file))[0] + '.wav')
        video_frames_dir = os.path.join(self.preprocessed_video_dir, split,
                                        os.path.splitext(os.path.basename(video_file))[0])

        if os.path.exists(text_file_path) and os.path.exists(audio_file_path) and os.path.exists(video_frames_dir):
            with open(text_file_path, 'r') as text_file:
                text = text_file.read()

            audio_features = self.extract_audio_features(audio_file_path)
            video_features = self.extract_video_features(video_frames_dir)
            text_features = self.extract_text_features(text)

            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )

            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'audio': torch.tensor(audio_features, dtype=torch.float),
                'video': torch.tensor(video_features, dtype=torch.float),
                'text_features': torch.tensor(text_features, dtype=torch.float),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        else:
            raise FileNotFoundError(f"Missing components for {video_file}")

    def extract_audio_features(self, audio_file_path):
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
        return features

    def extract_video_features(self, video_frames_dir):
        video_features = []
        for frame_file in sorted(os.listdir(video_frames_dir)):
            frame_path = os.path.join(video_frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            frame = cv2.resize(frame, (224, 224))
            frame = frame.transpose((2, 0, 1))  # Convert to CHW format
            video_features.append(frame)
        video_features = np.stack(video_features)

        if len(video_features) > self.max_frames:
            video_features = video_features[:self.max_frames]
        else:
            pad_len = self.max_frames - len(video_features)
            padding = np.zeros((pad_len, 3, 224, 224))
            video_features = np.concatenate((video_features, padding), axis=0)

        return video_features

    def extract_text_features(self, text):
        doc = nlp(text)
        sentiment_score = sentiment_analyzer.polarity_scores(text)["compound"]
        num_words = len(doc)
        num_sentences = len(list(doc.sents))
        num_entities = len(doc.ents)
        features = [sentiment_score, num_words, num_sentences, num_entities]
        return features

def load_metadata(metadata_path, sample_ratio=0.1):  # Reduced sample ratio for quicker trials
    logger.info("Loading metadata...")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    sample_size = int(len(metadata) * sample_ratio)

    sampled_metadata = []
    with tqdm(total=sample_size, desc="Sampling metadata") as pbar:
        for _ in range(sample_size):
            sampled_metadata.append(random.choice(metadata))
            pbar.update(1)

    logger.info(f"Sampled {len(sampled_metadata)} metadata entries out of {len(metadata)} total entries.")
    return sampled_metadata

def extract_labels_from_dataset(dataset):
    labels = []
    for data in tqdm(dataset, desc="Extracting labels from dataset"):
        labels.append(data['labels'].item())
    return labels

def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.info(f"Memory Usage: {memory_info.rss / 1024 ** 2:.2f} MB")
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        logger.info(f"GPU {gpu.id} - Memory Usage: {gpu.memoryUsed / 1024:.2f} GB / {gpu.memoryTotal / 1024:.2f} GB")

def compute_metrics(pred: EvalPrediction):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(labels, preds)

    if len(np.unique(labels)) == 1:
        roc_auc = float('nan')
    else:
        roc_auc = roc_auc_score(labels, preds)

    precision_vals, recall_vals, _ = precision_recall_curve(labels, preds)
    pr_auc = auc(recall_vals, precision_vals)

    return {
        'accuracy': (preds == labels).mean(),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'conf_matrix': conf_matrix.tolist(),  # Convert to list for JSON serialization
    }

def plot_loss(training_loss, validation_loss):
    epochs = range(1, len(training_loss) + 1)
    plt.plot(epochs, training_loss, 'b-o', label='Training Loss')
    plt.plot(epochs, validation_loss, 'r-o', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()

def save_optuna_plots(study):
    fig1 = plot_optimization_history(study)
    fig1.write_image("optuna_optimization_history.png")
    fig2 = plot_param_importances(study)
    fig2.write_image("optuna_param_importances.png")

def save_progress(study=None):
    global model, text_model, tokenizer, current_trial, current_epoch, training_loss, validation_loss
    torch.save(model.state_dict(), 'model/model.pth')
    text_model.save_pretrained('model/text_model')
    tokenizer.save_pretrained('tokenizer')
    with open('progress.json', 'w') as f:
        json.dump({
            'current_trial': current_trial,
            'current_epoch': current_epoch,
            'training_loss': training_loss,
            'validation_loss': validation_loss,
        }, f)
    logger.info("Progress saved.")
    if study is not None:
        save_optuna_plots(study)

def signal_handler(sig, frame):
    logger.info("Interrupt signal received. Saving progress...")
    save_progress()
    plot_loss(training_loss, validation_loss)
    sys.exit(0)

def train_model():
    global model, text_model, tokenizer, current_trial, current_epoch, training_loss, validation_loss

    preprocessed_text_dir = "LAV-DF/preprocessed_text"
    preprocessed_audio_dir = "LAV-DF/preprocessed_audio"
    preprocessed_video_dir = "LAV-DF/preprocessed_videos"
    metadata_path = "LAV-DF/metadata.json"

    metadata = load_metadata(metadata_path, sample_ratio=0.1)  # Adjust sample ratio as needed

    logger.info("Initializing tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    max_len = 128

    logger.info("Initializing datasets...")
    train_metadata, temp_metadata = train_test_split(metadata, test_size=0.4, random_state=42)
    dev_metadata, test_metadata = train_test_split(temp_metadata, test_size=0.5, random_state=42)

    logger.info("Creating training dataset...")
    train_dataset = MultiModalDataset(train_metadata, preprocessed_text_dir, preprocessed_audio_dir, preprocessed_video_dir, tokenizer, max_len)
    logger.info("Creating validation dataset...")
    dev_dataset = MultiModalDataset(dev_metadata, preprocessed_text_dir, preprocessed_audio_dir, preprocessed_video_dir, tokenizer, max_len)
    logger.info("Creating test dataset...")
    test_dataset = MultiModalDataset(test_metadata, preprocessed_text_dir, preprocessed_audio_dir, preprocessed_video_dir, tokenizer, max_len)

    logger.info("Creating training dataloader...")
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=16, pin_memory=True)
    logger.info("Creating validation dataloader...")
    dev_dataloader = DataLoader(dev_dataset, batch_size=256, num_workers=16, pin_memory=True)
    logger.info("Creating test dataloader...")
    test_dataloader = DataLoader(test_dataset, batch_size=256, num_workers=16, pin_memory=True)

    logger.info("Extracting labels from training dataset...")
    labels = extract_labels_from_dataset(train_dataset)
    logger.info(f"Extracted labels from training dataset: {np.unique(labels, return_counts=True)}")

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    if len(class_weights) < 2:
        class_weights = torch.tensor([1.0, 1.0], dtype=torch.float).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    logger.info(f"Computed class weights: {class_weights}")

    logger.info("Initializing model...")

    class CrossModalAttention(nn.Module):
        def __init__(self, embed_dim, num_heads=8, num_layers=2):
            super(CrossModalAttention, self).__init__()
            self.attn_layers = nn.ModuleList(
                [nn.MultiheadAttention(embed_dim, num_heads, batch_first=True) for _ in range(num_layers)]
            )
            self.norm_layers = nn.ModuleList(
                [nn.LayerNorm(embed_dim) for _ in range(num_layers)]
            )

        def forward(self, query, key, value):
            for attn, norm in zip(self.attn_layers, self.norm_layers):
                attn_output, _ = attn(query, key, value)
                query = norm(query + attn_output)  # Add & Norm
            return query

    class MultiModalModel(nn.Module):
        def __init__(self, text_model, num_labels, class_weights):
            super(MultiModalModel, self).__init__()
            self.text_model = text_model

            self.audio_cnn = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(32 * 3 * 1, 768),
                nn.ReLU()
            )

            self.video_cnn = models.resnet18(pretrained=True)
            self.video_cnn.fc = nn.Linear(self.video_cnn.fc.in_features, 768)

            self.cross_modal_attn = CrossModalAttention(768, num_heads=8, num_layers=2)
            self.classifier = nn.Linear(768 * 3 + 4, num_labels)  # Adjusted for 4 text features
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

            self.dropout = nn.Dropout(p=0.3)  # Adding dropout for regularization

        def forward(self, input_ids, attention_mask, audio=None, video=None, text_features=None, labels=None):
            with autocast():
                outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
                text_features_embedding = outputs.last_hidden_state

                if audio is not None:
                    audio_features = self.audio_cnn(audio).unsqueeze(1).expand(-1, text_features_embedding.size(1), -1)
                else:
                    audio_features = torch.zeros(text_features_embedding.size(0), text_features_embedding.size(1), 768).to(text_features_embedding.device)

                if video is not None:
                    batch_size, num_frames, channels, height, width = video.size()
                    video = video.view(batch_size * num_frames, channels, height, width)
                    video_features = self.video_cnn(video)
                    video_features = video_features.view(batch_size, num_frames, -1)
                    video_features = torch.mean(video_features, dim=1).unsqueeze(1).expand(-1, text_features_embedding.size(1), -1)
                else:
                    video_features = torch.zeros(text_features_embedding.size(0), text_features_embedding.size(1), 768).to(text_features_embedding.device)

                audio_attn = self.cross_modal_attn(audio_features, text_features_embedding, text_features_embedding)
                video_attn = self.cross_modal_attn(video_features, text_features_embedding, text_features_embedding)

                combined_features = torch.cat((text_features_embedding[:, 0, :], audio_attn[:, 0, :], video_attn[:, 0, :], text_features), dim=1)  # Include text features
                combined_features = self.dropout(combined_features)  # Applying dropout

                logits = self.classifier(combined_features)

            if labels is not None:
                loss = self.loss_fn(logits, labels)
                return {"loss": loss, "logits": logits}
            else:
                return {"logits": logits}

        def summary(self):
            logger.info("Model Summary:")
            logger.info(f"Text Model: {self.text_model}")
            logger.info(f"Audio CNN: {self.audio_cnn}")
            logger.info(f"Video CNN: {self.video_cnn}")
            logger.info(f"Cross-Modal Attention: {self.cross_modal_attn}")
            logger.info(f"Classifier: {self.classifier}")
            logger.info(f"Dropout: {self.dropout}")

    text_model = RobertaModel.from_pretrained('roberta-base')
    model = MultiModalModel(text_model, num_labels=2, class_weights=class_weights)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    scaler = GradScaler()

    def objective(trial):
        global current_trial, current_epoch, training_loss, validation_loss
        current_trial = trial.number

        logger.info("Setting up training arguments and Trainer...")
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=trial.suggest_int('num_train_epochs', 1, 5),
            per_device_train_batch_size=trial.suggest_categorical('batch_size', [16, 32, 64]),
            learning_rate=trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True),
            warmup_steps=trial.suggest_int('warmup_steps', 100, 1000),
            weight_decay=trial.suggest_float('weight_decay', 0.01, 0.1),
            logging_dir='./logs',
            logging_steps=10,
            eval_strategy="steps",
            save_steps=500,
            eval_steps=250,
            save_total_limit=2,
            fp16=True,
            gradient_accumulation_steps=trial.suggest_int('gradient_accumulation_steps', 1, 4),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False
        )

        num_heads = trial.suggest_categorical('num_heads', [1, 2, 4, 8, 16])
        num_layers = trial.suggest_int('num_layers', 1, 3)
        model.cross_modal_attn = CrossModalAttention(768, num_heads=num_heads, num_layers=num_layers)

        training_loss = []
        validation_loss = []

        def custom_training_loop():
            best_eval_loss = float('inf')
            patience = 2
            patience_counter = 0

            for epoch in range(training_args.num_train_epochs):
                current_epoch = epoch
                trainer.train()
                training_metrics = trainer.evaluate(eval_dataset=train_dataset)
                validation_metrics = trainer.evaluate(eval_dataset=dev_dataset)
                training_loss.append(training_metrics['eval_loss'])
                validation_loss.append(validation_metrics['eval_loss'])

                if validation_metrics['eval_loss'] < best_eval_loss:
                    best_eval_loss = validation_metrics['eval_loss']
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info("Early stopping triggered.")
                    break

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            compute_metrics=compute_metrics,
            data_collator=lambda data: {
                'input_ids': torch.stack([f['input_ids'] for f in data]),
                'attention_mask': torch.stack([f['attention_mask'] for f in data]),
                'audio': torch.stack([f['audio'] for f in data]),
                'video': torch.stack([f['video'] for f in data]),
                'text_features': torch.stack([f['text_features'] for f in data]),
                'labels': torch.stack([f['labels'] for f in data])
            }
        )

        logger.info("Starting training...")
        custom_training_loop()
        log_memory_usage()
        logger.info("Training finished.")

        logger.info("Starting evaluation...")
        eval_result = trainer.evaluate()
        logger.info(eval_result)
        log_memory_usage()
        logger.info("Evaluation finished.")

        return eval_result['eval_loss']

    # Load or create a new study
    storage_name = "sqlite:///study.db"
    study_name = "deepfake-detection-study"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_name)
    except KeyError:
        study = optuna.create_study(direction='minimize', study_name=study_name, storage=storage_name)

    study.optimize(objective, n_trials=10, n_jobs=1, callbacks=[MaxTrialsCallback(10)])

    logger.info(f"Best trial: {study.best_trial.params}")

    best_params = study.best_trial.params
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=best_params['num_train_epochs'],
        per_device_train_batch_size=best_params['batch_size'],
        learning_rate=best_params['learning_rate'],
        warmup_steps=best_params['warmup_steps'],
        weight_decay=best_params['weight_decay'],
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="steps",
        save_steps=500,
        eval_steps=250,
        save_total_limit=2,
        fp16=True,
        gradient_accumulation_steps=best_params['gradient_accumulation_steps'],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

    num_heads = best_params['num_heads']
    num_layers = best_params['num_layers']
    model.cross_modal_attn = CrossModalAttention(768, num_heads=num_heads, num_layers=num_layers)

    def final_training_loop():
        best_eval_loss = float('inf')
        patience = 2
        patience_counter = 0

        for epoch in range(training_args.num_train_epochs):
            current_epoch = epoch
            trainer.train()
            training_metrics = trainer.evaluate(eval_dataset=train_dataset)
            validation_metrics = trainer.evaluate(eval_dataset=dev_dataset)
            training_loss.append(training_metrics['eval_loss'])
            validation_loss.append(validation_metrics['eval_loss'])

            if validation_metrics['eval_loss'] < best_eval_loss:
                best_eval_loss = validation_metrics['eval_loss']
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        data_collator=lambda data: {
            'input_ids': torch.stack([f['input_ids'] for f in data]),
            'attention_mask': torch.stack([f['attention_mask'] for f in data]),
            'audio': torch.stack([f['audio'] for f in data]),
            'video': torch.stack([f['video'] for f in data]),
            'text_features': torch.stack([f['text_features'] for f in data]),
            'labels': torch.stack([f['labels'] for f in data])
        }
    )

    logger.info("Starting final training with best parameters...")
    final_training_loop()
    log_memory_usage()
    logger.info("Training finished.")

    logger.info("Starting evaluation...")
    trainer_metrics = trainer.evaluate()
    logger.info(trainer_metrics)
    log_memory_usage()
    logger.info("Evaluation finished.")

    test_trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        data_collator=lambda data: {
            'input_ids': torch.stack([f['input_ids'] for f in data]),
            'attention_mask': torch.stack([f['attention_mask'] for f in data]),
            'audio': torch.stack([f['audio'] for f in data]),
            'video': torch.stack([f['video'] for f in data]),
            'text_features': torch.stack([f['text_features'] for f in data]),
            'labels': torch.stack([f['labels'] for f in data])
        }
    )

    logger.info("Evaluating on test dataset...")
    test_metrics = test_trainer.evaluate()
    logger.info(test_metrics)
    log_memory_usage()

    logger.info("Saving the model...")
    save_progress(study)
    logger.info("Model saved.")

    plot_loss(training_loss, validation_loss)

    model.summary()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    train_model()
