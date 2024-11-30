import numpy as np
import librosa
from scipy.signal import resample
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch

# Function to combine f0 and log-mel spectrogram
def add_f0_as_additional_channel(audio_array, sampling_rate, feature_extractor):
    try:
        # Ensure audio_array is a NumPy array
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array)
            print("Converted audio_array to NumPy array in add_f0_as_additional_channel.")

        # Check if audio_array is empty
        if audio_array.size == 0:
            raise ValueError("Audio data is empty.")

        # Proceed with feature extraction
        log_mel = feature_extractor(audio_array, sampling_rate=sampling_rate).input_features[0]

        # Extract f0 using librosa.pyin
        f0, _, _ = librosa.pyin(
            audio_array,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            frame_length=2048,
            hop_length=512,
        )

        # Handle NaNs and resample f0 to match log-mel frames
        f0 = np.nan_to_num(f0)
        f0_resampled = resample(f0, log_mel.shape[1])

        # Normalize f0
        f0_min, f0_max = f0_resampled.min(), f0_resampled.max()
        f0_normalized = (f0_resampled - f0_min) / (f0_max - f0_min + 1e-7)

        # Expand dimensions and concatenate
        f0_expanded = np.expand_dims(f0_normalized, axis=0)
        combined_features = np.concatenate((log_mel, f0_expanded), axis=0)

        return combined_features
    except Exception as e:
        print(f"Error in add_f0_as_additional_channel: {e}")
        raise e

# Function to prepare dataset by adding features and tokenizing labels
def prepare_dataset(batch, feature_extractor, tokenizer):
    audio = batch["audio"]
    audio_array = audio["array"]
    sampling_rate = audio["sampling_rate"]

    # Resample the audio to 16000 Hz if necessary
    if sampling_rate != 16000:
        print(f"Resampling from {sampling_rate} to 16000 Hz")
        audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
        sampling_rate = 16000

    # Ensure audio_array is a NumPy array
    if not isinstance(audio_array, np.ndarray):
        audio_array = np.array(audio_array)
        # print("Converted audio_array to NumPy array in prepare_dataset.")

    # Add f0 features
    batch["input_features"] = add_f0_as_additional_channel(
        audio_array, sampling_rate, feature_extractor
    )
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

# Data collator definition
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Convert input_features to float16 for faster computation if supported
        if torch.cuda.is_available():
            batch["input_features"] = batch["input_features"].half()

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (
            labels[:, 0] == self.processor.tokenizer.bos_token_id
        ).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch
