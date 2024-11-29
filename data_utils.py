import numpy as np
import librosa
from scipy.signal import resample
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch


# Function to combine f0 and log-mel spectrogram
def add_f0_as_additional_channel(audio, sampling_rate, feature_extractor):
    log_mel = feature_extractor(audio, sampling_rate=sampling_rate).input_features[0]
    f0, _, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        frame_length=2048,
        hop_length=512,
    )
    f0 = np.nan_to_num(f0)
    f0_resampled = resample(f0, log_mel.shape[1])
    f0_min, f0_max = f0_resampled.min(), f0_resampled.max()
    f0_normalized = (f0_resampled - f0_min) / (f0_max - f0_min + 1e-7)
    f0_expanded = np.expand_dims(f0_normalized, axis=0)
    combined_features = np.concatenate((log_mel, f0_expanded), axis=0)
    return combined_features


# Function to prepare dataset by adding features and tokenizing labels
def prepare_dataset(batch, feature_extractor, tokenizer):
    audio = batch["audio"]
    batch["input_features"] = add_f0_as_additional_channel(
        audio["array"], audio["sampling_rate"], feature_extractor
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
