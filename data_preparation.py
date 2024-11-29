import pandas as pd
import requests
from datasets import DatasetDict, Dataset
import os
import librosa
from glob import glob
from tqdm import tqdm
import yaml
import argparse

# Function to download audio files
def download_audio(url, save_dir):
    filename = url.split('/')[-1]
    save_path = os.path.join(save_dir, filename)

    if not os.path.exists(save_path):
        response = requests.get(url)
        with open(save_path, 'wb') as f:
            f.write(response.content)
    return save_path

# Function to collect data up to a specified limit
def collect_data(csv_dir, audio_download_dir, data_limit):
    all_data = []
    processed_files = []
    total_count = 0

    # Ensure the audio download directory exists
    os.makedirs(audio_download_dir, exist_ok=True)

    # Process CSV files
    for csv_file in tqdm(glob(os.path.join(csv_dir, '*.csv')), desc="Processing CSV files"):
        file_number = csv_file.split('_')[-1].split('.')[0]

        processed_files.append(file_number)

        # Read CSV file
        df = pd.read_csv(csv_file)

        # Filter data where 'tc_text' length is at least 15 characters
        df_filtered = df[df['tc_text'].str.len() >= 15]

        # Keep necessary columns
        df_filtered = df_filtered[['fi_sound_filepath', 'tc_text']]

        # Download audio files with progress bar
        df_filtered['audio_path'] = list(tqdm(
            df_filtered['fi_sound_filepath'].apply(lambda x: download_audio(x, audio_download_dir)),
            total=len(df_filtered),
            desc=f"Downloading audio for {file_number}"
        ))

        # Append filtered data
        all_data.extend(df_filtered.to_dict(orient='records'))

        # Update total count
        total_count += len(df_filtered)

        # Stop if data limit is reached
        if total_count >= data_limit:
            break

    # Create DataFrame with limited data
    return pd.DataFrame(all_data[:data_limit]), processed_files

# Function to prepare dataset entries
def prepare_dataset_entry(batch):
    # Load audio and resample to 16000 Hz
    audio_array, sampling_rate = librosa.load(batch["audio_path"], sr=16000)

    # Create audio dictionary
    batch["audio"] = {
        "path": batch["audio_path"],
        "array": audio_array,
        "sampling_rate": sampling_rate
    }

    # Store transcript under 'sentence' key
    batch["sentence"] = batch["tc_text"]

    return batch


# Main function to prepare the dataset
def prepare_common_voice_dataset(config):
    csv_dir = config["csv_dir"]
    audio_download_dir = config["audio_download_dir"]
    data_limit = config["data_limit"]

    # Collect data
    df, processed_files = collect_data(csv_dir, audio_download_dir, data_limit)

    # Create dataset from DataFrame
    dataset = Dataset.from_pandas(df)

    # Split dataset into train and test (80% train, 20% test)
    train_test_split = dataset.train_test_split(test_size=0.2)
    common_voice = DatasetDict({
        'train': train_test_split['train'],
        'test': train_test_split['test']
    })

    # Prepare dataset entries
    common_voice = common_voice.map(
        prepare_dataset_entry,
        desc="Preparing dataset",
        load_from_cache_file=False
    )

    # Optionally, save the dataset to disk for reuse
    common_voice.save_to_disk(config.get("prepared_dataset_path", "common_voice_dataset"))

    return common_voice

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Prepare the Common Voice dataset")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file")
    args = parser.parse_args()

    # Load configurations from YAML file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Prepare the dataset
    prepare_common_voice_dataset(config)

if __name__ == "__main__":
    main()
