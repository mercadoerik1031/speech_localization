from config import config

import os
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import train_test_split

def filter_data():
    paths = config["paths"]
    metadata = pd.read_parquet(paths["metadata_path"], engine="pyarrow")
    if config["training"]["is_lite"]: 
        metadata = metadata[metadata["lite_version"] == True]

    data = []
    for _, row in metadata.iterrows():
        sample = {
            "sample_id": row["sample_id"],
            "speech_path": os.path.join(paths["speech_path"], f"{row['sample_id']:06}.flac"),
            "noise_path": os.path.join(paths["noise_path"], f"{row['sample_id']:06}.flac") if paths["noise_path"] else None,
            "azimuth": row["speech/azimuth"],
            "elevation": row["speech/elevation"],
            "split": row["split"]
        }
        data.append(sample)
    return data

def split_data(data):
    val_size = config["training"]["val_size"]
    seed = config["training"]["random_state"]
    train_data = [sample for sample in data if sample["split"] == "train"]
    test_data = [sample for sample in data if sample["split"] == "test"]
    train_data, val_data = train_test_split(train_data, test_size=val_size, random_state=seed)
    return train_data, val_data, test_data

def load_and_preprocess_audio(file_path, sr=config["audio_processing"]["sr"], 
                              start=config["audio_processing"]["start"], 
                              duration=config["audio_processing"]["duration"]):

    # Calculate start and stop samples
    start_sample = int(start * sr)
    stop_sample = int(start_sample + duration * sr)

    # Load the specified segment of the audio file
    try:
        audio, file_sr = sf.read(file_path, start=start_sample, stop=stop_sample, dtype='float32')
        audio = audio.T  # Transpose to [channels, samples]

        # Padding if necessary
        if audio.shape[1] < stop_sample - start_sample:
            padding_size = stop_sample - start_sample - audio.shape[1]
            audio = np.pad(audio, ((0, 0), (0, padding_size)), mode='constant', constant_values=0)

        # Normalization
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio /= max_val
        return audio

    except Exception as e:
        print(f"Failed to load {file_path}: {str(e)}")
        raise e  # Optionally re-raise the exception after logging it