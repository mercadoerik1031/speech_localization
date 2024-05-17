import torch
from torch.utils.data import Dataset, DataLoader, Subset
from snntorch import spikegen

from config import config
from utils import load_and_preprocess_audio
from feature_extractor import FeatureExtractor
from data_augmentation import DirectionalLoudness
import numpy as np

class AudioDataset(Dataset):
    def __init__(self, data, apply_augmentations=False):
        self.data = data
        self.apply_augmentations = apply_augmentations
        
        # Setup from config
        self.setup_from_config()
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor()
        
        # Augmentations
        self.setup_augmentations()
        
    
    def setup_from_config(self):
        self.sr = config["audio_processing"]["sr"]
        self.start = config["audio_processing"]["start"]
        self.duration = config["audio_processing"]["duration"]
        self.noise_ratio = config["audio_processing"]["noise_ratio"]
        self.use_snn = config["model_config"]["use_snn"]
        self.num_steps = config["model_config"]["num_steps"]
        self.encoding_type = config["model_config"]["encoding_type"]
    
    def setup_augmentations(self):
        if self.apply_augmentations:
            self.directional_loudness_transform = DirectionalLoudness() if config["data_augmentation"]["apply_directional_loudness"] else None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        speech_audio = load_and_preprocess_audio(item["speech_path"], self.sr, self.start, self.duration)
        
        if item.get("noise_path") and item["split"] == "train":
            noise_audio = load_and_preprocess_audio(item["noise_path"], self.sr, self.start, self.duration)
            mixed_audio = speech_audio + self.noise_ratio * noise_audio
        else:
            mixed_audio = speech_audio
            
        if self.apply_augmentations and self.directional_loudness_transform:
            mixed_audio = self.directional_loudness_transform.forward(mixed_audio)
            azimuth, elevation = self.directional_loudness_transform.transform_labels(item['azimuth'], item['elevation'])
            labels_array = np.stack([azimuth, elevation], axis=-1)
        else:
            labels_array = np.stack([item['azimuth'], item['elevation']], axis=-1)
            
                    
        features = self.feature_extractor.transform(mixed_audio)
        norm_features = self.feature_extractor.normalize(features)
        
        if self.use_snn and self.encoding_type == "rate":
            norm_features = torch.from_numpy(norm_features).float()
            features_torch = spikegen.rate(norm_features, num_steps=self.num_steps)
        else:
            features_torch = torch.from_numpy(norm_features).float()
            
        # Ensure the labels_array has the shape (2,) and convert to tensor
        labels_tensor = torch.from_numpy(labels_array).float().view(2)
        
        return features_torch, labels_tensor


def get_preloaded_data_loaders(data, apply_augmentations, shuffle, batch_size=config["model_config"]["batch_size"], rank=0, world_size=1):
    dataset = AudioDataset(
        data=data, 
        apply_augmentations=apply_augmentations,
    )
    dataset_size = len(dataset)
    partition_size = dataset_size // world_size
    lower_bound = rank * partition_size
    upper_bound = lower_bound + partition_size if rank != world_size - 1 else dataset_size
    
    # Create a subset of the dataset based on the rank
    subset_data = Subset(dataset, range(lower_bound, upper_bound))
    
    # Create the DataLoader
    loader = DataLoader(
        subset_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config["model_config"]["num_workers"],  # Adjust as per your system's capabilities
        pin_memory=True,
        drop_last=True
    )
    
    return loader