import numpy as np
import torch


config = {
    "env":{
        "ip": "10.182.0.5",
        "port": "49152"},
    
    "paths": {
        "metadata_path": "../data/metadata.parquet",
        "speech_path": "../data/speech",
        "noise_path": "../data/noise", # "../data/noise" | None
        "save_dir": "../data/preprocessed",
        "model_path": "../pretrained_models/cnn_gcc_hilbert_20_46.pth", # srnn_gcc_hilbert_32_58.pth ../pretrained_models/cnn_gcc_hilbert_20_46.pth
    },
    
    
    "audio_processing": {
        "sr": 16000,
        "start": 0,
        "duration": 2,
        "noise_ratio": 1,
        "n_fft": 512,
        "stht_frames": 512,
        "hop_length": 256,
        "mel_bins": 128,
        "f_min": 50,
        "window": "hann",
    },
    
    
    "data_augmentation": {
        "apply_directional_loudness": True,
        "directional_loudness_params": {
            "order_input": 1,
            "t_design_degree": 6,
            "order_output": 1,
        },
    },
    
    
    "model_config": {
        "batch_size": 32,
        "num_workers": 16,
        "use_snn": False,
        "model_type": "CNN", # "CNN", "CRNN", "SNN", "SRNN"
        "num_steps": 4,
        "dropout_rates": [0.3, 0.25, 0.22, 0.6, 0.5, 0.55, 0.35, 0.2], # [0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.3, 0.2]
        "thresholds": [1.0] * 8,
        "betas": [0.9] * 8,
        "features": "gcc_hilbert", # "hilbert_transform" | "log_mel_spectrogram" | "gcc_phat" | "active_reactive_intensities" | "gcc_hilbert"
        "encoding_type": "rate", # "rate" | None   
    },
    
    
    "training": {
        "is_lite": True,
        "num_epochs": 100,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "val_size": 0.2,
        "learning_rate": 0.003,
        "l2": 1e-4,
        
        "lr_scheduler": {
            "t_max": 20,
            "eta_min": 1e-7
        },
        
        "early_stopping": {
            "patience": 7,
            "delta": 0.05
        },
        
        "random_state": 11,
    },
}
