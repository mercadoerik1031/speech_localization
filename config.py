import numpy as np
import torch


config = {
    "paths": {
        "metadata_path": "../data/metadata.parquet",
        "speech_path": "../data/speech",
        "noise_path": "../data/noise", # "../data/noise" | None
        "save_dir": "../data/preprocessed",
        "model_path": "best_model.pth",
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
        
        "apply_directional_loudness": False,
        "directional_loudness_params": {
            "order_input": 1,
            "t_design_degree": 3,
            "order_output": 1,
            "g_type": ["spherical_cap", "hard"],
            "g_values": None,
            "T_pseudo_floor": 1e-8,
            "backend": "basic",
            "w_pattern": "hypercardioid",
            "use_slepian": True,
            "save_plots": False,
        },
        
        "apply_spherical_rotation": False,
        "spherical_rotation_params": {
            "rotation_angles_rad": (np.pi/6, np.pi/3, np.pi/6), # pi/6 = 30 degrees, pi/3 = 60 degrees, pi/2 = 90 degrees
            "mode": "random", # "single" | "random"
            "num_random_rotations": 4,
            "t_design_degree": 6,
            "order_input": 1,
            "order_output": 1,
            "backend": "basic",
            "w_pattern": "cardioid", # hypercardioid | cardioid | maxre
            "save_plots": True,
        },
    },
    
    
    "model_config": {
        "batch_size": 32,
        "num_workers": 16,
        "use_snn": False,
        "num_steps": 4,
        "dropout_rates": [0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.3, 0.2], # [0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.3, 0.2]
        "thresholds": [1.0] * 8,
        "betas": [0.9] * 8,
        "features": "gcc_hilbert", # "hilbert_transform" | "log_mel_spectrogram" | "gcc_phat" | "active_reactive_intensities" | "gcc_hilbert"
        "encoding_type": "rate", # "rate" or "zero_crossing"
    },
    
    
    "training": {
        "is_lite": True,
        "num_epochs": 100,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "val_size": 0.2,
        "learning_rate": 0.003,
        
        "l2_scheduler": {
            "init": 1e-4,
            "increase_factor": 1.5,
            "decrease_factor": 0.9,
            "threshold": 1.1, # 1.05 - 1.2
            "patience": 3,
            "verbose": False
        },
        
        "lr_scheduler": {
            "t_max": 20, # 10
            "eta_min": 1e-7
        },
        
        "early_stopping": {
            "patience": 7,
            "delta": 0.05
        },
        
        "random_state": 11,
    },
}
