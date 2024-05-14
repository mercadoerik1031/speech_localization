import torch
import torch.nn as nn
from config import config

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.dropout_rates = config["model_config"]["dropout_rates"]
        self.num_features, self.feature_height, self.feature_width = self.determine_feature_dimensions()
        
        self.conv = nn.Sequential(
            nn.Conv2d(self.num_features, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(self.dropout_rates[0]),
            
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(self.dropout_rates[1]),
            
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(self.dropout_rates[2]),
            
            nn.Flatten() # Flattened size=Channels×Height×Width=96×64×15=92160
        )
        
        self._init_feature_size()
        
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size//32),
            nn.BatchNorm1d(self.feature_size//32),
            nn.ReLU(),
            nn.Dropout(self.dropout_rates[3]),
            
            nn.Linear(self.feature_size//32, self.feature_size//64),
            nn.BatchNorm1d(self.feature_size//64),
            nn.ReLU(),
            nn.Dropout(self.dropout_rates[3]),
            
            nn.Linear(self.feature_size//64, self.feature_size//128),
            nn.BatchNorm1d(self.feature_size//128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rates[3]),
            
            nn.Linear(self.feature_size//128, self.feature_size//256),
            nn.BatchNorm1d(self.feature_size//256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rates[3]),
            
            nn.Linear(self.feature_size//256, self.feature_size//512),
            nn.BatchNorm1d(self.feature_size//512),
            nn.ReLU(),
            nn.Dropout(self.dropout_rates[3]),
            
            nn.Linear(self.feature_size//512, (self.feature_size//512)//2),
            nn.BatchNorm1d((self.feature_size//512)//2),
            nn.ReLU(),
            nn.Linear((self.feature_size//512)//2, 2),  # Output layer for azimuth and elevation
        )
        
        self.initialize_weights()
        
    def _init_feature_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.num_features, self.feature_height, self.feature_width)
            dummy_features = self.conv(dummy_input)
            self.feature_size = dummy_features.shape[-1]
            
    def determine_feature_dimensions(self):
        feature_dims = {
            "log_mel_spectrogram": (4, 126, 128),
            "gcc_phat": (6, 512, 126),
            "hilbert_transform": (4, 512, 124),
            "active_reactive_intensities": (6, 257, 126),
            "gcc_hilbert": (10, 512, 124)
        }
        selected_feature = config["model_config"]["features"]
        return feature_dims.get(selected_feature, (1, 64, 64))
    
    def forward(self, x):
        conv_results = self.conv(x)
        output = self.mlp(conv_results)
        return output
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)