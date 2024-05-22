import torch
import torch.nn as nn
from config import config

class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        
        self.dropout_rates = config["model_config"]["dropout_rates"]
        self.num_features, self.feature_height, self.feature_width = self.determine_feature_dimensions()
        
        # Convolutional layers
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
        )
        
        self._init_feature_size()

        # Bidirectional GRU layers
        self.hidden_size = 256  # 128
        self.gru = nn.GRU(
            input_size=self.feature_size,
            hidden_size=self.hidden_size,
            num_layers=2,  # Number of GRU layers
            bidirectional=True,  # Enable bidirectional GRU
            batch_first=True,
            dropout=self.dropout_rates[3]
        )
        
        # Fully connected layers
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 64),  # Adjust input size to 2 * hidden_size for bidirectional GRU
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_rates[3]),
            
            nn.Linear(64, 2),  # Output layer for azimuth and elevation
        )
        
        self.initialize_weights()
        
    def _init_feature_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.num_features, self.feature_height, self.feature_width)
            dummy_features = self.conv(dummy_input)
            c, h, w = dummy_features.size(1), dummy_features.size(2), dummy_features.size(3)
            self.feature_size = c * w  # Channels * Width

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
        batch_size, channels, height, width = conv_results.size()
        conv_results = conv_results.permute(0, 2, 1, 3).reshape(batch_size, height, -1)  # Reshape to (batch, seq, features)
        gru_out, _ = self.gru(conv_results)
        output = self.mlp(gru_out[:, -1, :])  # Using the last time step for prediction
        return output
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)