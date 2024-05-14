from config import config

import numpy as np
import librosa
from scipy.signal import hilbert

class FeatureExtractor():
    def __init__(self):
        self.fs = config["audio_processing"]["sr"]
        self.n_fft = config["audio_processing"]["n_fft"]
        self.frame_length = config["audio_processing"]["stht_frames"]
        self.hop_length = config["audio_processing"]["hop_length"]
        self.mel_bins = config["audio_processing"]["mel_bins"]
        self.fmin = config["audio_processing"]["f_min"]
        self.window = config["audio_processing"]["window"]
        self.features = config["model_config"]["features"]
        if "log_mel_spectrogram" in self.features:
            self.mel_filter = librosa.filters.mel(sr=self.fs, n_fft=self.n_fft, n_mels=self.mel_bins, fmin=self.fmin)

    def log_mel_spectrogram(self, audio):
        # Assuming audio is of shape [channels, samples]
        channel_features = []
        for i in range(audio.shape[0]):  # iterate over each channel
            S = np.abs(librosa.stft(y=audio[i], 
                                    n_fft=self.n_fft, 
                                    hop_length=self.hop_length, 
                                    window=self.window,
                                    pad_mode='reflect')) ** 2
            mel_spectrogram = np.dot(self.mel_filter, S).T
            log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=1.0, amin=1e-10, top_db=None)
            channel_features.append(log_mel_spectrogram)
        # Stack along a new dimension to keep channels separate
        log_mel_spectrograms = np.stack(channel_features, axis=0)
        return log_mel_spectrograms
    
    
    def gcc_phat(self, audio):
        gcc_phat_features = []
        for i in range(audio.shape[0]):
            for j in range(i + 1, audio.shape[0]):
                px = librosa.stft(y=audio[i], n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, pad_mode='reflect')
                ref_px = librosa.stft(y=audio[j], n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, pad_mode='reflect')
                r = px * np.conj(ref_px)
                spec = np.exp(1.j * np.angle(r))  # Calculate phase
                cc = np.fft.irfft(spec, n=self.n_fft, axis=0)  # Inverse FFT
                cc = np.fft.fftshift(cc, axes=0)  # Shift zero frequency to center
                gcc_phat_features.append(cc)  # Collect all channel pairs
        
        gcc_phat_features = np.stack(gcc_phat_features, axis=0)  # [channel_pairs, n_fft, frames]
        return gcc_phat_features
    
    def hilbert_transform(self, signal):
        """Apply the Hilbert transform to each channel of the signal."""
        # Apply along the last axis to support multi-channel input
        analytic_signal = np.apply_along_axis(hilbert, axis=-1, arr=signal)
        return analytic_signal

    def compute_short_time_hilbert(self, signal):
        """Compute the Short-Time Hilbert Transform of a multichannel signal."""
        num_channels, num_samples = signal.shape
        num_frames = 1 + int(np.floor((num_samples - self.frame_length) / self.hop_length))
        pad_length = int((num_frames - 1) * self.hop_length + self.frame_length)
        pad_signal = np.pad(signal, ((0, 0), (0, max(0, pad_length - num_samples))), 'constant')

        stht_foa = np.empty((num_channels, self.frame_length, num_frames), dtype=complex)
        phase_foa = np.empty((num_channels, self.frame_length, num_frames), dtype=float)
        window = librosa.filters.get_window(self.window, self.frame_length)

        for channel in range(num_channels):
            for i in range(num_frames):
                start = i * self.hop_length
                end = start + self.frame_length
                frame = pad_signal[channel, start:end] * window
                stht_foa[channel, :, i] = self.hilbert_transform(frame.reshape(1, -1))
                phase_foa[channel, :, i] = np.angle(stht_foa[channel, :, i])

        return stht_foa, phase_foa
    
    
    def intensities(self, audio):
        # Ensure audio has the correct shape [channels, samples]
        if audio.shape[0] != 4:
            raise ValueError("Audio must have exactly 4 channels for FOA intensity calculations.")

        stft = librosa.stft(y=audio, 
                            n_fft=self.n_fft, 
                            hop_length=self.hop_length, 
                            window=self.window, 
                            pad_mode='reflect')
        
        # Constants for intensity calculations
        # Assuming `rho` (air density) and `c` (speed of sound) need to be defined
        rho = config.get('rho', 1.21)  # example value in kg/m^3
        c = config.get('c', 343)      # example value in m/s
        normalization_factor = -1 / (rho * c * np.sqrt(3))
        
        # Decompose the STFT output to get pressure and particle velocity components
        p = stft[0, :]  # pressure (W channel)
        vx = stft[1, :] * normalization_factor  # particle velocity x (X channel)
        vy = stft[2, :] * normalization_factor  # particle velocity y (Y channel)
        vz = stft[3, :] * normalization_factor  # particle velocity z (Z channel)
        
        # Calculate the conjugate of pressure
        p_star = np.conj(p)
        
        # Calculate active intensities
        Ia_x = np.real(p_star * vx)
        Ia_y = np.real(p_star * vy)
        Ia_z = np.real(p_star * vz)
        
        # Calculate reactive intensities
        Ir_x = np.imag(p_star * vx)
        Ir_y = np.imag(p_star * vy)
        Ir_z = np.imag(p_star * vz)
        
        # Stack active and reactive intensity components
        Ia = np.stack((Ia_x, Ia_y, Ia_z), axis=0)
        Ir = np.stack((Ir_x, Ir_y, Ir_z), axis=0)
        
        # Concatenate active and reactive intensities
        intensities = np.concatenate((Ia, Ir), axis=0)
        return intensities
    
    def combine_features(self, audio):
        # Extract features from the same audio signal
        gcc_features = self.gcc_phat(audio)
        _, hilbert_features = self.compute_short_time_hilbert(audio)  # Assuming this returns phase directly

        # Determine the target size for alignment
        target_shape = min(gcc_features.shape[2], hilbert_features.shape[2])

        # Align features by trimming or padding to the target size
        if gcc_features.shape[2] > target_shape:
            gcc_features = gcc_features[:, :, :target_shape]
        elif gcc_features.shape[2] < target_shape:
            padding_amount = target_shape - gcc_features.shape[2]
            gcc_features = np.pad(gcc_features, ((0, 0), (0, 0), (0, padding_amount)), mode='constant')

        if hilbert_features.shape[2] > target_shape:
            hilbert_features = hilbert_features[:, :, :target_shape]
        elif hilbert_features.shape[2] < target_shape:
            padding_amount = target_shape - hilbert_features.shape[2]
            hilbert_features = np.pad(hilbert_features, ((0, 0), (0, 0), (0, padding_amount)), mode='constant')

        # Stack the aligned features along a new axis to concatenate them
        combined_features = np.concatenate([gcc_features, hilbert_features], axis=0)
        return combined_features
    
    def normalize(self, features):
        """
        Normalize features by subtracting the mean and dividing by the standard deviation.
        """
        if features.ndim == 3:
            mean = np.mean(features, axis=(1, 2), keepdims=True)
            std = np.std(features, axis=(1, 2), keepdims=True)
        elif features.ndim == 2:
            mean = np.mean(features, axis=1, keepdims=True)
            std = np.std(features, axis=1, keepdims=True)
        else:
            raise ValueError("Unsupported number of dimensions: {}".format(features.ndim))

        normalized_features = (features - mean) / (std + 1e-6)
        return normalized_features

    def transform(self, audio):
        features = []
        if "gcc_phat" in self.features:
            gcc = self.gcc_phat(audio)
            features.append(self.normalize(gcc))
            
        if "hilbert_transform" in self.features:
            stht, phase = self.compute_short_time_hilbert(audio)
            features.append(self.normalize(phase))
            
        if "log_mel_spectrogram" in self.features:
            log_mel = self.log_mel_spectrogram(audio)
            features.append(self.normalize(log_mel))
        
        if "active_reactive_intensities" in self.features:
            intensity_features = self.intensities(audio)
            features.append(self.normalize(intensity_features))
            
        if "gcc_hilbert" in self.features:
            gcc_hil = self.combine_features(audio)
            features.append(self.normalize(gcc_hil))
            

        return np.concatenate(features, axis=0)
