import os
import numpy as np
import matplotlib.pyplot as plt

from config import config
from utils import filter_data, split_data, load_and_preprocess_audio
from feature_extractor import FeatureExtractor

if not os.path.exists('plots'):
    os.makedirs('plots')

def plot_gcc_phat(gcc_phat_features, save_path):
    #print(f"GCC-PHAT features shape: {gcc_phat_features.shape}")
    
    if gcc_phat_features.ndim == 3:
        num_pairs, n_fft, num_frames = gcc_phat_features.shape
    else:
        # Handle the case where gcc_phat_features does not have the expected shape
        print("Unexpected shape for GCC-PHAT features")
        return
    
    plt.figure(figsize=(15, 5))
    for i in range(num_pairs):
        plt.subplot(1, num_pairs, i + 1)
        plt.imshow(gcc_phat_features[i], aspect='auto', origin='lower', 
                   extent=[0, num_frames, -n_fft//2, n_fft//2])
        plt.colorbar(label='Cross-correlation')
        plt.xlabel('Frames')
        plt.ylabel('FFT Bins')
        plt.title(f'GCC-PHAT Channel Pair {i + 1}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_hilbert_transform(signal, analytic_signal, save_path):
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(signal[0], label='Original Signal')
    plt.plot(np.real(analytic_signal[0]), label='Real Part')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('Real Part of Analytic Signal')
    
    plt.subplot(2, 2, 2)
    plt.plot(np.imag(analytic_signal[0]), label='Imaginary Part')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('Imaginary Part of Analytic Signal')
    
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    
    plt.subplot(2, 2, 3)
    plt.plot(amplitude_envelope[0], label='Amplitude Envelope')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('Amplitude Envelope')
    
    plt.subplot(2, 2, 4)
    plt.plot(instantaneous_phase[0], label='Instantaneous Phase')
    plt.xlabel('Samples')
    plt.ylabel('Phase (radians)')
    plt.legend()
    plt.title('Instantaneous Phase')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_short_time_hilbert(stht_foa, phase_foa, save_path):
    num_channels, frame_length, num_frames = stht_foa.shape
    
    plt.figure(figsize=(15, 10))
    for channel in range(num_channels):
        plt.subplot(2, num_channels, channel + 1)
        plt.imshow(np.abs(stht_foa[channel]), aspect='auto', origin='lower', 
                   extent=[0, num_frames, 0, frame_length])
        plt.colorbar(label='Magnitude')
        plt.xlabel('Frames')
        plt.ylabel('Samples')
        plt.title(f'STHT Magnitude Channel {channel + 1}')
        
        plt.subplot(2, num_channels, num_channels + channel + 1)
        plt.imshow(phase_foa[channel], aspect='auto', origin='lower', 
                   extent=[0, num_frames, 0, frame_length])
        plt.colorbar(label='Phase (radians)')
        plt.xlabel('Frames')
        plt.ylabel('Samples')
        plt.title(f'STHT Phase Channel {channel + 1}')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
def plot_gcc_phat_delay(gcc_phat_features, save_path):
    num_pairs, n_fft, num_frames = gcc_phat_features.shape
    
    for i in range(num_pairs):
        plt.figure(figsize=(15, 5))
        delays = np.argmax(gcc_phat_features[i], axis=0) - (n_fft // 2)
        plt.plot(delays, label=f'Channel Pair {i + 1}')
        
        plt.xlabel('Frames')
        plt.ylabel('Delay (samples)')
        plt.legend()
        plt.title(f'Estimated Time Delays from GCC-PHAT - Channel Pair {i + 1}')
        plt.tight_layout()
        plt.savefig(f'{save_path}_channel_pair_{i + 1}.png')
        plt.close()

def main():
    # Load a sample from the dataset
    data = filter_data()
    train_data, _, _ = split_data(data)
    
    feature_extractor = FeatureExtractor()
    
    for i, sample in enumerate(train_data[:5]):  # Just process the first 5 samples for demonstration
        audio_path = sample["speech_path"]
        audio_data = load_and_preprocess_audio(audio_path)
        
        # Plot GCC-PHAT
        gcc_phat_features = feature_extractor.gcc_phat(audio_data)
        plot_gcc_phat(gcc_phat_features, f'plots/gcc_phat_{i}.png')
        
        gcc_phat_features = feature_extractor.gcc_phat(audio_data)
        plot_gcc_phat_delay(gcc_phat_features, 'plots/gcc_phat_delay.png')
        
        # Plot Hilbert Transform
        analytic_signal = feature_extractor.hilbert_transform(audio_data)
        plot_hilbert_transform(audio_data, analytic_signal, f'plots/hilbert_transform_{i}.png')
        
        # Plot Short-Time Hilbert Transform
        stht_foa, phase_foa = feature_extractor.compute_short_time_hilbert(audio_data)
        plot_short_time_hilbert(stht_foa, phase_foa, f'plots/short_time_hilbert_{i}.png')
        
        # You can add more plotting or feature extraction steps as needed

if __name__ == "__main__":
    main()
