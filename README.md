# Comparative Analysis of Neural Network Architectures for Sound Source Localization using Spatial LibriSpeech Dataset

## Abstract

Sound source localization (SSL) is a key challenge in auditory scene analysis. SSL focuses on estimating the position of sound sources relative to a reference point, typically a microphone array. This estimation primarily involves determining the sound's direction of arrival (DoA), encompassing both azimuth and elevation angles. SSL is critical in numerous applications, including source separation, automatic speech recognition, speech enhancement, human-robot interaction, noise control, and room acoustic analysis.

Historically, SSL methods have relied on signal processing techniques such as Steered Response Power (SRP) and beamforming. Although these methods are effective in many scenarios, their performance often deteriorates in complex and dynamic acoustic environments, particularly those with high reverberation levels.

Given the limitations of traditional SSL methods and the advancements in deep learning, this project aims to compare Convolutional Neural Networks (CNNs), Convolutional Recurrent Neural Networks (CRNNs), Spiking Neural Networks (SNNs), and Spiking Recurrent Neural Networks (SRNNs) models for speech localization. This comparative analysis builds on existing research, offering insights into the strengths and weaknesses of each model. By leveraging Apple's Spatial LibriSpeech Dataset, this study seeks to advance the understanding of neural network-based approaches for SSL and contribute to developing more effective auditory processing systems.

## Introduction

Sound source localization (SSL) focuses on estimating the position of sound sources relative to a reference point, typically a microphone array. This estimation primarily involves determining the sound's direction of arrival (DoA), encompassing both azimuth and elevation angles. SSL is critical in numerous applications, including source separation, automatic speech recognition, speech enhancement, human-robot interaction, noise control, and room acoustic analysis.

## Related Work

The advent of deep learning has introduced new possibilities for SSL by leveraging the capabilities of neural networks to model complex spatial and temporal patterns in audio signals. CNNs have effectively captured spatial features, while CRNNs combine CNNs' spatial feature extraction capabilities with RNNs' temporal sequence modeling strengths. SNNs and SRNNs offer biologically plausible models for auditory processing, with the latter integrating spiking neurons with recurrent architectures to enhance temporal sequence learning.

## Dataset

### Overview of Spatial LibriSpeech

The Spatial LibriSpeech dataset is a spatially augmented version of the widely used LibriSpeech dataset. It includes over 650 hours of 19-channel audio recordings, first-order ambisonics, and optional distractor noise. Spatial LibriSpeech is designed specifically for machine learning model training and includes comprehensive labels for source position, speaking direction, room acoustics, and geometry.

### First-Order Ambisonics

First-order ambisonics is a technique used to capture and reproduce spatial sound. It uses a set of spherical harmonics to encode the sound field, providing a full-sphere surround sound experience. Unlike traditional stereo or surround sound formats, ambisonics can represent sound from any direction, including above and below the listener. This makes it particularly useful for applications requiring accurate spatial audio reproduction, such as virtual reality, 3D audio for gaming, and spatial audio research.

### Using the Lite Version of Spatial LibriSpeech

For this project, the lite version of the Spatial LibriSpeech dataset was utilized, which constitutes 10% of the original dataset size. This lite version maintains the diversity and richness of the full dataset while significantly reducing computational requirements. The lite version includes a proportional representation of the acoustic conditions, ensuring the training and evaluation processes remain robust and comprehensive.

## Feature Extraction

Feature extraction is a crucial step in developing models for speech localization. This project utilizes two advanced feature extraction techniques: GCC-PHAT and Short-Time Hilbert Transform (STHT). Each technique captures different aspects of the audio signals, providing complementary information that enhances the performance of the localization models.

### Generalized Cross-Correlation with Phase Transform (GCC-PHAT)

The Generalized Cross-Correlation with Phase Transform (GCC-PHAT) is a widely used method for estimating the Time Difference of Arrival (TDOA) between pairs of microphones. This method is particularly effective in reverberant and noisy environments.

### Short-Time Hilbert Transform (STHT) and Instantaneous Phase

The Short-Time Hilbert Transform (STHT) provides a way to analyze the instantaneous amplitude and phase of an audio signal over short time windows. The STHT is applied to short audio signal frames, resulting in a time-varying representation of the signal's phase. The instantaneous phase captures the fine temporal structure of the audio signal, which is sensitive to the direction of arrival.

## Models

In this project, I utilized four different neural network architectures: CNN, CRNN, SNN, and SRNN. Each of these models brings unique capabilities to the task of speech localization.

### Common Structure

All the models share a common backbone architecture consisting of three convolutional layers followed by fully connected layers. This structure is designed to capture spatial and temporal features.

### Model-Specific Features and Differences

#### Convolutional Neural Network (CNN), Spiking Neural Network (SNN), and Spiking Recurrent Neural Network (SRNN)

The CNN, SNN, and SRNN models all share the common structure described above. The primary differences lie in the types of neurons used and the inclusion of recurrent connections in the SRNN.

#### Convolutional Recurrent Neural Network (CRNN)

The CRNN extends the CNN by adding Gated Recurrent Units (GRUs) to capture temporal dependencies. The GRU layers process the reshaped convolutional features to capture temporal dependencies essential for accurately determining the DoA of sound sources.

## Training Setup and Optimization

This project's training setup and optimization process are designed to leverage available computing resources effectively, ensuring efficient training and evaluation of the models. This section outlines the configuration details, optimization techniques, loss functions, and learning rate scheduling strategies employed.

### Training Environment

All models were trained on a system with the following specifications:

- **CPU**: 16 vCPUs, 60 GB RAM, Intel Skylake
- **GPU**: NVIDIA Tesla T4

### Optimization Techniques

The optimization process utilizes the Adam optimizer, which is well-suited for training deep neural networks due to its adaptive learning rate capabilities.

### Loss Function

The Mean Squared Error (MSE) loss function measures the difference between the predicted and true values of the sound source directions.

### Learning Rate Scheduler

A Cosine Annealing Learning Rate (LR) scheduler adjusts the learning rate during training, ensuring smooth and gradual reduction in learning rate.

### Early Stopping

To prevent overfitting and unnecessary computation, early stopping is implemented, monitoring the validation loss and stopping training if there is no improvement after a specified number of epochs.

## Evaluation Metrics

Evaluating the performance of speech localization models requires accurate and reliable metrics that reflect the models' ability to predict the direction of sound sources. In this project, angular errors are used as the primary evaluation metric.

### Conversion to Cartesian Coordinates

The first step in evaluating the angular error is to convert the azimuth and elevation angles into Cartesian coordinates.

### Angular Distance Calculation

The angular distance is computed using the dot product of the normalized vectors, followed by calculating the arccosine of the result.

### Median Angular Error Calculation

The median angular error provides a robust measure of the average performance of the model, mitigating the effect of outliers.

## Results

### Performance Metrics

The table below summarizes the performance of each model in terms of median angular error, average inference time per batch, and the number of parameters:

| Model | Loss | Median Error (degrees) | Avg Inference Time (s) | Number of Parameters | Avg Time Per Epochs (min:sec) | Epochs Trained |
|-------|------|-------------------------|------------------------|----------------------|-------------------------------|----------------|
| CNN   | 0.1186 | 20.47                  | 0.0094                 | 271,016,696          | 0.0                           | 0.0            |
| CRNN  | 0.1408 | 21.37                  | 0.0051                 | 1,573,418            | 3:49                          | 24             |
| SNN   | 0.6580 | 32.76                  | 0.0417                 | 271,000,422          | 0.0                           | 0.0            |
| SRNN  | 0.5690 | 32.06                  | 0.0403                 | 271,000,430          | 0.0                           | 0.0            |

### Visualizations

Visualizations of the predictions versus the ground truth for selected samples help to understand the spatial localization performance of the models.

### Analysis of Results

The results indicate that:

- **CNN Model**: Achieved the lowest median error (20.47 degrees) with a moderate inference time (0.0094 seconds per batch) but has the highest number of parameters.
- **CRNN Model**: Performed well with a median error of 21.37 degrees and the lowest average inference time (0.0051 seconds per batch), making it a more lightweight and efficient model compared to the CNN.
- **SNN and SRNN Models**: Exhibited higher median errors and longer inference times, with performance not as competitive as the CNN and CRNN models.

The visualizations demonstrate that the CNN and CRNN models generally predict positions closer to the ground truth compared to the SNN and SRNN models.

## References

1. Chiariotti, Federico, et al. "Sound source localization using signal processing techniques."
2. Chakrabarty, Sanket, and Emanuël AP Habets. "Multi-microphone speaker localization using convolutional neural networks."
3. Grondin, François, et al. "Sound event detection and localization using a convolutional recurrent neural network."
4. Tavanaei, Amir, and Anthony Maida. "Spiking neural networks for speech recognition."
5. Yuan, Zhe, et al. "Motion prediction using spiking recurrent neural networks."
6. Spatial LibriSpeech dataset. "Spatially augmented version of the LibriSpeech dataset."
7. Mazzon, Giancarlo, et al. "Order and spatial resolution of ambisonics."
8. Cao, Yuchen, et al. "Generalized cross-correlation with phase transform for time difference of arrival estimation."
9. Haghighatshoar, Saeid, et al. "Low-power sound event detection using short-time Hilbert transform."
