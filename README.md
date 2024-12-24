# image-denoising-compression

This repository implements image denoising *and* compression using autoencoders (AE) with both Fully Connected (FC) and Convolutional Neural Network (CNN) architectures, along with Principal Component Analysis (PCA) as a comparative technique. The project explores these methods on both RGB and grayscale versions of the CIFAR-10 dataset (32x32 images), applying noise in two distinct ways: directly to the input image and to the compressed latent representation (bottleneck).

## Overview

This project investigates the effectiveness of autoencoders and PCA for combined image denoising and compression. Autoencoders learn a compressed representation of the input, enabling compression, and their reconstruction capability allows for noise reduction. 
PCA achieves compression by reducing dimensionality and can also be used for denoising by discarding lower-variance components assumed to represent noise. The project compares these techniques across various configurations.

## Key Features

*   **Denoising and Compression:** Implements both functionalities with both AE and PCA.
*   **RGB and Grayscale Images:** Applies methods to both color (RGB) and grayscale versions of CIFAR-10.
*   **Two Noise Application Methods:**
    *   **Noisy Input:** Noise is added directly to the input images before encoding.
    *   **Noisy Bottleneck:** Noise is added to the compressed latent representation (bottleneck) after encoding.
*   **FC and CNN Architectures:** Autoencoders are implemented with both fully connected and convolutional layers.
*   **Visual and Quantitative Evaluation:** Results are presented with visual comparisons (plots of original, noisy, denoised/compressed images) and quantitative metrics (Mean Squared Error - MSE).
*   **CIFAR-10 Dataset:** Uses 32x32 images from the CIFAR-10 dataset.

## Dataset

The project uses the CIFAR-10 dataset, consisting of 60,000 32x32 color images in 10 classes. The dataset is automatically downloaded using `torchvision.datasets.CIFAR10`.

## Model Architectures

*   **Fully Connected (FC) Autoencoders:** Use fully connected layers for encoding and decoding.
*   **Convolutional Neural Network (CNN) Autoencoders:** Use convolutional and transposed convolutional layers.

## Training

Models are trained using MSE loss and the Adam optimizer.

## Testing and Evaluation

The models are tested under four conditions:

1.  **RGB Images, Noisy Input**
2.  **Grayscale Images, Noisy Input**
3.  **RGB Images, Noisy Bottleneck**
4.  **Grayscale Images, Noisy Bottleneck**

For each condition, both FC and CNN autoencoders are evaluated and compared with PCA. Visualizations and MSE values are provided.

## Results

The Jupyter Notebook (`image_denoising_compression.ipynb`) contains detailed results, including:

*   Plots comparing original, noisy, PCA-processed, and autoencoder-processed images for all four test conditions.
*   MSE values for PCA and autoencoders under each condition, quantifying denoising and compression performance.
*   Compression ratios for PCA and autoencoders.

## Bonus Features

*   **Combined Denoising and Compression:** The project explicitly addresses both aspects.
*   **Comprehensive Comparison:** A thorough comparison of AE and PCA across RGB/grayscale images and noise application methods.

![image](https://github.com/user-attachments/assets/364758ec-b9af-4932-b10c-01de7d6155c2)
![image](https://github.com/user-attachments/assets/85e22abb-b169-4134-b9ea-4280c8057906)
