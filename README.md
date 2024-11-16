# synthetic-uv

This repo contains the script to generate synthetic visibility (Fourier domain) from DSA-2000 simulated gravitational lensing image data generated by [POLISH (Connor et al. 2022)](https://github.com/liamconnor/polish-pub/tree/main).
---

# Get started

## Step 1: Generate POLISH data

First, generate the data to using [`make_img_pairs.py` provided by POLISH](https://github.com/liamconnor/polish-pub/blob/main/make_img_pairs.py) with the root directory of this repo to be the output.

```
python make_img_pairs.py -o /path/to/this/repo/ -k psf/dsa-2000-fullband-psf.fits -s 512
```

It should then create the following subdirectories in the repo root directory:

- `POLISH_train_HR`: True sky image (noise-free, full resolution) for training set.
- `POLISH_train_HR_Noise`: `POLISH_train_HR` but with noise. Used to generated `observed visibility` that includes `n_sky`.
- `POLISH_valid_HR`: True sky image (noise-free, full resolution) for validation set.
- `POLISH_valid_HR_Noise`: `POLISH_valid_HR` but with noise. Used to generated `observed visibility` that includes `n_sky`.
- `POLISH_train_LR_bicubic`: Low-resolution dirty images used for POLISH training input. This is unused for our purposes.
- `POLISH_valid_LR_bicubic`: Low-resolution dirty images used for POLISH validation input. This is unused for our purposes.
- `psf`: Stores the ideal PSF. This is unused for our purposes.

## Step 2: Download pre-generated `synthetic_data` (Alternatively: You can generate your own using the provided `data_processing.ipynb`. WARNING: Will take a long time (6 hrs) and ~100TB of space!)

Download the pre-generated `synthetic_data` directory (LINK TBD) and extract to the root directory of this repo. It should have the following structure:

```
synthetic_data/
├── train/
│   ├── freq_10MHz/
│   │   ├── antennas_16_s1/
│   │   └── ...
│   └── ...
├── valid/
│   ├── freq_10MHz/
│   │   ├── antennas_16_s1/
│   │   └── ...
│   └── ...
```

Further documentation for `synthetic_data` is in a following section.

## Step 3: Done!

The directory structure of this repo can be seen below:
```
.
├── data_processing.ipynb
├── POLISH_train_HR
│   ├── 0000.npy
│   ├── ...
│   └── 0799.npy
├── POLISH_train_HR_Noise
│   ├── 0000_noise.npy
│   ├── ...
│   └── 0799_noise.npy
├── POLISH_train_LR_bicubic
│   └── X2
├── POLISH_valid_HR
│   ├── 0800.npy
│   ├── ...
│   └── 0899.npy
├── POLISH_valid_HR_Noise
│   ├── 0800_noise.npy
│   ├── ...
│   └── 0899_noise.npy
├── POLISH_valid_LR_bicubic
│   └── X2
├── psf
│   └── psf_ideal.npy
├── synthetic_data
│   ├── train
│   ├── uv
│   └── valid
```
---
# Documentation for Synthetic Data Directory Structure

This document explains the structure of the `synthetic_data` directory generated for training and validation datasets. It describes the folder hierarchy, contents, and naming conventions, making it easy for others to navigate and understand the data.

---

## Overview
The `synthetic_data` directory contains training (`train`) and validation (`valid`) datasets for synthetic visibility simulations. Data is organized by:
1. **Dataset type**: Training or validation.
2. **Frequency**: The observing frequency in MHz.
3. **Antenna count**: The number of antennas used for observations.
4. **Scaling factors**: Downscaling factors applied to images and visibilities.

---

## Directory Structure
```
synthetic_data/
├── train/
│   ├── freq_10MHz/
│   │   ├── antennas_16_s1/
│   │   ├── antennas_16_s2/
│   │   ├── antennas_32_s1/
│   │   ├── antennas_32_s2/
│   │   └── ...
│   ├── freq_700MHz/
│   ├── freq_1300MHz/
│   ├── freq_2000MHz/
│   └── ...
├── valid/
│   ├── freq_10MHz/
│   │   ├── antennas_16_s1/
│   │   ├── antennas_16_s2/
│   │   ├── antennas_32_s1/
│   │   ├── antennas_32_s2/
│   │   └── ...
│   ├── freq_700MHz/
│   ├── freq_1300MHz/
│   ├── freq_2000MHz/
│   └── ...
```
---

## Explanation of Directory Components

### 1. **Top-Level Folders**
- `train/`: Contains synthetic data for the training dataset.
- `valid/`: Contains synthetic data for the validation dataset.

### 2. **Frequency Subfolders**
Each dataset (`train` or `valid`) is divided into subfolders by the observing frequency:
- Example: `freq_10MHz`, `freq_700MHz`, `freq_1300MHz`, and `freq_2000MHz`.

These correspond to:
- 10 MHz
- 700 MHz
- 1.3 GHz
- 2 GHz

### 3. **Antenna Subfolders**
Inside each frequency folder, data is divided by the number of antennas and the scaling factor:
- Example: `antennas_16_s1` represents 16 antennas with no downscaling.
- `_s1`: No downscaling applied.
- `_s2`: Downscaled by a factor of 2.

Other examples:
- `antennas_32_s1`: 32 antennas, no downscaling.
- `antennas_2048_s2`: 2048 antennas, downscaled by 2.

---

## File Naming and Contents

Each subfolder (e.g., `antennas_16_s1`) contains six files per image, corresponding to different outputs:

| **File Name**                     | **Description**                                                                 |
|------------------------------------|---------------------------------------------------------------------------------|
| `fft_full.npy`                     | Full observed noise-free visibility (Fourier transform of the true sky).        |
| `masked_fft_full.npy`              | Sampled noise-free visibility (Fourier transform after applying the UV mask).   |
| `dirty_image_full.npy`             | Dirty image reconstructed from sampled noise-free visibility.                   |
| `noisy_fft_full.npy`               | Full observed noisy visibility (Fourier transform of the noisy true sky).       |
| `noisy_masked_fft_full.npy`        | Sampled noisy visibility.                                                      |
| `noisy_dirty_image_full.npy`       | Dirty image reconstructed from sampled noisy visibility.                        |

---

## How to Access Data

### Example: Accessing 16 Antennas at 2 GHz
The folder path:
synthetic_data/train/freq_2000MHz/antennas_16_s1/

Files in this folder:
- `fft_full.npy`: Full noise-free visibility.
- `masked_fft_full.npy`: Masked noise-free visibility.
- `dirty_image_full.npy`: Dirty image from noise-free visibility.
- `noisy_fft_full.npy`: Full noisy visibility.
- `noisy_masked_fft_full.npy`: Masked noisy visibility.
- `noisy_dirty_image_full.npy`: Dirty image from noisy visibility.

### Example: Accessing Validation Data
For 2048 antennas at 700 MHz, the folder path:
```
synthetic_data/valid/freq_700MHz/antennas_2048_s2/
```
---

## Usage Notes

### 1. **Adding New Frequencies or Antennas**
To include additional frequencies or antennas:
1. Add the corresponding frequency or antenna count to the generation script.
2. Regenerate the synthetic data.

### 2. **Zipping Specific Folders**
To zip only a specific frequency and antenna:
```
zip -r output.zip synthetic_data/train/freq_2000MHz/antennas_2048_s1 synthetic_data/valid/freq_2000MHz/antennas_2048_s1
```

---

## FAQ

### 1. **What is `_s1` and `_s2`?**
   - `_s1` refers to data with no downscaling.
   - `_s2` refers to data downscaled by a factor of 2.

### 2. **What is the difference between `fft_full.npy` and `masked_fft_full.npy`?**
   - `fft_full.npy`: Contains the full Fourier transform of the image without any sampling.
   - `masked_fft_full.npy`: Contains the Fourier transform after applying the UV sampling mask.

### 3. **How is the data generated?**
   - Data is generated using Python scripts that process true sky images, apply UV sampling masks, and produce outputs (FFT, masked FFT, dirty images).

### 4. **What tools can I use to view `.npy` files?**
   - Use Python with NumPy:
     ```
     import numpy as np
     data = np.load("path_to_file.npy")
     print(data.shape)
     ```
   - Visualize images using `matplotlib`:
     ```
     import matplotlib.pyplot as plt
     plt.imshow(data, cmap="gray")
     plt.show()
     ```

---
