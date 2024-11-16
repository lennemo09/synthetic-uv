# Import and initialize stuff

Additionally, here is the baseline DSA-2000 specs from the [DSA-2000 white paper](https://sites.astro.caltech.edu/~vikram/DSA_2000__Specifications.pdf):

| **Quantity**                          | **Value**                                                                 |
|---------------------------------------|---------------------------------------------------------------------------|
| **Reflectors**                        | 2000 × 5-m dishes                                                        |
| **Frequency Coverage**                | 0.7–2 GHz                                                                |
| **Bandwidth**                         | 1.3 GHz                                                                  |
| **Field of View**                     | 10.6 deg²                                                                |
| **Spatial Resolution**                | 3.5″                                                                     |
| **System Temperature**                | 25 K                                                                     |
| **Aperture Efficiency**               | 70%                                                                      |
| **System-equivalent flux density (SEFD)** | 2.5 Jy                                                                 |
| **Survey Speed Figure of Merit**      | 1.3 × 10⁷ deg² m⁴ K⁻²                                                   |
| **Continuum Sensitivity (1 hour)**    | 1 μJy                                                                    |
| **All-Sky Survey (per epoch)**        | 30,000 deg² @ 2 μJy/bm                                                  |
| **All-Sky Survey (combined)**         | 30,000 deg² @ 500 nJy/beam                                              |
| **Pulsar Timing Fields**              | 2000 deg² @ 200 nJy/beam                                                |
| **Deep Drilling Fields**              | 30 deg² @ 100 nJy/beam                                                  |
| **Brightness Temperature (1σ)**       | 5 mK                                                                     |
| **Number of Unique Sources**          | > 1 billion                                                              |

*Sensitivity calculations assume 65% usable bandwidth and 20% time loss to overheads. The field of view assumes a standard illumination taper.*


```python
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.metrics import peak_signal_noise_ratio as psnr 
from concurrent.futures import ThreadPoolExecutor
# from scipy.ndimage import zoom
from skimage.transform import rescale
import time

np.random.seed(2048)
DSA_2000_ANTENNA_DIAMETER_KM = 5e-3 # km

DSA_2000_MAX_RADIUS_KM = 15

DSA_2000_MIN_FREQUENCY_GHZ = 0.7
DSA_2000_MAX_FREQUENCY_GHZ = 2

DSA_2000_MIN_FREQUENCY_HZ = 0.7e9
DSA_2000_MAX_FREQUENCY_HZ = 2e9

DSA_2000_NPIX = 1024

c = 299792 # km/s

DSA_2000_MIN_WAVELENGTH_KM = c / DSA_2000_MAX_FREQUENCY_HZ
DSA_2000_MAX_WAVELENGTH_KM = c / DSA_2000_MIN_FREQUENCY_HZ
DSA_2000_MIN_WAVELENGTH_KM, DSA_2000_MAX_WAVELENGTH_KM

DSA_2000_MAX_BASELINE_KM = 2 * DSA_2000_MAX_RADIUS_KM
DSA_2000_MAX_UV_EXTENT = DSA_2000_MAX_BASELINE_KM / DSA_2000_MIN_WAVELENGTH_KM
DSA_2000_UV_SCALE_FACTOR = DSA_2000_NPIX / (2*DSA_2000_MAX_UV_EXTENT) # Assume the UV plane is normalized to [-npix/2, npix/2]
```


```python
# To stop VERY time (and space) consuming cells from running
# Only switch to True if you want to regenerate some or all of the data
REGENERATE_DATA = False
```

# Preview clean true sky images


```python
def load_random_images(image_folder, n, m):
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.npy')]
    selected_files = random.sample(image_files, n * m)  # Randomly select n * m images
    images = [np.load(file) for file in selected_files]
    return images

def plot_image_grid(images, n, m, figsize=(15, 15)):
    fig, axes = plt.subplots(n, m, figsize=figsize)
    fig.suptitle("Randomly Selected True Sky Images", fontsize=20)

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i], cmap='gray')
            ax.axis('off')
        else:
            ax.remove()  # Remove empty axes for any extra grid cells
    
    plt.tight_layout()
    plt.show()

image_folder = './POLISH_train_HR'  # tRue sky images
n, m = 2, 2 

# Load and plot images
images = np.array(load_random_images(image_folder, n, m))
plot_image_grid(images, n, m)

```


    
![png](output_4_0.png)
    



```python
print(f"Image shape: {images[0].shape}")
print(f"Images min: {images.min()}")
print(f"Images max: {images.max()}") # Max flux per pixel is 16-bit integer (65535) (units of 1e-7 Jy?)
```

    Image shape: (1024, 1024, 1)
    Images min: 0.0
    Images max: 4744.586923829704


# Preview POLISH's dirty (LR) images


```python
def load_random_images(image_folder, n, m):
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.npy')]
    selected_files = random.sample(image_files, n * m)  # Randomly select n * m images
    images = [np.load(file) for file in selected_files]
    return images

def plot_image_grid(images, n, m, figsize=(15, 15)):
    fig, axes = plt.subplots(n, m, figsize=figsize)
    fig.suptitle("Randomly Selected LR Sky Images", fontsize=20)

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i], cmap='gray')
            ax.axis('off')
        else:
            ax.remove()  # Remove empty axes for any extra grid cells
    
    plt.tight_layout()
    plt.show()

image_folder = './POLISH_train_LR_bicubic/X2'  # dirty image
n, m = 2, 2 

# Load and plot images
images = np.array(load_random_images(image_folder, n, m))
plot_image_grid(images, n, m)

```


    
![png](output_7_0.png)
    



```python
print(f"Image shape: {images[0].shape}")
print(f"Images min: {images.min()}") # Min ~ 0
print(f"Images max: {images.max()}") # Max = 1
```

    Image shape: (512, 512, 1)
    Images min: 9.695717418723427e-06
    Images max: 0.9999594891379777


# Preview PSF used for POLISH LR images


```python
psf = np.load("./psf/psf_ideal.npy")
plt.imshow(psf)
```




    <matplotlib.image.AxesImage at 0x7f4eda03e720>




    
![png](output_10_1.png)
    


# Generate and save UV pattern

Based on this tutorial: https://nbviewer.org/github/ratt-ru/ratt-interferometry-course/blob/master/ipython-notebooks/aperture-synthesis.ipynb
Additional reading:
- https://github.com/ratt-ru/fundamentals_of_interferometry/blob/21d7754b273373adf161306f34aa920506d766a5/5_Imaging/5_2_sampling_functions_and_psfs.ipynb
- https://github.com/ratt-ru/fundamentals_of_interferometry/blob/master/5_Imaging/5_1_spatial_frequencies.ipynb


```python
def generate_random_antennas(num_antennas, max_radius=15):
    """
    Generate random antenna positions within a circular area.
    
    inputs:
        num_antennas (int): Number of antennas.
        max_radius (float): Radius of the circular area in kilometers.
        
    outputs:
        antenna_positions (ndarray): Antenna positions (x, y) in kilometers.
    """
    # Random points in polar coorsd
    r = max_radius * np.sqrt(np.random.uniform(0, 1, num_antennas))
    r = np.clip(r, 0, max_radius)  # Ensure r <= max_radius
    theta = np.random.uniform(0, 2 * np.pi, num_antennas)
    
    # polar to Cartesian coords
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # SANITY CHECK: Check for points outside the circle
    max_r_calculated = np.sqrt(x**2 + y**2).max()
    assert max_r_calculated <= max_radius, f"Point outside the circle detected: r = {max_r_calculated:.2f} km"

    return np.stack((x, y), axis=1)

def compute_uv_coverage(antenna_positions, wavelength):
    """
    Compute UV coverage from antenna positions.
    
    inputs:
        antenna_positions (ndarray): Antenna positions (x, y) in kilometers.
        wavelength (float): Wavelength in kilometers. This scales the baseline to UV coordinates.
        
    outputs:
        uv_points (ndarray): UV coverage points (u, v).
    """
    num_antennas = len(antenna_positions)
    uv_points = []
    max_baseline_magnitue = -9999999
    min_baseline_magnitude = 9999999
    # pairwise baselines
    for i in range(num_antennas):
        for j in range(i + 1, num_antennas):
            baseline = antenna_positions[j] - antenna_positions[i]
            uv_point = baseline / wavelength
            uv_points.append(uv_point)
            
            baseline_magnitude = np.sqrt(baseline.dot(baseline))
            if baseline_magnitude > max_baseline_magnitue:
                max_baseline_magnitue = baseline_magnitude
                
            if baseline_magnitude < min_baseline_magnitude:
                min_baseline_magnitude = baseline_magnitude
#             uv_points.append(-uv_point)  # conjugate (u, v) point # Note: Ignore this, will be handled when generating mask
    return np.array(uv_points), min_baseline_magnitude, max_baseline_magnitue


def create_uv_mask(image_shape, uv_points, uv_scale=1.0):
    """
    Create a binary UV mask based on UV points for a given Fourier grid shape.
    
    inputs:
        image_shape (tuple): Shape of the Fourier grid (ny, nx).
        uv_points (ndarray): UV coverage points (u, v) in wavelengths.
        uv_scale (float): Scaling factor for UV points to fit on grid.
        
    outputs:
        uv_mask (ndarray): Binary UV mask for the Fourier domain.
    """
    ny, nx = image_shape
    center_y, center_x = ny // 2, nx // 2
    uv_mask = np.zeros((ny, nx), dtype=int)
    
    for u, v in uv_points:
        u_idx = int(np.round(center_x + np.ceil(u * uv_scale)))
        v_idx = int(np.round(center_y + np.ceil(v * uv_scale)))
        
        u_idx_conj = int(np.round(center_x - np.ceil(u * uv_scale)))
        v_idx_conj = int(np.round(center_y - np.ceil(v * uv_scale)))
        
        # Check bounds
        if 0 <= u_idx < nx and 0 <= v_idx < ny:
            uv_mask[v_idx, u_idx] = 1
        if 0 <= u_idx_conj < nx and 0 <= v_idx_conj < ny:
            uv_mask[v_idx_conj, u_idx_conj] = 1
    
    return uv_mask

def compute_uv_scale(image_shape):
    """
    Compute the UV scaling factor to map UV coordinates to Fourier grid indices.
    """
    ny, nx = image_shape
    return min(nx, ny) / (2 * DSA_2000_MAX_UV_EXTENT)


def calculate_psnr(img1, img2):
    """
    Basoc PSNR compute.
    """
    return psnr(img1, img2, data_range=img1.max() - img1.min())


def normalize_image(img):
    return (img - img.min()) / (img.max() - img.min())

def plot_uv_coverage(uv_points, wavelength_km=1):
    """
    Plot the UV coverage.
    """
    uv_points_normalized = uv_points
    plt.figure(figsize=(8, 8))
    plt.scatter(uv_points_normalized[:, 0], uv_points_normalized[:, 1], s=5, color='blue', alpha=0.8, label="UV Points")
    plt.scatter(-uv_points_normalized[:, 0], -uv_points_normalized[:, 1], s=5, color='red', alpha=0.8, label="Conjugate UV Points")
    plt.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    plt.xlabel("u (wavelengths)")
    plt.ylabel("v (wavelengths)")
    plt.title("UV Coverage")
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.show()
```

## Test an example with 2 antennas


```python
# Parameters
num_antennas = 8 
max_radius_km = 15  # Max radius(km) for antenna positions

freq = 1e9 
wavelength_km = c / freq
# wavelength_km = DSA_2000_MAX_WAVELENGTH_KM
# wavelength_scaled = speed_of_light_km_per_s / frequency_resolution
image_shape = (256, 256)

antenna_positions = generate_random_antennas(num_antennas, max_radius=max_radius_km)
uv_points, min_baseline, max_baseline = compute_uv_coverage(antenna_positions, wavelength_km)
uv_scale = compute_uv_scale(image_shape) 

# Generate UV mask (grid sampling with points takes too long per image - use mask for speed)
uv_mask = create_uv_mask(image_shape, uv_points, uv_scale=uv_scale)

plot_uv_coverage(uv_points, wavelength_km)
plt.figure(figsize=(8, 8))
plt.imshow(uv_mask, cmap='turbo', origin='lower')
plt.title("UV Mask")
plt.xlabel("u (pixels)")
plt.ylabel("v (pixels)")
plt.show()

```


    
![png](output_14_0.png)
    



    
![png](output_14_1.png)
    



```python
def generate_primal_domain_image(uv_points, image_shape, uv_scale):
    """
    ONLY FOR SANITY CHECK.
    Generate a primal domain image based on the UV points.
    """
    ny, nx = image_shape
    center_y, center_x = ny // 2, nx // 2
    fft_image = np.zeros((ny, nx), dtype=complex)
    amplitude = 1.0  # Amplitude of sin
    
    for u, v in uv_points:
        u_idx = int(np.round(center_x + np.ceil(u * uv_scale)))
        v_idx = int(np.round(center_y + np.ceil(v * uv_scale)))
        
        if 0 <= u_idx < nx and 0 <= v_idx < ny:
            fft_image[v_idx, u_idx] = amplitude
            
            # conjugate point 
            fft_image[center_y - (v_idx), center_x - (u_idx)] = amplitude
    
    primal_image = np.real(np.fft.ifft2(np.fft.ifftshift(fft_image)))
    return primal_image

image_shape = (1024, 1024) 
uv_scale = compute_uv_scale(image_shape)

primal_image = generate_primal_domain_image(uv_points, image_shape, uv_scale)

plt.figure(figsize=(8, 8))
plt.imshow(primal_image, cmap='gray', origin='lower')
plt.title("Primal Domain Image (Corresponding Sinusoids)")
plt.colorbar(label="Amplitude")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```


    
![png](output_15_0.png)
    


## Generate UV coverages and masks to use for synthesizing data (Takes ~5-10 minutes for 2048 antennas)


```python
assert REGENERATE_DATA, "Not in data generation mode. Stopping this cell from wasting your time..."
# check output directory exists
output_dir_points = "./synthetic_data/uv/"
os.makedirs(output_dir_points, exist_ok=True)
np.random.seed(2048)
# params
antenna_counts = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
max_radius_km = 15
# frequencies_hz = [10e6, 0.7e9, 1.3e9, 2e9]
frequencies_hz = [1.3e9]

image_shape = (1024, 1024)
scaling_factors = [1,2,3,4]

# for plots
uv_masks = []
uv_titles = []

for frequency_hz in frequencies_hz:
    print(f"\nProcessing {frequency_hz} Hz...")
    wavelength_km = c / frequency_hz
    for num_antennas in antenna_counts:
        print(f"\nProcessing {num_antennas} antennas...")

        # Generate antennas and UV coverage
        antenna_positions = generate_random_antennas(num_antennas, max_radius=max_radius_km)
        uv_points, min_baseline, max_baseline = compute_uv_coverage(antenna_positions, wavelength_km)

        for s in scaling_factors:
            scaled_image_shape = (image_shape[0]//s,image_shape[1]//s) 
            uv_scale = compute_uv_scale(scaled_image_shape) 
            # Generate UV mask
            uv_mask = create_uv_mask(scaled_image_shape, uv_points, uv_scale=uv_scale)

            # Save UV points and mask
            np.save(f"{output_dir_points}/points_{num_antennas}antennas_{int(frequency_hz * 1e-6)}MHz.npy", uv_points)
            np.save(f"{output_dir_points}/mask_{num_antennas}antennas_{int(frequency_hz * 1e-6)}MHz_downscaled{s}x.npy", uv_mask)

            # for plots
            if s == 1:
                uv_masks.append(uv_mask)
        uv_titles.append(f"{num_antennas} Antennas")

num_plots = len(antenna_counts)
plt.figure(figsize=(15, 3 * num_plots))
for i, (uv_mask, title) in enumerate(zip(uv_masks, uv_titles)):
    plt.subplot(num_plots, 1, i + 1)
    plt.imshow(uv_mask, cmap='turbo', origin='lower')
    plt.title(title)
    plt.xlabel("u (pixels)")
    plt.ylabel("v (pixels)")
    plt.colorbar(label="Mask Value")
plt.tight_layout()
plt.show()
```

    
    Processing 10000000.0 Hz...
    
    Processing 8 antennas...
    
    Processing 16 antennas...
    
    Processing 32 antennas...
    
    Processing 64 antennas...
    
    Processing 128 antennas...
    
    Processing 256 antennas...
    
    Processing 512 antennas...
    
    Processing 1024 antennas...
    
    Processing 2048 antennas...
    
    Processing 700000000.0 Hz...
    
    Processing 8 antennas...
    
    Processing 16 antennas...
    
    Processing 32 antennas...
    
    Processing 64 antennas...
    
    Processing 128 antennas...
    
    Processing 256 antennas...
    
    Processing 512 antennas...
    
    Processing 1024 antennas...
    
    Processing 2048 antennas...
    
    Processing 1300000000.0 Hz...
    
    Processing 8 antennas...
    
    Processing 16 antennas...
    
    Processing 32 antennas...
    
    Processing 64 antennas...
    
    Processing 128 antennas...
    
    Processing 256 antennas...
    
    Processing 512 antennas...
    
    Processing 1024 antennas...
    
    Processing 2048 antennas...
    
    Processing 2000000000.0 Hz...
    
    Processing 8 antennas...
    
    Processing 16 antennas...
    
    Processing 32 antennas...
    
    Processing 64 antennas...
    
    Processing 128 antennas...
    
    Processing 256 antennas...
    
    Processing 512 antennas...
    
    Processing 1024 antennas...
    
    Processing 2048 antennas...



    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[347], line 46
         44 plt.figure(figsize=(15, 3 * num_plots))
         45 for i, (uv_mask, title) in enumerate(zip(uv_masks, uv_titles)):
    ---> 46     plt.subplot(num_plots, 1, i + 1)
         47     plt.imshow(uv_mask, cmap='turbo', origin='lower')
         48     plt.title(title)


    File ~/.conda/envs/jax/lib/python3.12/site-packages/matplotlib/pyplot.py:1534, in subplot(*args, **kwargs)
       1531 fig = gcf()
       1533 # First, search for an existing subplot with a matching spec.
    -> 1534 key = SubplotSpec._from_subplot_args(fig, args)
       1536 for ax in fig.axes:
       1537     # If we found an Axes at the position, we can reuse it if the user passed no
       1538     # kwargs or if the Axes class and kwargs are identical.
       1539     if (ax.get_subplotspec() == key
       1540         and (kwargs == {}
       1541              or (ax._projection_init
       1542                  == fig._process_projection_requirements(**kwargs)))):


    File ~/.conda/envs/jax/lib/python3.12/site-packages/matplotlib/gridspec.py:589, in SubplotSpec._from_subplot_args(figure, args)
        587 else:
        588     if not isinstance(num, Integral) or num < 1 or num > rows*cols:
    --> 589         raise ValueError(
        590             f"num must be an integer with 1 <= num <= {rows*cols}, "
        591             f"not {num!r}"
        592         )
        593     i = j = num
        594 return gs[i-1:j]


    ValueError: num must be an integer with 1 <= num <= 9, not 10



    
![png](output_17_2.png)
    



```python
output_dir = "./synthetic_data/uv/"
num_antennas = 32
frequency_hz = 1.3e9 # one of [10e6, 0.7e9, 1.3e9, 2e9]
s = 3
uv_mask_path = f"{output_dir}/mask_{num_antennas}antennas_{int(frequency_hz * 1e-6)}MHz_downscaled{s}x.npy"
uv_mask = np.load(uv_mask_path)  # Load UV mask

psf = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(uv_mask)))
psf = np.abs(psf)
psf /= psf.max()

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot UV mask
axes[0].imshow(uv_mask, cmap='viridis', origin='lower', interpolation='none')
axes[0].set_title("UV Mask")
axes[0].set_xlabel("U")
axes[0].set_ylabel("V")

# Plot PSF
axes[1].imshow(psf, cmap='inferno', origin='lower', interpolation='none')
axes[1].set_title("Point Spread Function (PSF)")
axes[1].set_xlabel("X")
axes[1].set_ylabel("Y")

plt.tight_layout()
plt.show()
```


    
![png](output_18_0.png)
    


## Dirty images test with some examples (no noise)


```python
output_dir = "./synthetic_data/uv/"
true_sky_dir = "./POLISH_train_HR/"
antenna_counts = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
s = 3 # downscale factor
max_radius_km = 15
num_images = 1  # Number of test images 
frequency_hz = 1.3e9

# For plots
results = {antenna_count: {"true_sky": [], "original_fft": [], "masked_fft": [], "dirty_image": []}
           for antenna_count in antenna_counts}

# Precompute mask indices - Significantly faster than using the entire mask!
def precompute_uv_mask_indices(uv_mask):
    """
    Precompute the indices of non-zero elements in the UV mask.
    
    Assumption: The mask is binary and no values inbetween 0-1.

    inputs:
        uv_mask (ndarray): The UV mask.

    outputs:
        indices (tuple): Indices of non-zero elements.
    """
    indices = np.nonzero(uv_mask)  # Only get non-zero indices
    return indices


def apply_uv_mask(fft_image, uv_indices):
    """
    Apply the UV mask using precomputed indices.

    inputs:
        fft_image (ndarray): The FFT of the image to be masked.
        uv_indices (tuple): Precomputed indices of the UV mask.

    outputs:
        masked_fft (ndarray): The masked FFT image.
    """
    masked_fft = np.zeros_like(fft_image, dtype=fft_image.dtype)
    masked_fft[uv_indices] = fft_image[uv_indices]
    return masked_fft

def downscale_image(image, scale_factor):
    """
    Downscale an image using bicubic interpolation.
    
    inputs:
        image (ndarray): Input image.
        scale_factor (float): Factor by which to downscale the image.
        
    outputs:
        downscaled_image (ndarray): Downscaled image.
    """
    return rescale(image, 1 / scale_factor, order=3) 

# process a single image
def process_image_with_precomputed_mask(true_sky, uv_indices):
    """
    Process a singel true sky image using the precomputed UV mask

    inputs:
        true_sky (ndarray): True sky image.
        uv_indices (tuple): Precomputed UV mask indices.

    outputs:
        fft_image (ndarray): FFT of the true sky.
        masked_fft (ndarray): Masked FFT.
        dirty_image (ndarray): Dirty image after applying the mask and inverse FFT.
    """
    fft_image = np.fft.fftshift(np.fft.fft2(true_sky))  # F(I_sky)

    masked_fft = apply_uv_mask(fft_image, uv_indices)  # F(I_sky) . S
    dirty_image = np.real(np.fft.ifft2(np.fft.ifftshift(masked_fft)))  # I_d

    return fft_image, masked_fft, dirty_image

# Load true sky images
print("Loading true sky images...")
image_files = sorted([f for f in os.listdir(true_sky_dir) if f.endswith(".npy")])[:num_images]
true_sky_images = [np.load(os.path.join(true_sky_dir, image_file)).squeeze() for image_file in image_files]
true_sky_images_downscaled = [downscale_image(np.load(os.path.join(true_sky_dir, image_file)).squeeze(), s) for image_file in image_files]
print(f"Loaded {len(true_sky_images)} true sky images.")


for antenna_count in antenna_counts:
    print(f"\nProcessing {antenna_count} antennas...")
    uv_mask_path = f"{output_dir}/mask_{antenna_count}antennas_{int(frequency_hz * 1e-6)}MHz_downscaled{s}x.npy"
    uv_mask = np.load(uv_mask_path)  # Load UV mask
    print(f"{output_dir}/mask_{antenna_count}antennas_{int(frequency_hz * 1e-6)}MHz_downscaled{s}x.npy")

    # Precompute mask indices
    uv_indices = precompute_uv_mask_indices(uv_mask)

    with ThreadPoolExecutor() as executor:
        futures = []
        for idx, true_sky in enumerate(true_sky_images_downscaled):
            start_time = time.time()  # Start timing
            futures.append(
            executor.submit(process_image_with_precomputed_mask, 
                                true_sky, uv_indices)
            )
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            print(f"  Image {idx + 1}/{num_images} for {antenna_count} antennas took {elapsed_time:.4f} seconds")

        for idx, future in enumerate(futures):
            fft_image, masked_fft, dirty_image = future.result()
            results[antenna_count]["true_sky"].append(true_sky_images[idx])
            results[antenna_count]["original_fft"].append(fft_image)
            results[antenna_count]["masked_fft"].append(masked_fft)
            results[antenna_count]["dirty_image"].append(dirty_image)

```

    Loading true sky images...
    Loaded 1 true sky images.
    
    Processing 8 antennas...
    ./synthetic_data/uv//mask_8antennas_1300MHz_downscaled3x.npy
      Image 1/1 for 8 antennas took 0.0016 seconds
    
    Processing 16 antennas...
    ./synthetic_data/uv//mask_16antennas_1300MHz_downscaled3x.npy
      Image 1/1 for 16 antennas took 0.0135 seconds
    
    Processing 32 antennas...
    ./synthetic_data/uv//mask_32antennas_1300MHz_downscaled3x.npy
      Image 1/1 for 32 antennas took 0.0011 seconds
    
    Processing 64 antennas...
    ./synthetic_data/uv//mask_64antennas_1300MHz_downscaled3x.npy
      Image 1/1 for 64 antennas took 0.0012 seconds
    
    Processing 128 antennas...
    ./synthetic_data/uv//mask_128antennas_1300MHz_downscaled3x.npy
      Image 1/1 for 128 antennas took 0.0012 seconds
    
    Processing 256 antennas...
    ./synthetic_data/uv//mask_256antennas_1300MHz_downscaled3x.npy
      Image 1/1 for 256 antennas took 0.0012 seconds
    
    Processing 512 antennas...
    ./synthetic_data/uv//mask_512antennas_1300MHz_downscaled3x.npy
      Image 1/1 for 512 antennas took 0.0013 seconds
    
    Processing 1024 antennas...
    ./synthetic_data/uv//mask_1024antennas_1300MHz_downscaled3x.npy
      Image 1/1 for 1024 antennas took 0.0006 seconds
    
    Processing 2048 antennas...
    ./synthetic_data/uv//mask_2048antennas_1300MHz_downscaled3x.npy
      Image 1/1 for 2048 antennas took 0.0010 seconds



```python
# Plot results
print("\nPlotting results...")
rows = len(antenna_counts)
cols = 4  # True Sky, Original FFT, Masked FFT, Dirty Image
plt.figure(figsize=(16, rows ))
for i, antenna_count in enumerate(antenna_counts):
    for j in range(num_images):
        # True Sky
        plt.subplot(rows * num_images, cols, i * num_images * cols + j * cols + 1)
        plt.imshow(np.abs(results[antenna_count]["true_sky"][j]), cmap="gray", origin="lower")
        plt.title(f"True Sky ({antenna_count} Antennas)")
        plt.axis("off")

        # Original FFT
        plt.subplot(rows * num_images, cols, i * num_images * cols + j * cols + 2)
        plt.imshow(np.log1p(np.abs(results[antenna_count]["original_fft"][j])), cmap="gray", origin="lower")
        plt.title(f"Original FFT (downscaled by {s})")
        plt.axis("off")

        # Masked FFT
        plt.subplot(rows * num_images, cols, i * num_images * cols + j * cols + 3)
        plt.imshow(np.log1p(np.abs(results[antenna_count]["masked_fft"][j])), cmap="gray", origin="lower")
        plt.title("Masked FFT")
        plt.axis("off")

        # Dirty Image
        plt.subplot(rows * num_images, cols, i * num_images * cols + j * cols + 4)
        plt.imshow(results[antenna_count]["dirty_image"][j], cmap="gray", origin="lower")
        plt.title("Dirty Image")
        plt.axis("off")

plt.tight_layout()
plt.show()
print("Processing and plotting completed.")

```

    
    Plotting results...



    
![png](output_21_1.png)
    


    Processing and plotting completed.



```python
true_sky_dir = "./POLISH_train_HR/"
num_images = 5  # Number of images to load

# Load true sky images
print("Loading true sky images...")
image_files = sorted([f for f in os.listdir(true_sky_dir) if f.endswith(".npy")])[:num_images]
true_sky_images = [np.load(os.path.join(true_sky_dir, image_file)).squeeze() for image_file in image_files]
print(f"Loaded {len(true_sky_images)} true sky images.")

# Plot FFT of the images
plt.figure(figsize=(15, 10))
for idx, true_sky in enumerate(true_sky_images):
    # Compute FFT
    fft_image = np.fft.fftshift(np.fft.fft2(true_sky))
    fft_magnitude = np.log1p(np.abs(fft_image))  # Log scale for better visualization

    # Plot
    plt.subplot(2, 3, idx + 1)
    plt.imshow(fft_magnitude, cmap="gray", origin="lower")
    plt.title(f"Image {idx + 1} FFT")
    plt.axis("off")

plt.tight_layout()
plt.show()

```

    Loading true sky images...
    Loaded 5 true sky images.



    
![png](output_22_1.png)
    


## Test to generate images with measurement noise
Generate dirty images with measurement noise added to the true sky image before applying sampling function.
[POLISH paper](https://arxiv.org/pdf/2111.03249) and the [code](https://github.com/liamconnor/polish-pub/blob/main/make_img_pairs.py) described adding noise to $I_sky$ before sampling and downscaling:
> Before applying the PSF kernel to obtain $I_d$, zero-mean i.i.d. Gaussian noise with a standard deviation $\sigma_{sky}$ is applied
to mimic $n_{sky}$. The convolved image is then downsampled by a factor $s$.

In the code, line 117:
```python
if noise:
    data_noise = data + np.random.normal(0,5,data.shape)
```

Using the code, we get noisy images in `./POLISH_train_HR_Noise` whose pixel values are in units of mJy, and the noise is $5$

Here, since we also need to use the visibilities for training, we downscale the original image before adding noise and applying sampling function. Which differs from POLISH paper where they only downsampled the dirty image.



```python
output_dir = "./synthetic_data/uv/"
true_sky_dir = "./POLISH_train_HR/" # Use noisy true sky
true_sky_noise_dir = "./POLISH_train_HR_Noise/" # Use noisy true sky
antenna_counts = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
max_radius_km = 15
frequency_hz = 1.3e9 # Hz
num_images = 5  # Number of test images 
s = 3 # downscale factor = 3 for DSA (=2 for VLA)

# For plots
results = {antenna_count: {"true_sky": [], "true_sky_noisy": [], "original_fft": [], "masked_fft": [], "dirty_image": []}
           for antenna_count in antenna_counts}

# Load true sky images
print("Loading true sky images...")
image_files = sorted([f for f in os.listdir(true_sky_dir) if f.endswith(".npy")])[:num_images]
true_sky_images = [np.load(os.path.join(true_sky_dir, image_file)).squeeze() for image_file in image_files]

noisy_image_files = sorted([f for f in os.listdir(true_sky_noise_dir) if f.endswith(".npy")])[:num_images]
true_sky_noisy_images = [np.clip(
        np.load(os.path.join(true_sky_noise_dir, image_file)).squeeze(), 
        a_min=0, a_max=None
    ) for image_file in noisy_image_files]
true_sky_noisy_images_downscaled = [downscale_image(image,s) for image in true_sky_noisy_images]
print(f"Loaded {len(true_sky_images)} true sky images and {len(true_sky_images_downscaled)} and downsampled by {s}.")

# Process all images and antenna counts
for antenna_count in antenna_counts:
    print(f"\nProcessing {antenna_count} antennas...")
    uv_mask_path = f"{output_dir}/mask_{antenna_count}antennas_{int(frequency_hz * 1e-6)}MHz_downscaled{s}x.npy"
    uv_mask = np.load(uv_mask_path)  # Load UV mask
    print(f"{output_dir}/mask_{antenna_count}antennas_{int(frequency_hz * 1e-6)}MHz_downscaled{s}x.npy")

    # Precompute mask indices
    uv_indices = precompute_uv_mask_indices(uv_mask)

    with ThreadPoolExecutor() as executor:
        futures = []
        for idx, true_sky_noisy in enumerate(true_sky_noisy_images_downscaled):
            start_time = time.time()  # Start timing
            futures.append(
            executor.submit(process_image_with_precomputed_mask, 
                                true_sky_noisy, uv_indices)
            )
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            print(f"  Image {idx + 1}/{num_images} for {antenna_count} antennas took {elapsed_time:.4f} seconds")

        for idx, future in enumerate(futures):
            fft_image, masked_fft, dirty_image = future.result()
            results[antenna_count]["true_sky"].append(true_sky_images[idx])
            results[antenna_count]["true_sky_noisy"].append(true_sky_noisy_images[idx])
            results[antenna_count]["original_fft"].append(fft_image)
            results[antenna_count]["masked_fft"].append(masked_fft)
            results[antenna_count]["dirty_image"].append(dirty_image)

```

    Loading true sky images...
    Loaded 5 true sky images and 1 and downsampled by 3.
    
    Processing 8 antennas...
    ./synthetic_data/uv//mask_8antennas_1300MHz_downscaled3x.npy
      Image 1/5 for 8 antennas took 0.0010 seconds
      Image 2/5 for 8 antennas took 0.0007 seconds
      Image 3/5 for 8 antennas took 0.0006 seconds
      Image 4/5 for 8 antennas took 0.0007 seconds
      Image 5/5 for 8 antennas took 0.0005 seconds
    
    Processing 16 antennas...
    ./synthetic_data/uv//mask_16antennas_1300MHz_downscaled3x.npy
      Image 1/5 for 16 antennas took 0.0003 seconds
      Image 2/5 for 16 antennas took 0.0003 seconds
      Image 3/5 for 16 antennas took 0.0002 seconds
      Image 4/5 for 16 antennas took 0.0002 seconds
      Image 5/5 for 16 antennas took 0.0005 seconds
    
    Processing 32 antennas...
    ./synthetic_data/uv//mask_32antennas_1300MHz_downscaled3x.npy
      Image 1/5 for 32 antennas took 0.0004 seconds
      Image 2/5 for 32 antennas took 0.0005 seconds
      Image 3/5 for 32 antennas took 0.0005 seconds
      Image 4/5 for 32 antennas took 0.0005 seconds
      Image 5/5 for 32 antennas took 0.0005 seconds
    
    Processing 64 antennas...
    ./synthetic_data/uv//mask_64antennas_1300MHz_downscaled3x.npy
      Image 1/5 for 64 antennas took 0.0006 seconds
      Image 2/5 for 64 antennas took 0.0005 seconds
      Image 3/5 for 64 antennas took 0.0003 seconds
      Image 4/5 for 64 antennas took 0.0003 seconds
      Image 5/5 for 64 antennas took 0.0003 seconds
    
    Processing 128 antennas...
    ./synthetic_data/uv//mask_128antennas_1300MHz_downscaled3x.npy
      Image 1/5 for 128 antennas took 0.0004 seconds
      Image 2/5 for 128 antennas took 0.0003 seconds
      Image 3/5 for 128 antennas took 0.0005 seconds
      Image 4/5 for 128 antennas took 0.0003 seconds
      Image 5/5 for 128 antennas took 0.0005 seconds
    
    Processing 256 antennas...
    ./synthetic_data/uv//mask_256antennas_1300MHz_downscaled3x.npy
      Image 1/5 for 256 antennas took 0.0004 seconds
      Image 2/5 for 256 antennas took 0.0005 seconds
      Image 3/5 for 256 antennas took 0.0005 seconds
      Image 4/5 for 256 antennas took 0.0005 seconds
      Image 5/5 for 256 antennas took 0.0006 seconds
    
    Processing 512 antennas...
    ./synthetic_data/uv//mask_512antennas_1300MHz_downscaled3x.npy
      Image 1/5 for 512 antennas took 0.0006 seconds
      Image 2/5 for 512 antennas took 0.0003 seconds
      Image 3/5 for 512 antennas took 0.0003 seconds
      Image 4/5 for 512 antennas took 0.0005 seconds
      Image 5/5 for 512 antennas took 0.0004 seconds
    
    Processing 1024 antennas...
    ./synthetic_data/uv//mask_1024antennas_1300MHz_downscaled3x.npy
      Image 1/5 for 1024 antennas took 0.0004 seconds
      Image 2/5 for 1024 antennas took 0.0003 seconds
      Image 3/5 for 1024 antennas took 0.0005 seconds
      Image 4/5 for 1024 antennas took 0.0003 seconds
      Image 5/5 for 1024 antennas took 0.0003 seconds
    
    Processing 2048 antennas...
    ./synthetic_data/uv//mask_2048antennas_1300MHz_downscaled3x.npy
      Image 1/5 for 2048 antennas took 0.0004 seconds
      Image 2/5 for 2048 antennas took 0.0003 seconds
      Image 3/5 for 2048 antennas took 0.0003 seconds
      Image 4/5 for 2048 antennas took 0.0005 seconds
      Image 5/5 for 2048 antennas took 0.0004 seconds



```python
# Plot results
print("\nPlotting results...")
rows = len(antenna_counts)
cols = 4  # True Sky, Original FFT, Masked FFT, Dirty Image
plt.figure(figsize=(16, rows * 32))
for i, antenna_count in enumerate(antenna_counts):
    for j in range(num_images):
        # True Sky
        plt.subplot(rows * num_images, cols, i * num_images * cols + j * cols + 1)
        plt.imshow(np.abs(results[antenna_count]["true_sky_noisy"][j]), cmap="gray", origin="lower")
        plt.title(f"True Sky ({antenna_count} Antennas) + noise")
        plt.axis("off")

        # Original FFT
        plt.subplot(rows * num_images, cols, i * num_images * cols + j * cols + 2)
        plt.imshow(np.log1p(np.abs(results[antenna_count]["original_fft"][j])), cmap="gray", origin="lower")
        plt.title(f"Original FFT (downscaled by {s})")
        plt.axis("off")

        # Masked FFT
        plt.subplot(rows * num_images, cols, i * num_images * cols + j * cols + 3)
        plt.imshow(np.log1p(np.abs(results[antenna_count]["masked_fft"][j])), cmap="gray", origin="lower")
        plt.title("Masked FFT")
        plt.axis("off")

        # Dirty Image
        plt.subplot(rows * num_images, cols, i * num_images * cols + j * cols + 4)
        plt.imshow(results[antenna_count]["dirty_image"][j], cmap="gray", origin="lower")
        plt.title("Dirty Image")
        plt.axis("off")

plt.tight_layout()
plt.show()
print("Processing and plotting completed.")

```

    
    Plotting results...



    
![png](output_25_1.png)
    


    Processing and plotting completed.



```python
antenna_count = antenna_counts[-1]
image_idx = 3

dirty_image = results[antenna_count]["dirty_image"][image_idx]
true_sky_noisy = rescale(results[antenna_count]["true_sky_noisy"][image_idx], 1/s)
true_sky = rescale(results[antenna_count]["true_sky"][image_idx], 1/s)

psnr_dirty = calculate_psnr(dirty_image, true_sky)
psnr_noisy = calculate_psnr(true_sky_noisy, true_sky)

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(dirty_image, cmap='turbo', origin='lower')
plt.title(f"Dirty Image\nPSNR: {psnr_dirty:.2f} dB")
plt.axis('off')
plt.subplot(2, 3, 2)
plt.imshow(true_sky_noisy, cmap='turbo', origin='lower')
plt.title(f"Noisy True Sky\nPSNR: {psnr_noisy:.2f} dB")
plt.axis('off')
plt.subplot(2, 3, 3)
plt.imshow(true_sky, cmap='turbo', origin='lower')
plt.title("True Sky (Reference)")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(np.log1p(dirty_image), cmap='turbo', origin='lower')
plt.title(f"log(Dirty Image)")
plt.axis('off')
plt.subplot(2, 3, 5)
plt.imshow(np.log1p(true_sky_noisy), cmap='turbo', origin='lower')
plt.title(f"log(Noisy True Sky)")
plt.axis('off')
plt.subplot(2, 3, 6)
plt.imshow(np.log1p(true_sky), cmap='turbo', origin='lower')
plt.title("log(True Sky) (Reference)")
plt.axis('off')

plt.tight_layout()
plt.show()

```

    /tmp/ipykernel_328518/3397546844.py:27: RuntimeWarning: invalid value encountered in log1p
      plt.imshow(np.log1p(dirty_image), cmap='turbo', origin='lower')



    
![png](output_26_1.png)
    



```python
antenna_count = 32
image_idx = 0

dirty_image = rescale(results[antenna_count]["dirty_image"][image_idx],2)
plt.figure(figsize=(15, 15))
plt.imshow(dirty_image, cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7f4ec5230890>




    
![png](output_27_1.png)
    


## Generate the training and validation set
We want the following for our data set:
- True sky image (1024x1024) as labels (done already).
- True sky noisy image (1024x1024) (done already).


Then:
- For each image in training, training noisy and validation, validation noisy sets:
    - Save the following:
        - True sky image visibility (FFT True sky) (original size) as labels (to be computed).
        - True sky noisy image visibility (FFT True sky noisy) (original size) (to be computed).
    - For each `num_antenna` in [32, 128, 512, 2048]:
        - For each downscaling factor `s` in [1,2]:
            - For each frequency `freq` in [10e6, 0.7e9, 1.3e9, 2e9]:
                - Use the saved mask `mask_{antenna_count}antennas_{int(frequency_hz * 1e-6)}MHz_downscaled{s}x.npy`.
                - Downsample the true sky image by `s`, get the visibility.
                - Downsample the true sky noisy image by `s`, get the visibility.
                - Apply loaded saved mask to apply sampling function to the visibilities.
                - Save the following:
                    - Dirty image of from sparsely sampled true sky downsampled visibility.
                    - Dirty image of from sparsely sampled true sky noisy downsampled visibility.


```python
output_dir = "./synthetic_data"
true_sky_dirs = {
    "train": "./POLISH_train_HR/",
    "train_noisy": "./POLISH_train_HR_Noise/",
    "valid": "./POLISH_valid_HR/",
    "valid_noisy": "./POLISH_valid_HR_Noise/",
}
frequencies_hz = [10e6, 0.7e9, 1.3e9, 2e9]  # Frequencies in Hz
antenna_counts = [32, 128, 512, 2048]
scaling_factors = [1, 2]
c = 299792  # Speed of light in km/s

def load_images(folder, num_images=None):
    """
    Load images from a given folder.
    """
    files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".npy")])
    if num_images:
        files = files[:num_images]
    return [np.load(f) for f in files]

def load_images_with_filenames(folder):
    """
    Load images and their filenames from a folder.
    """
    files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".npy")])
    return [(np.load(f), os.path.basename(f).replace(".npy", "")) for f in files]
```


```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Constants
MAX_WORKERS = 16  # Number of parallel threads (adjust based on your system)

# Correct calculation of total files
total_files = sum(
    len(os.listdir(true_sky_dirs[dataset_type])) * len(frequencies_hz) * len(antenna_counts) * len(scaling_factors)
    for dataset_type in ["train", "valid"]
)

# Precomputed UV masks and indices cache
precomputed_masks = {}

def get_precomputed_uv_indices(uv_mask_path):
    """
    Load or retrieve precomputed UV mask indices.
    """
    if uv_mask_path not in precomputed_masks:
        uv_mask = np.load(uv_mask_path)
        uv_indices = precompute_uv_mask_indices(uv_mask)
        precomputed_masks[uv_mask_path] = uv_indices
    return precomputed_masks[uv_mask_path]

# Function to process a single file
def process_single_file(image, filename, noisy_image, noisy_filename, uv_indices, scaling_factor, frequency_name, antenna_count, dataset_type):
    assert filename == noisy_filename.replace("_noise", ""), "Mismatch between true and noisy filenames"
    
    # Squeeze images to ensure proper dimensions
    image = image.squeeze()
    noisy_image = noisy_image.squeeze()
    
    # Determine save path
    save_path = os.path.join(output_dir, f"{dataset_type}/freq_{frequency_name}/antennas_{antenna_count}_s{scaling_factor}")
    os.makedirs(save_path, exist_ok=True)

    # File paths
    file_paths = {
        "fft_full": os.path.join(save_path, f"{filename}_fft_full.npy"),
        "masked_fft_full": os.path.join(save_path, f"{filename}_masked_fft_full.npy"),
        "dirty_image_full": os.path.join(save_path, f"{filename}_dirty_image_full.npy"),
        "noisy_fft_full": os.path.join(save_path, f"{filename}_noisy_fft_full.npy"),
        "noisy_masked_fft_full": os.path.join(save_path, f"{filename}_noisy_masked_fft_full.npy"),
        "noisy_dirty_image_full": os.path.join(save_path, f"{filename}_noisy_dirty_image_full.npy"),
    }

    # Check if all files exist
    if all(os.path.exists(path) for path in file_paths.values()):
        return False  # Skip this file

    # Downscale images
    image_downscaled = downscale_image(image, scaling_factor)
    noisy_image_downscaled = downscale_image(noisy_image, scaling_factor)

    # Process images
    outputs = process_image_with_precomputed_mask(image_downscaled, uv_indices)
    noisy_outputs = process_image_with_precomputed_mask(noisy_image_downscaled, uv_indices)

    # Save outputs
    np.save(file_paths["fft_full"], outputs[0])
    np.save(file_paths["masked_fft_full"], outputs[1])
    np.save(file_paths["dirty_image_full"], outputs[2])

    np.save(file_paths["noisy_fft_full"], noisy_outputs[0])
    np.save(file_paths["noisy_masked_fft_full"], noisy_outputs[1])
    np.save(file_paths["noisy_dirty_image_full"], noisy_outputs[2])
    
    return True  # File was processed
```


```python
assert REGENERATE_DATA, "Not in data generation mode. Stopping this cell from wasting your time..."

# Parallel processing with ThreadPoolExecutor
processed_files = 0  # Counter for processed files
skipped_files = 0  # Counter for skipped files

for dataset_type in ["train", "valid"]:
    print(f"Processing {dataset_type} dataset...")
    
    true_sky_images = load_images_with_filenames(true_sky_dirs[dataset_type])
    true_sky_noisy_images = load_images_with_filenames(true_sky_dirs[f"{dataset_type}_noisy"])
    
    for frequency_hz in frequencies_hz[::-1]:
        frequency_name = f"{int(frequency_hz * 1e-6)}MHz"
        print(f"  Processing frequency {frequency_name}...")

        for antenna_count in antenna_counts:
            print(f"    Processing {antenna_count} antennas...")

            for scaling_factor in scaling_factors:
                print(f"      Processing downscale factor {scaling_factor}...")

                # Load UV mask indices specific to scaling factor
                uv_mask_path = f"{output_dir}/uv/mask_{antenna_count}antennas_{frequency_name}_downscaled{scaling_factor}x.npy"
                uv_indices = get_precomputed_uv_indices(uv_mask_path)
                
                # Parallel processing
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = [
                        executor.submit(
                            process_single_file,
                            image, filename, noisy_image, noisy_filename,
                            uv_indices, scaling_factor, frequency_name, antenna_count, dataset_type
                        )
                        for (image, filename), (noisy_image, noisy_filename) in zip(true_sky_images, true_sky_noisy_images)
                    ]

                    for future in as_completed(futures):
                        if future.result():  # If file was processed
                            processed_files += 1
                        else:  # If file was skipped
                            skipped_files += 1
                            processed_files += 1
                        if (processed_files) % 10 == 0:
                            print(f"        Processed: {processed_files}/{total_files}, Skipped: {skipped_files}")

print("Dataset generation completed.")
```

    Processing train dataset...
      Processing frequency 2000MHz...
        Processing 32 antennas...
          Processing downscale factor 1...
            Processed: 10/28800, Skipped: 0
            Processed: 20/28800, Skipped: 0
            Processed: 30/28800, Skipped: 0
            Processed: 40/28800, Skipped: 0
            Processed: 50/28800, Skipped: 0
            Processed: 60/28800, Skipped: 0
            Processed: 70/28800, Skipped: 0
            Processed: 80/28800, Skipped: 0
            Processed: 90/28800, Skipped: 0
            Processed: 100/28800, Skipped: 0
            Processed: 110/28800, Skipped: 0
            Processed: 120/28800, Skipped: 0
            Processed: 130/28800, Skipped: 0
            Processed: 140/28800, Skipped: 0
            Processed: 150/28800, Skipped: 0
            Processed: 160/28800, Skipped: 0
            Processed: 170/28800, Skipped: 0
            Processed: 180/28800, Skipped: 0
            Processed: 190/28800, Skipped: 0
            Processed: 200/28800, Skipped: 0
            Processed: 210/28800, Skipped: 0
            Processed: 220/28800, Skipped: 0
            Processed: 230/28800, Skipped: 0
            Processed: 240/28800, Skipped: 0
            Processed: 250/28800, Skipped: 0
            Processed: 260/28800, Skipped: 0
            Processed: 270/28800, Skipped: 0
            Processed: 280/28800, Skipped: 0
            Processed: 290/28800, Skipped: 0
            Processed: 300/28800, Skipped: 0
            Processed: 310/28800, Skipped: 0
            Processed: 320/28800, Skipped: 0
            Processed: 330/28800, Skipped: 0
            Processed: 340/28800, Skipped: 0
            Processed: 350/28800, Skipped: 0
            Processed: 360/28800, Skipped: 0
            Processed: 370/28800, Skipped: 0
            Processed: 380/28800, Skipped: 0
            Processed: 390/28800, Skipped: 0
            Processed: 400/28800, Skipped: 0
            Processed: 410/28800, Skipped: 0
            Processed: 420/28800, Skipped: 0
            Processed: 430/28800, Skipped: 0
            Processed: 440/28800, Skipped: 0
            Processed: 450/28800, Skipped: 0
            Processed: 460/28800, Skipped: 0
            Processed: 470/28800, Skipped: 0
            Processed: 480/28800, Skipped: 0
            Processed: 490/28800, Skipped: 0
            Processed: 500/28800, Skipped: 0
            Processed: 510/28800, Skipped: 0
            Processed: 520/28800, Skipped: 0
            Processed: 530/28800, Skipped: 0
            Processed: 540/28800, Skipped: 0
            Processed: 550/28800, Skipped: 0
            Processed: 560/28800, Skipped: 0
            Processed: 570/28800, Skipped: 0
            Processed: 580/28800, Skipped: 0
            Processed: 590/28800, Skipped: 0
            Processed: 600/28800, Skipped: 0
            Processed: 610/28800, Skipped: 0
            Processed: 620/28800, Skipped: 0
            Processed: 630/28800, Skipped: 0
            Processed: 640/28800, Skipped: 0
            Processed: 650/28800, Skipped: 0
            Processed: 660/28800, Skipped: 0
            Processed: 670/28800, Skipped: 0
            Processed: 680/28800, Skipped: 0
            Processed: 690/28800, Skipped: 0
            Processed: 700/28800, Skipped: 0
            Processed: 710/28800, Skipped: 0
            Processed: 720/28800, Skipped: 0
            Processed: 730/28800, Skipped: 0
            Processed: 740/28800, Skipped: 0
            Processed: 750/28800, Skipped: 0
            Processed: 760/28800, Skipped: 0
            Processed: 770/28800, Skipped: 0
            Processed: 780/28800, Skipped: 0
            Processed: 790/28800, Skipped: 0
            Processed: 800/28800, Skipped: 0
          Processing downscale factor 2...
            Processed: 810/28800, Skipped: 0
            Processed: 820/28800, Skipped: 0
            Processed: 830/28800, Skipped: 0
            Processed: 840/28800, Skipped: 0
            Processed: 850/28800, Skipped: 0
            Processed: 860/28800, Skipped: 0
            Processed: 870/28800, Skipped: 0
            Processed: 880/28800, Skipped: 0
            Processed: 890/28800, Skipped: 0
            Processed: 900/28800, Skipped: 0
            Processed: 910/28800, Skipped: 0
            Processed: 920/28800, Skipped: 0
            Processed: 930/28800, Skipped: 0
            Processed: 940/28800, Skipped: 0
            Processed: 950/28800, Skipped: 0
            Processed: 960/28800, Skipped: 0
            Processed: 970/28800, Skipped: 0
            Processed: 980/28800, Skipped: 0
            Processed: 990/28800, Skipped: 0
            Processed: 1000/28800, Skipped: 0
            Processed: 1010/28800, Skipped: 0
            Processed: 1020/28800, Skipped: 0
            Processed: 1030/28800, Skipped: 0
            Processed: 1040/28800, Skipped: 0
            Processed: 1050/28800, Skipped: 0
            Processed: 1060/28800, Skipped: 0
            Processed: 1070/28800, Skipped: 0
            Processed: 1080/28800, Skipped: 0
            Processed: 1090/28800, Skipped: 0
            Processed: 1100/28800, Skipped: 0
            Processed: 1110/28800, Skipped: 0
            Processed: 1120/28800, Skipped: 0
            Processed: 1130/28800, Skipped: 0
            Processed: 1140/28800, Skipped: 0
            Processed: 1150/28800, Skipped: 0
            Processed: 1160/28800, Skipped: 0
            Processed: 1170/28800, Skipped: 0
            Processed: 1180/28800, Skipped: 0
            Processed: 1190/28800, Skipped: 0
            Processed: 1200/28800, Skipped: 0
            Processed: 1210/28800, Skipped: 0
            Processed: 1220/28800, Skipped: 0
            Processed: 1230/28800, Skipped: 0
            Processed: 1240/28800, Skipped: 0
            Processed: 1250/28800, Skipped: 0
            Processed: 1260/28800, Skipped: 0
            Processed: 1270/28800, Skipped: 0
            Processed: 1280/28800, Skipped: 0
            Processed: 1290/28800, Skipped: 0
            Processed: 1300/28800, Skipped: 0
            Processed: 1310/28800, Skipped: 0
            Processed: 1320/28800, Skipped: 0
            Processed: 1330/28800, Skipped: 0
            Processed: 1340/28800, Skipped: 0
            Processed: 1350/28800, Skipped: 0
            Processed: 1360/28800, Skipped: 0
            Processed: 1370/28800, Skipped: 0
            Processed: 1380/28800, Skipped: 0
            Processed: 1390/28800, Skipped: 0
            Processed: 1400/28800, Skipped: 0
            Processed: 1410/28800, Skipped: 0
            Processed: 1420/28800, Skipped: 0
            Processed: 1430/28800, Skipped: 0
            Processed: 1440/28800, Skipped: 0
            Processed: 1450/28800, Skipped: 0
            Processed: 1460/28800, Skipped: 0
            Processed: 1470/28800, Skipped: 0
            Processed: 1480/28800, Skipped: 0
            Processed: 1490/28800, Skipped: 0
            Processed: 1500/28800, Skipped: 0
            Processed: 1510/28800, Skipped: 0
            Processed: 1520/28800, Skipped: 0
            Processed: 1530/28800, Skipped: 0
            Processed: 1540/28800, Skipped: 0
            Processed: 1550/28800, Skipped: 0
            Processed: 1560/28800, Skipped: 0
            Processed: 1570/28800, Skipped: 0
            Processed: 1580/28800, Skipped: 0
            Processed: 1590/28800, Skipped: 0
            Processed: 1600/28800, Skipped: 0
        Processing 128 antennas...
          Processing downscale factor 1...
            Processed: 1610/28800, Skipped: 0
            Processed: 1620/28800, Skipped: 0
            Processed: 1630/28800, Skipped: 0
            Processed: 1640/28800, Skipped: 0
            Processed: 1650/28800, Skipped: 0
            Processed: 1660/28800, Skipped: 0
            Processed: 1670/28800, Skipped: 0
            Processed: 1680/28800, Skipped: 0
            Processed: 1690/28800, Skipped: 0
            Processed: 1700/28800, Skipped: 0
            Processed: 1710/28800, Skipped: 0
            Processed: 1720/28800, Skipped: 0
            Processed: 1730/28800, Skipped: 0
            Processed: 1740/28800, Skipped: 0
            Processed: 1750/28800, Skipped: 0
            Processed: 1760/28800, Skipped: 0
            Processed: 1770/28800, Skipped: 0
            Processed: 1780/28800, Skipped: 0
            Processed: 1790/28800, Skipped: 0
            Processed: 1800/28800, Skipped: 0
            Processed: 1810/28800, Skipped: 0
            Processed: 1820/28800, Skipped: 0
            Processed: 1830/28800, Skipped: 0
            Processed: 1840/28800, Skipped: 0
            Processed: 1850/28800, Skipped: 0
            Processed: 1860/28800, Skipped: 0
            Processed: 1870/28800, Skipped: 0
            Processed: 1880/28800, Skipped: 0
            Processed: 1890/28800, Skipped: 0
            Processed: 1900/28800, Skipped: 0
            Processed: 1910/28800, Skipped: 0
            Processed: 1920/28800, Skipped: 0
            Processed: 1930/28800, Skipped: 0
            Processed: 1940/28800, Skipped: 0
            Processed: 1950/28800, Skipped: 0
            Processed: 1960/28800, Skipped: 0
            Processed: 1970/28800, Skipped: 0
            Processed: 1980/28800, Skipped: 0
            Processed: 1990/28800, Skipped: 0
            Processed: 2000/28800, Skipped: 0
            Processed: 2010/28800, Skipped: 0
            Processed: 2020/28800, Skipped: 0
            Processed: 2030/28800, Skipped: 0
            Processed: 2040/28800, Skipped: 0
            Processed: 2050/28800, Skipped: 0
            Processed: 2060/28800, Skipped: 0
            Processed: 2070/28800, Skipped: 0
            Processed: 2080/28800, Skipped: 0
            Processed: 2090/28800, Skipped: 0
            Processed: 2100/28800, Skipped: 0
            Processed: 2110/28800, Skipped: 0
            Processed: 2120/28800, Skipped: 0
            Processed: 2130/28800, Skipped: 0
            Processed: 2140/28800, Skipped: 0
            Processed: 2150/28800, Skipped: 0
            Processed: 2160/28800, Skipped: 0
            Processed: 2170/28800, Skipped: 0
            Processed: 2180/28800, Skipped: 0
            Processed: 2190/28800, Skipped: 0
            Processed: 2200/28800, Skipped: 0
            Processed: 2210/28800, Skipped: 0
            Processed: 2220/28800, Skipped: 0
            Processed: 2230/28800, Skipped: 0
            Processed: 2240/28800, Skipped: 0
            Processed: 2250/28800, Skipped: 0
            Processed: 2260/28800, Skipped: 0
            Processed: 2270/28800, Skipped: 0
            Processed: 2280/28800, Skipped: 0
            Processed: 2290/28800, Skipped: 0
            Processed: 2300/28800, Skipped: 0
            Processed: 2310/28800, Skipped: 0
            Processed: 2320/28800, Skipped: 0
            Processed: 2330/28800, Skipped: 0
            Processed: 2340/28800, Skipped: 0
            Processed: 2350/28800, Skipped: 0
            Processed: 2360/28800, Skipped: 0
            Processed: 2370/28800, Skipped: 0
            Processed: 2380/28800, Skipped: 0
            Processed: 2390/28800, Skipped: 0
            Processed: 2400/28800, Skipped: 0
          Processing downscale factor 2...
            Processed: 2410/28800, Skipped: 0
            Processed: 2420/28800, Skipped: 0
            Processed: 2430/28800, Skipped: 0
            Processed: 2440/28800, Skipped: 0
            Processed: 2450/28800, Skipped: 0
            Processed: 2460/28800, Skipped: 0
            Processed: 2470/28800, Skipped: 0
            Processed: 2480/28800, Skipped: 0
            Processed: 2490/28800, Skipped: 0
            Processed: 2500/28800, Skipped: 0
            Processed: 2510/28800, Skipped: 0
            Processed: 2520/28800, Skipped: 0
            Processed: 2530/28800, Skipped: 0
            Processed: 2540/28800, Skipped: 0
            Processed: 2550/28800, Skipped: 0
            Processed: 2560/28800, Skipped: 0
            Processed: 2570/28800, Skipped: 0
            Processed: 2580/28800, Skipped: 0
            Processed: 2590/28800, Skipped: 0
            Processed: 2600/28800, Skipped: 0
            Processed: 2610/28800, Skipped: 0
            Processed: 2620/28800, Skipped: 0
            Processed: 2630/28800, Skipped: 0
            Processed: 2640/28800, Skipped: 0
            Processed: 2650/28800, Skipped: 0
            Processed: 2660/28800, Skipped: 0
            Processed: 2670/28800, Skipped: 0
            Processed: 2680/28800, Skipped: 0
            Processed: 2690/28800, Skipped: 0
            Processed: 2700/28800, Skipped: 0
            Processed: 2710/28800, Skipped: 0
            Processed: 2720/28800, Skipped: 0
            Processed: 2730/28800, Skipped: 0
            Processed: 2740/28800, Skipped: 0
            Processed: 2750/28800, Skipped: 0
            Processed: 2760/28800, Skipped: 0
            Processed: 2770/28800, Skipped: 0
            Processed: 2780/28800, Skipped: 0
            Processed: 2790/28800, Skipped: 0
            Processed: 2800/28800, Skipped: 0
            Processed: 2810/28800, Skipped: 0
            Processed: 2820/28800, Skipped: 0
            Processed: 2830/28800, Skipped: 0
            Processed: 2840/28800, Skipped: 0
            Processed: 2850/28800, Skipped: 0
            Processed: 2860/28800, Skipped: 0
            Processed: 2870/28800, Skipped: 0
            Processed: 2880/28800, Skipped: 0
            Processed: 2890/28800, Skipped: 0
            Processed: 2900/28800, Skipped: 0
            Processed: 2910/28800, Skipped: 0
            Processed: 2920/28800, Skipped: 0
            Processed: 2930/28800, Skipped: 0
            Processed: 2940/28800, Skipped: 0
            Processed: 2950/28800, Skipped: 0
            Processed: 2960/28800, Skipped: 0
            Processed: 2970/28800, Skipped: 0
            Processed: 2980/28800, Skipped: 0
            Processed: 2990/28800, Skipped: 0
            Processed: 3000/28800, Skipped: 0
            Processed: 3010/28800, Skipped: 0
            Processed: 3020/28800, Skipped: 0
            Processed: 3030/28800, Skipped: 0
            Processed: 3040/28800, Skipped: 0
            Processed: 3050/28800, Skipped: 0
            Processed: 3060/28800, Skipped: 0
            Processed: 3070/28800, Skipped: 0
            Processed: 3080/28800, Skipped: 0
            Processed: 3090/28800, Skipped: 0
            Processed: 3100/28800, Skipped: 0
            Processed: 3110/28800, Skipped: 0
            Processed: 3120/28800, Skipped: 0
            Processed: 3130/28800, Skipped: 0
            Processed: 3140/28800, Skipped: 0
            Processed: 3150/28800, Skipped: 0
            Processed: 3160/28800, Skipped: 0
            Processed: 3170/28800, Skipped: 0
            Processed: 3180/28800, Skipped: 0
            Processed: 3190/28800, Skipped: 0
            Processed: 3200/28800, Skipped: 0
        Processing 512 antennas...
          Processing downscale factor 1...
            Processed: 3210/28800, Skipped: 0
            Processed: 3220/28800, Skipped: 0
            Processed: 3230/28800, Skipped: 0
            Processed: 3240/28800, Skipped: 0
            Processed: 3250/28800, Skipped: 0
            Processed: 3260/28800, Skipped: 0
            Processed: 3270/28800, Skipped: 0
            Processed: 3280/28800, Skipped: 0
            Processed: 3290/28800, Skipped: 0
            Processed: 3300/28800, Skipped: 0
            Processed: 3310/28800, Skipped: 0
            Processed: 3320/28800, Skipped: 0
            Processed: 3330/28800, Skipped: 0
            Processed: 3340/28800, Skipped: 0
            Processed: 3350/28800, Skipped: 0
            Processed: 3360/28800, Skipped: 0
            Processed: 3370/28800, Skipped: 0
            Processed: 3380/28800, Skipped: 0
            Processed: 3390/28800, Skipped: 0
            Processed: 3400/28800, Skipped: 0
            Processed: 3410/28800, Skipped: 0
            Processed: 3420/28800, Skipped: 0
            Processed: 3430/28800, Skipped: 0
            Processed: 3440/28800, Skipped: 0
            Processed: 3450/28800, Skipped: 0
            Processed: 3460/28800, Skipped: 0
            Processed: 3470/28800, Skipped: 0
            Processed: 3480/28800, Skipped: 0
            Processed: 3490/28800, Skipped: 0
            Processed: 3500/28800, Skipped: 0
            Processed: 3510/28800, Skipped: 0
            Processed: 3520/28800, Skipped: 0
            Processed: 3530/28800, Skipped: 0
            Processed: 3540/28800, Skipped: 0
            Processed: 3550/28800, Skipped: 0
            Processed: 3560/28800, Skipped: 0
            Processed: 3570/28800, Skipped: 0
            Processed: 3580/28800, Skipped: 0
            Processed: 3590/28800, Skipped: 0
            Processed: 3600/28800, Skipped: 0
            Processed: 3610/28800, Skipped: 0
            Processed: 3620/28800, Skipped: 0
            Processed: 3630/28800, Skipped: 0
            Processed: 3640/28800, Skipped: 0
            Processed: 3650/28800, Skipped: 0
            Processed: 3660/28800, Skipped: 0
            Processed: 3670/28800, Skipped: 0
            Processed: 3680/28800, Skipped: 0
            Processed: 3690/28800, Skipped: 0
            Processed: 3700/28800, Skipped: 0
            Processed: 3710/28800, Skipped: 0
            Processed: 3720/28800, Skipped: 0
            Processed: 3730/28800, Skipped: 0
            Processed: 3740/28800, Skipped: 0
            Processed: 3750/28800, Skipped: 0
            Processed: 3760/28800, Skipped: 0
            Processed: 3770/28800, Skipped: 0
            Processed: 3780/28800, Skipped: 0
            Processed: 3790/28800, Skipped: 0
            Processed: 3800/28800, Skipped: 0
            Processed: 3810/28800, Skipped: 0
            Processed: 3820/28800, Skipped: 0
            Processed: 3830/28800, Skipped: 0
            Processed: 3840/28800, Skipped: 0
            Processed: 3850/28800, Skipped: 0
            Processed: 3860/28800, Skipped: 0
            Processed: 3870/28800, Skipped: 0
            Processed: 3880/28800, Skipped: 0
            Processed: 3890/28800, Skipped: 0
            Processed: 3900/28800, Skipped: 0
            Processed: 3910/28800, Skipped: 0
            Processed: 3920/28800, Skipped: 0
            Processed: 3930/28800, Skipped: 0
            Processed: 3940/28800, Skipped: 0
            Processed: 3950/28800, Skipped: 0
            Processed: 3960/28800, Skipped: 0
            Processed: 3970/28800, Skipped: 0
            Processed: 3980/28800, Skipped: 0
            Processed: 3990/28800, Skipped: 0
            Processed: 4000/28800, Skipped: 0
          Processing downscale factor 2...
            Processed: 4010/28800, Skipped: 0
            Processed: 4020/28800, Skipped: 0
            Processed: 4030/28800, Skipped: 0
            Processed: 4040/28800, Skipped: 0
            Processed: 4050/28800, Skipped: 0
            Processed: 4060/28800, Skipped: 0
            Processed: 4070/28800, Skipped: 0
            Processed: 4080/28800, Skipped: 0
            Processed: 4090/28800, Skipped: 0
            Processed: 4100/28800, Skipped: 0
            Processed: 4110/28800, Skipped: 0
            Processed: 4120/28800, Skipped: 0
            Processed: 4130/28800, Skipped: 0
            Processed: 4140/28800, Skipped: 0
            Processed: 4150/28800, Skipped: 0
            Processed: 4160/28800, Skipped: 0
            Processed: 4170/28800, Skipped: 0
            Processed: 4180/28800, Skipped: 0
            Processed: 4190/28800, Skipped: 0
            Processed: 4200/28800, Skipped: 0
            Processed: 4210/28800, Skipped: 0
            Processed: 4220/28800, Skipped: 0
            Processed: 4230/28800, Skipped: 0
            Processed: 4240/28800, Skipped: 0
            Processed: 4250/28800, Skipped: 0
            Processed: 4260/28800, Skipped: 0
            Processed: 4270/28800, Skipped: 0
            Processed: 4280/28800, Skipped: 0
            Processed: 4290/28800, Skipped: 0
            Processed: 4300/28800, Skipped: 0
            Processed: 4310/28800, Skipped: 0
            Processed: 4320/28800, Skipped: 0
            Processed: 4330/28800, Skipped: 0
            Processed: 4340/28800, Skipped: 0
            Processed: 4350/28800, Skipped: 0
            Processed: 4360/28800, Skipped: 0
            Processed: 4370/28800, Skipped: 0
            Processed: 4380/28800, Skipped: 0
            Processed: 4390/28800, Skipped: 0
            Processed: 4400/28800, Skipped: 0
            Processed: 4410/28800, Skipped: 0
            Processed: 4420/28800, Skipped: 0
            Processed: 4430/28800, Skipped: 0
            Processed: 4440/28800, Skipped: 0
            Processed: 4450/28800, Skipped: 0
            Processed: 4460/28800, Skipped: 0
            Processed: 4470/28800, Skipped: 0
            Processed: 4480/28800, Skipped: 0
            Processed: 4490/28800, Skipped: 0
            Processed: 4500/28800, Skipped: 0
            Processed: 4510/28800, Skipped: 0
            Processed: 4520/28800, Skipped: 0
            Processed: 4530/28800, Skipped: 0
            Processed: 4540/28800, Skipped: 0
            Processed: 4550/28800, Skipped: 0
            Processed: 4560/28800, Skipped: 0
            Processed: 4570/28800, Skipped: 0
            Processed: 4580/28800, Skipped: 0
            Processed: 4590/28800, Skipped: 0
            Processed: 4600/28800, Skipped: 0
            Processed: 4610/28800, Skipped: 0
            Processed: 4620/28800, Skipped: 0
            Processed: 4630/28800, Skipped: 0
            Processed: 4640/28800, Skipped: 0
            Processed: 4650/28800, Skipped: 0
            Processed: 4660/28800, Skipped: 0
            Processed: 4670/28800, Skipped: 0
            Processed: 4680/28800, Skipped: 0
            Processed: 4690/28800, Skipped: 0
            Processed: 4700/28800, Skipped: 0
            Processed: 4710/28800, Skipped: 0
            Processed: 4720/28800, Skipped: 0
            Processed: 4730/28800, Skipped: 0
            Processed: 4740/28800, Skipped: 0
            Processed: 4750/28800, Skipped: 0
            Processed: 4760/28800, Skipped: 0
            Processed: 4770/28800, Skipped: 0
            Processed: 4780/28800, Skipped: 0
            Processed: 4790/28800, Skipped: 0
            Processed: 4800/28800, Skipped: 0
        Processing 2048 antennas...
          Processing downscale factor 1...
            Processed: 4810/28800, Skipped: 0
            Processed: 4820/28800, Skipped: 0
            Processed: 4830/28800, Skipped: 0
            Processed: 4840/28800, Skipped: 0
            Processed: 4850/28800, Skipped: 0
            Processed: 4860/28800, Skipped: 0
            Processed: 4870/28800, Skipped: 0
            Processed: 4880/28800, Skipped: 0
            Processed: 4890/28800, Skipped: 0
            Processed: 4900/28800, Skipped: 0
            Processed: 4910/28800, Skipped: 0
            Processed: 4920/28800, Skipped: 0
            Processed: 4930/28800, Skipped: 0
            Processed: 4940/28800, Skipped: 0
            Processed: 4950/28800, Skipped: 0
            Processed: 4960/28800, Skipped: 0
            Processed: 4970/28800, Skipped: 0
            Processed: 4980/28800, Skipped: 0
            Processed: 4990/28800, Skipped: 0
            Processed: 5000/28800, Skipped: 0
            Processed: 5010/28800, Skipped: 0
            Processed: 5020/28800, Skipped: 0
            Processed: 5030/28800, Skipped: 0
            Processed: 5040/28800, Skipped: 0
            Processed: 5050/28800, Skipped: 0
            Processed: 5060/28800, Skipped: 0
            Processed: 5070/28800, Skipped: 0
            Processed: 5080/28800, Skipped: 0
            Processed: 5090/28800, Skipped: 0
            Processed: 5100/28800, Skipped: 0
            Processed: 5110/28800, Skipped: 0
            Processed: 5120/28800, Skipped: 0
            Processed: 5130/28800, Skipped: 0
            Processed: 5140/28800, Skipped: 0
            Processed: 5150/28800, Skipped: 0
            Processed: 5160/28800, Skipped: 0
            Processed: 5170/28800, Skipped: 0
            Processed: 5180/28800, Skipped: 0
            Processed: 5190/28800, Skipped: 0
            Processed: 5200/28800, Skipped: 0
            Processed: 5210/28800, Skipped: 0
            Processed: 5220/28800, Skipped: 0
            Processed: 5230/28800, Skipped: 0
            Processed: 5240/28800, Skipped: 0
            Processed: 5250/28800, Skipped: 0
            Processed: 5260/28800, Skipped: 0
            Processed: 5270/28800, Skipped: 0
            Processed: 5280/28800, Skipped: 0
            Processed: 5290/28800, Skipped: 0
            Processed: 5300/28800, Skipped: 0
            Processed: 5310/28800, Skipped: 0
            Processed: 5320/28800, Skipped: 0
            Processed: 5330/28800, Skipped: 0
            Processed: 5340/28800, Skipped: 0
            Processed: 5350/28800, Skipped: 0
            Processed: 5360/28800, Skipped: 0
            Processed: 5370/28800, Skipped: 0
            Processed: 5380/28800, Skipped: 0
            Processed: 5390/28800, Skipped: 0
            Processed: 5400/28800, Skipped: 0
            Processed: 5410/28800, Skipped: 0
            Processed: 5420/28800, Skipped: 0
            Processed: 5430/28800, Skipped: 0
            Processed: 5440/28800, Skipped: 0
            Processed: 5450/28800, Skipped: 0
            Processed: 5460/28800, Skipped: 0
            Processed: 5470/28800, Skipped: 0
            Processed: 5480/28800, Skipped: 0
            Processed: 5490/28800, Skipped: 0
            Processed: 5500/28800, Skipped: 0
            Processed: 5510/28800, Skipped: 0
            Processed: 5520/28800, Skipped: 0
            Processed: 5530/28800, Skipped: 0
            Processed: 5540/28800, Skipped: 0
            Processed: 5550/28800, Skipped: 0
            Processed: 5560/28800, Skipped: 0
            Processed: 5570/28800, Skipped: 0
            Processed: 5580/28800, Skipped: 0
            Processed: 5590/28800, Skipped: 0
            Processed: 5600/28800, Skipped: 0
          Processing downscale factor 2...
            Processed: 5610/28800, Skipped: 0
            Processed: 5620/28800, Skipped: 0
            Processed: 5630/28800, Skipped: 0
            Processed: 5640/28800, Skipped: 0
            Processed: 5650/28800, Skipped: 0
            Processed: 5660/28800, Skipped: 0
            Processed: 5670/28800, Skipped: 0
            Processed: 5680/28800, Skipped: 0
            Processed: 5690/28800, Skipped: 0
            Processed: 5700/28800, Skipped: 0
            Processed: 5710/28800, Skipped: 0
            Processed: 5720/28800, Skipped: 0
            Processed: 5730/28800, Skipped: 0
            Processed: 5740/28800, Skipped: 0
            Processed: 5750/28800, Skipped: 0
            Processed: 5760/28800, Skipped: 0
            Processed: 5770/28800, Skipped: 0
            Processed: 5780/28800, Skipped: 0
            Processed: 5790/28800, Skipped: 0
            Processed: 5800/28800, Skipped: 0
            Processed: 5810/28800, Skipped: 0
            Processed: 5820/28800, Skipped: 0
            Processed: 5830/28800, Skipped: 0
            Processed: 5840/28800, Skipped: 0
            Processed: 5850/28800, Skipped: 0
            Processed: 5860/28800, Skipped: 0
            Processed: 5870/28800, Skipped: 0
            Processed: 5880/28800, Skipped: 0
            Processed: 5890/28800, Skipped: 0
            Processed: 5900/28800, Skipped: 0
            Processed: 5910/28800, Skipped: 0
            Processed: 5920/28800, Skipped: 0
            Processed: 5930/28800, Skipped: 0
            Processed: 5940/28800, Skipped: 0
            Processed: 5950/28800, Skipped: 0
            Processed: 5960/28800, Skipped: 0
            Processed: 5970/28800, Skipped: 0
            Processed: 5980/28800, Skipped: 0
            Processed: 5990/28800, Skipped: 0
            Processed: 6000/28800, Skipped: 0
            Processed: 6010/28800, Skipped: 0
            Processed: 6020/28800, Skipped: 0
            Processed: 6030/28800, Skipped: 0
            Processed: 6040/28800, Skipped: 0
            Processed: 6050/28800, Skipped: 0
            Processed: 6060/28800, Skipped: 0
            Processed: 6070/28800, Skipped: 0
            Processed: 6080/28800, Skipped: 0
            Processed: 6090/28800, Skipped: 0
            Processed: 6100/28800, Skipped: 0
            Processed: 6110/28800, Skipped: 0
            Processed: 6120/28800, Skipped: 0
            Processed: 6130/28800, Skipped: 0
            Processed: 6140/28800, Skipped: 0
            Processed: 6150/28800, Skipped: 0
            Processed: 6160/28800, Skipped: 0
            Processed: 6170/28800, Skipped: 0
            Processed: 6180/28800, Skipped: 0
            Processed: 6190/28800, Skipped: 0
            Processed: 6200/28800, Skipped: 0
            Processed: 6210/28800, Skipped: 0
            Processed: 6220/28800, Skipped: 0
            Processed: 6230/28800, Skipped: 0
            Processed: 6240/28800, Skipped: 0
            Processed: 6250/28800, Skipped: 0
            Processed: 6260/28800, Skipped: 0
            Processed: 6270/28800, Skipped: 0
            Processed: 6280/28800, Skipped: 0
            Processed: 6290/28800, Skipped: 0
            Processed: 6300/28800, Skipped: 0
            Processed: 6310/28800, Skipped: 0
            Processed: 6320/28800, Skipped: 0
            Processed: 6330/28800, Skipped: 0
            Processed: 6340/28800, Skipped: 0
            Processed: 6350/28800, Skipped: 0
            Processed: 6360/28800, Skipped: 0
            Processed: 6370/28800, Skipped: 0
            Processed: 6380/28800, Skipped: 0
            Processed: 6390/28800, Skipped: 0
            Processed: 6400/28800, Skipped: 0
      Processing frequency 1300MHz...
        Processing 32 antennas...
          Processing downscale factor 1...
            Processed: 6410/28800, Skipped: 0
            Processed: 6420/28800, Skipped: 0
            Processed: 6430/28800, Skipped: 0
            Processed: 6440/28800, Skipped: 0
            Processed: 6450/28800, Skipped: 0
            Processed: 6460/28800, Skipped: 0
            Processed: 6470/28800, Skipped: 0
            Processed: 6480/28800, Skipped: 0
            Processed: 6490/28800, Skipped: 0
            Processed: 6500/28800, Skipped: 0
            Processed: 6510/28800, Skipped: 0
            Processed: 6520/28800, Skipped: 0
            Processed: 6530/28800, Skipped: 0
            Processed: 6540/28800, Skipped: 0
            Processed: 6550/28800, Skipped: 0
            Processed: 6560/28800, Skipped: 0
            Processed: 6570/28800, Skipped: 0
            Processed: 6580/28800, Skipped: 0
            Processed: 6590/28800, Skipped: 0
            Processed: 6600/28800, Skipped: 0
            Processed: 6610/28800, Skipped: 0
            Processed: 6620/28800, Skipped: 0
            Processed: 6630/28800, Skipped: 0
            Processed: 6640/28800, Skipped: 0
            Processed: 6650/28800, Skipped: 0
            Processed: 6660/28800, Skipped: 0
            Processed: 6670/28800, Skipped: 0
            Processed: 6680/28800, Skipped: 0
            Processed: 6690/28800, Skipped: 0
            Processed: 6700/28800, Skipped: 0
            Processed: 6710/28800, Skipped: 0
            Processed: 6720/28800, Skipped: 0
            Processed: 6730/28800, Skipped: 0
            Processed: 6740/28800, Skipped: 0
            Processed: 6750/28800, Skipped: 0
            Processed: 6760/28800, Skipped: 0
            Processed: 6770/28800, Skipped: 0
            Processed: 6780/28800, Skipped: 0
            Processed: 6790/28800, Skipped: 0
            Processed: 6800/28800, Skipped: 0
            Processed: 6810/28800, Skipped: 0
            Processed: 6820/28800, Skipped: 0
            Processed: 6830/28800, Skipped: 0
            Processed: 6840/28800, Skipped: 0
            Processed: 6850/28800, Skipped: 0
            Processed: 6860/28800, Skipped: 0
            Processed: 6870/28800, Skipped: 0
            Processed: 6880/28800, Skipped: 0
            Processed: 6890/28800, Skipped: 0
            Processed: 6900/28800, Skipped: 0
            Processed: 6910/28800, Skipped: 0
            Processed: 6920/28800, Skipped: 0
            Processed: 6930/28800, Skipped: 0
            Processed: 6940/28800, Skipped: 0
            Processed: 6950/28800, Skipped: 0
            Processed: 6960/28800, Skipped: 0
            Processed: 6970/28800, Skipped: 0
            Processed: 6980/28800, Skipped: 0
            Processed: 6990/28800, Skipped: 0
            Processed: 7000/28800, Skipped: 0
            Processed: 7010/28800, Skipped: 0
            Processed: 7020/28800, Skipped: 0
            Processed: 7030/28800, Skipped: 0
            Processed: 7040/28800, Skipped: 0
            Processed: 7050/28800, Skipped: 0
            Processed: 7060/28800, Skipped: 0
            Processed: 7070/28800, Skipped: 0
            Processed: 7080/28800, Skipped: 0
            Processed: 7090/28800, Skipped: 0
            Processed: 7100/28800, Skipped: 0
            Processed: 7110/28800, Skipped: 0
            Processed: 7120/28800, Skipped: 0
            Processed: 7130/28800, Skipped: 0
            Processed: 7140/28800, Skipped: 0
            Processed: 7150/28800, Skipped: 0
            Processed: 7160/28800, Skipped: 0
            Processed: 7170/28800, Skipped: 0
            Processed: 7180/28800, Skipped: 0
            Processed: 7190/28800, Skipped: 0
            Processed: 7200/28800, Skipped: 0
          Processing downscale factor 2...
            Processed: 7210/28800, Skipped: 0
            Processed: 7220/28800, Skipped: 0
            Processed: 7230/28800, Skipped: 0
            Processed: 7240/28800, Skipped: 0
            Processed: 7250/28800, Skipped: 0
            Processed: 7260/28800, Skipped: 0
            Processed: 7270/28800, Skipped: 0
            Processed: 7280/28800, Skipped: 0
            Processed: 7290/28800, Skipped: 0
            Processed: 7300/28800, Skipped: 0
            Processed: 7310/28800, Skipped: 0
            Processed: 7320/28800, Skipped: 0
            Processed: 7330/28800, Skipped: 0
            Processed: 7340/28800, Skipped: 0
            Processed: 7350/28800, Skipped: 0
            Processed: 7360/28800, Skipped: 0
            Processed: 7370/28800, Skipped: 0
            Processed: 7380/28800, Skipped: 0
            Processed: 7390/28800, Skipped: 0
            Processed: 7400/28800, Skipped: 0
            Processed: 7410/28800, Skipped: 0
            Processed: 7420/28800, Skipped: 0
            Processed: 7430/28800, Skipped: 0
            Processed: 7440/28800, Skipped: 0
            Processed: 7450/28800, Skipped: 0
            Processed: 7460/28800, Skipped: 0
            Processed: 7470/28800, Skipped: 0
            Processed: 7480/28800, Skipped: 0
            Processed: 7490/28800, Skipped: 0
            Processed: 7500/28800, Skipped: 0
            Processed: 7510/28800, Skipped: 0
            Processed: 7520/28800, Skipped: 0
            Processed: 7530/28800, Skipped: 0
            Processed: 7540/28800, Skipped: 0
            Processed: 7550/28800, Skipped: 0
            Processed: 7560/28800, Skipped: 0
            Processed: 7570/28800, Skipped: 0
            Processed: 7580/28800, Skipped: 0
            Processed: 7590/28800, Skipped: 0
            Processed: 7600/28800, Skipped: 0
            Processed: 7610/28800, Skipped: 0
            Processed: 7620/28800, Skipped: 0
            Processed: 7630/28800, Skipped: 0
            Processed: 7640/28800, Skipped: 0
            Processed: 7650/28800, Skipped: 0
            Processed: 7660/28800, Skipped: 0
            Processed: 7670/28800, Skipped: 0
            Processed: 7680/28800, Skipped: 0
            Processed: 7690/28800, Skipped: 0
            Processed: 7700/28800, Skipped: 0
            Processed: 7710/28800, Skipped: 0
            Processed: 7720/28800, Skipped: 0
            Processed: 7730/28800, Skipped: 0
            Processed: 7740/28800, Skipped: 0
            Processed: 7750/28800, Skipped: 0
            Processed: 7760/28800, Skipped: 0
            Processed: 7770/28800, Skipped: 0
            Processed: 7780/28800, Skipped: 0
            Processed: 7790/28800, Skipped: 0
            Processed: 7800/28800, Skipped: 0
            Processed: 7810/28800, Skipped: 0
            Processed: 7820/28800, Skipped: 0
            Processed: 7830/28800, Skipped: 0
            Processed: 7840/28800, Skipped: 0
            Processed: 7850/28800, Skipped: 0
            Processed: 7860/28800, Skipped: 0
            Processed: 7870/28800, Skipped: 0
            Processed: 7880/28800, Skipped: 0
            Processed: 7890/28800, Skipped: 0
            Processed: 7900/28800, Skipped: 0
            Processed: 7910/28800, Skipped: 0
            Processed: 7920/28800, Skipped: 0
            Processed: 7930/28800, Skipped: 0
            Processed: 7940/28800, Skipped: 0
            Processed: 7950/28800, Skipped: 0
            Processed: 7960/28800, Skipped: 0
            Processed: 7970/28800, Skipped: 0
            Processed: 7980/28800, Skipped: 0
            Processed: 7990/28800, Skipped: 0
            Processed: 8000/28800, Skipped: 0
        Processing 128 antennas...
          Processing downscale factor 1...
            Processed: 8010/28800, Skipped: 0
            Processed: 8020/28800, Skipped: 0
            Processed: 8030/28800, Skipped: 0
            Processed: 8040/28800, Skipped: 0
            Processed: 8050/28800, Skipped: 0
            Processed: 8060/28800, Skipped: 0
            Processed: 8070/28800, Skipped: 0
            Processed: 8080/28800, Skipped: 0
            Processed: 8090/28800, Skipped: 0
            Processed: 8100/28800, Skipped: 0
            Processed: 8110/28800, Skipped: 0
            Processed: 8120/28800, Skipped: 0
            Processed: 8130/28800, Skipped: 0
            Processed: 8140/28800, Skipped: 0
            Processed: 8150/28800, Skipped: 0
            Processed: 8160/28800, Skipped: 0
            Processed: 8170/28800, Skipped: 0
            Processed: 8180/28800, Skipped: 0
            Processed: 8190/28800, Skipped: 0
            Processed: 8200/28800, Skipped: 0
            Processed: 8210/28800, Skipped: 0
            Processed: 8220/28800, Skipped: 0
            Processed: 8230/28800, Skipped: 0
            Processed: 8240/28800, Skipped: 0
            Processed: 8250/28800, Skipped: 0
            Processed: 8260/28800, Skipped: 0
            Processed: 8270/28800, Skipped: 0
            Processed: 8280/28800, Skipped: 0
            Processed: 8290/28800, Skipped: 0
            Processed: 8300/28800, Skipped: 0
            Processed: 8310/28800, Skipped: 0
            Processed: 8320/28800, Skipped: 0
            Processed: 8330/28800, Skipped: 0
            Processed: 8340/28800, Skipped: 0
            Processed: 8350/28800, Skipped: 0
            Processed: 8360/28800, Skipped: 0
            Processed: 8370/28800, Skipped: 0
            Processed: 8380/28800, Skipped: 0
            Processed: 8390/28800, Skipped: 0
            Processed: 8400/28800, Skipped: 0
            Processed: 8410/28800, Skipped: 0
            Processed: 8420/28800, Skipped: 0
            Processed: 8430/28800, Skipped: 0
            Processed: 8440/28800, Skipped: 0
            Processed: 8450/28800, Skipped: 0
            Processed: 8460/28800, Skipped: 0
            Processed: 8470/28800, Skipped: 0
            Processed: 8480/28800, Skipped: 0
            Processed: 8490/28800, Skipped: 0
            Processed: 8500/28800, Skipped: 0
            Processed: 8510/28800, Skipped: 0
            Processed: 8520/28800, Skipped: 0
            Processed: 8530/28800, Skipped: 0
            Processed: 8540/28800, Skipped: 0
            Processed: 8550/28800, Skipped: 0
            Processed: 8560/28800, Skipped: 0
            Processed: 8570/28800, Skipped: 0
            Processed: 8580/28800, Skipped: 0
            Processed: 8590/28800, Skipped: 0
            Processed: 8600/28800, Skipped: 0
            Processed: 8610/28800, Skipped: 0
            Processed: 8620/28800, Skipped: 0
            Processed: 8630/28800, Skipped: 0
            Processed: 8640/28800, Skipped: 0
            Processed: 8650/28800, Skipped: 0
            Processed: 8660/28800, Skipped: 0
            Processed: 8670/28800, Skipped: 0
            Processed: 8680/28800, Skipped: 0
            Processed: 8690/28800, Skipped: 0
            Processed: 8700/28800, Skipped: 0
            Processed: 8710/28800, Skipped: 0
            Processed: 8720/28800, Skipped: 0
            Processed: 8730/28800, Skipped: 0
            Processed: 8740/28800, Skipped: 0
            Processed: 8750/28800, Skipped: 0
            Processed: 8760/28800, Skipped: 0
            Processed: 8770/28800, Skipped: 0
            Processed: 8780/28800, Skipped: 0
            Processed: 8790/28800, Skipped: 0
            Processed: 8800/28800, Skipped: 0
          Processing downscale factor 2...
            Processed: 8810/28800, Skipped: 0
            Processed: 8820/28800, Skipped: 0
            Processed: 8830/28800, Skipped: 0
            Processed: 8840/28800, Skipped: 0
            Processed: 8850/28800, Skipped: 0
            Processed: 8860/28800, Skipped: 0
            Processed: 8870/28800, Skipped: 0
            Processed: 8880/28800, Skipped: 0
            Processed: 8890/28800, Skipped: 0
            Processed: 8900/28800, Skipped: 0
            Processed: 8910/28800, Skipped: 0
            Processed: 8920/28800, Skipped: 0
            Processed: 8930/28800, Skipped: 0
            Processed: 8940/28800, Skipped: 0
            Processed: 8950/28800, Skipped: 0
            Processed: 8960/28800, Skipped: 0
            Processed: 8970/28800, Skipped: 0
            Processed: 8980/28800, Skipped: 0
            Processed: 8990/28800, Skipped: 0
            Processed: 9000/28800, Skipped: 0
            Processed: 9010/28800, Skipped: 0
            Processed: 9020/28800, Skipped: 0
            Processed: 9030/28800, Skipped: 0
            Processed: 9040/28800, Skipped: 0
            Processed: 9050/28800, Skipped: 0
            Processed: 9060/28800, Skipped: 0
            Processed: 9070/28800, Skipped: 0
            Processed: 9080/28800, Skipped: 0
            Processed: 9090/28800, Skipped: 0
            Processed: 9100/28800, Skipped: 0
            Processed: 9110/28800, Skipped: 0
            Processed: 9120/28800, Skipped: 0
            Processed: 9130/28800, Skipped: 0
            Processed: 9140/28800, Skipped: 0
            Processed: 9150/28800, Skipped: 0
            Processed: 9160/28800, Skipped: 0
            Processed: 9170/28800, Skipped: 0
            Processed: 9180/28800, Skipped: 0
            Processed: 9190/28800, Skipped: 0
            Processed: 9200/28800, Skipped: 0
            Processed: 9210/28800, Skipped: 0
            Processed: 9220/28800, Skipped: 0
            Processed: 9230/28800, Skipped: 0
            Processed: 9240/28800, Skipped: 0
            Processed: 9250/28800, Skipped: 0
            Processed: 9260/28800, Skipped: 0
            Processed: 9270/28800, Skipped: 0
            Processed: 9280/28800, Skipped: 0
            Processed: 9290/28800, Skipped: 0
            Processed: 9300/28800, Skipped: 0
            Processed: 9310/28800, Skipped: 0
            Processed: 9320/28800, Skipped: 0
            Processed: 9330/28800, Skipped: 0
            Processed: 9340/28800, Skipped: 0
            Processed: 9350/28800, Skipped: 0
            Processed: 9360/28800, Skipped: 0
            Processed: 9370/28800, Skipped: 0
            Processed: 9380/28800, Skipped: 0
            Processed: 9390/28800, Skipped: 0
            Processed: 9400/28800, Skipped: 0
            Processed: 9410/28800, Skipped: 0
            Processed: 9420/28800, Skipped: 0
            Processed: 9430/28800, Skipped: 0
            Processed: 9440/28800, Skipped: 0
            Processed: 9450/28800, Skipped: 0
            Processed: 9460/28800, Skipped: 0
            Processed: 9470/28800, Skipped: 0
            Processed: 9480/28800, Skipped: 0
            Processed: 9490/28800, Skipped: 0
            Processed: 9500/28800, Skipped: 0
            Processed: 9510/28800, Skipped: 0
            Processed: 9520/28800, Skipped: 0
            Processed: 9530/28800, Skipped: 0
            Processed: 9540/28800, Skipped: 0
            Processed: 9550/28800, Skipped: 0
            Processed: 9560/28800, Skipped: 0
            Processed: 9570/28800, Skipped: 0
            Processed: 9580/28800, Skipped: 0
            Processed: 9590/28800, Skipped: 0
            Processed: 9600/28800, Skipped: 0
        Processing 512 antennas...
          Processing downscale factor 1...
            Processed: 9610/28800, Skipped: 0
            Processed: 9620/28800, Skipped: 0
            Processed: 9630/28800, Skipped: 0
            Processed: 9640/28800, Skipped: 0
            Processed: 9650/28800, Skipped: 0
            Processed: 9660/28800, Skipped: 0
            Processed: 9670/28800, Skipped: 0
            Processed: 9680/28800, Skipped: 0
            Processed: 9690/28800, Skipped: 0
            Processed: 9700/28800, Skipped: 0
            Processed: 9710/28800, Skipped: 0
            Processed: 9720/28800, Skipped: 0
            Processed: 9730/28800, Skipped: 0
            Processed: 9740/28800, Skipped: 0
            Processed: 9750/28800, Skipped: 0
            Processed: 9760/28800, Skipped: 0
            Processed: 9770/28800, Skipped: 0
            Processed: 9780/28800, Skipped: 0
            Processed: 9790/28800, Skipped: 0
            Processed: 9800/28800, Skipped: 0
            Processed: 9810/28800, Skipped: 0
            Processed: 9820/28800, Skipped: 0
            Processed: 9830/28800, Skipped: 0
            Processed: 9840/28800, Skipped: 0
            Processed: 9850/28800, Skipped: 0
            Processed: 9860/28800, Skipped: 0
            Processed: 9870/28800, Skipped: 0
            Processed: 9880/28800, Skipped: 0
            Processed: 9890/28800, Skipped: 0
            Processed: 9900/28800, Skipped: 0
            Processed: 9910/28800, Skipped: 0
            Processed: 9920/28800, Skipped: 0
            Processed: 9930/28800, Skipped: 0
            Processed: 9940/28800, Skipped: 0
            Processed: 9950/28800, Skipped: 0
            Processed: 9960/28800, Skipped: 0
            Processed: 9970/28800, Skipped: 0
            Processed: 9980/28800, Skipped: 0
            Processed: 9990/28800, Skipped: 0
            Processed: 10000/28800, Skipped: 0
            Processed: 10010/28800, Skipped: 0
            Processed: 10020/28800, Skipped: 0
            Processed: 10030/28800, Skipped: 0
            Processed: 10040/28800, Skipped: 0
            Processed: 10050/28800, Skipped: 0
            Processed: 10060/28800, Skipped: 0
            Processed: 10070/28800, Skipped: 0
            Processed: 10080/28800, Skipped: 0
            Processed: 10090/28800, Skipped: 0
            Processed: 10100/28800, Skipped: 0
            Processed: 10110/28800, Skipped: 0
            Processed: 10120/28800, Skipped: 0
            Processed: 10130/28800, Skipped: 0
            Processed: 10140/28800, Skipped: 0
            Processed: 10150/28800, Skipped: 0
            Processed: 10160/28800, Skipped: 0
            Processed: 10170/28800, Skipped: 0
            Processed: 10180/28800, Skipped: 0
            Processed: 10190/28800, Skipped: 0
            Processed: 10200/28800, Skipped: 0
            Processed: 10210/28800, Skipped: 0
            Processed: 10220/28800, Skipped: 0
            Processed: 10230/28800, Skipped: 0
            Processed: 10240/28800, Skipped: 0
            Processed: 10250/28800, Skipped: 0
            Processed: 10260/28800, Skipped: 0
            Processed: 10270/28800, Skipped: 0
            Processed: 10280/28800, Skipped: 0
            Processed: 10290/28800, Skipped: 0
            Processed: 10300/28800, Skipped: 0
            Processed: 10310/28800, Skipped: 0
            Processed: 10320/28800, Skipped: 0
            Processed: 10330/28800, Skipped: 0
            Processed: 10340/28800, Skipped: 0
            Processed: 10350/28800, Skipped: 0
            Processed: 10360/28800, Skipped: 0
            Processed: 10370/28800, Skipped: 0
            Processed: 10380/28800, Skipped: 0
            Processed: 10390/28800, Skipped: 0
            Processed: 10400/28800, Skipped: 0
          Processing downscale factor 2...
            Processed: 10410/28800, Skipped: 0
            Processed: 10420/28800, Skipped: 0
            Processed: 10430/28800, Skipped: 0
            Processed: 10440/28800, Skipped: 0
            Processed: 10450/28800, Skipped: 0
            Processed: 10460/28800, Skipped: 0
            Processed: 10470/28800, Skipped: 0
            Processed: 10480/28800, Skipped: 0
            Processed: 10490/28800, Skipped: 0
            Processed: 10500/28800, Skipped: 0
            Processed: 10510/28800, Skipped: 0
            Processed: 10520/28800, Skipped: 0
            Processed: 10530/28800, Skipped: 0
            Processed: 10540/28800, Skipped: 0
            Processed: 10550/28800, Skipped: 0
            Processed: 10560/28800, Skipped: 0
            Processed: 10570/28800, Skipped: 0
            Processed: 10580/28800, Skipped: 0
            Processed: 10590/28800, Skipped: 0
            Processed: 10600/28800, Skipped: 0
            Processed: 10610/28800, Skipped: 0
            Processed: 10620/28800, Skipped: 0
            Processed: 10630/28800, Skipped: 0
            Processed: 10640/28800, Skipped: 0
            Processed: 10650/28800, Skipped: 0
            Processed: 10660/28800, Skipped: 0
            Processed: 10670/28800, Skipped: 0
            Processed: 10680/28800, Skipped: 0
            Processed: 10690/28800, Skipped: 0
            Processed: 10700/28800, Skipped: 0
            Processed: 10710/28800, Skipped: 0
            Processed: 10720/28800, Skipped: 0
            Processed: 10730/28800, Skipped: 0
            Processed: 10740/28800, Skipped: 0
            Processed: 10750/28800, Skipped: 0
            Processed: 10760/28800, Skipped: 0
            Processed: 10770/28800, Skipped: 0
            Processed: 10780/28800, Skipped: 0
            Processed: 10790/28800, Skipped: 0
            Processed: 10800/28800, Skipped: 0
            Processed: 10810/28800, Skipped: 0
            Processed: 10820/28800, Skipped: 0
            Processed: 10830/28800, Skipped: 0
            Processed: 10840/28800, Skipped: 0
            Processed: 10850/28800, Skipped: 0
            Processed: 10860/28800, Skipped: 0
            Processed: 10870/28800, Skipped: 0
            Processed: 10880/28800, Skipped: 0
            Processed: 10890/28800, Skipped: 0
            Processed: 10900/28800, Skipped: 0
            Processed: 10910/28800, Skipped: 0
            Processed: 10920/28800, Skipped: 0
            Processed: 10930/28800, Skipped: 0
            Processed: 10940/28800, Skipped: 0
            Processed: 10950/28800, Skipped: 0
            Processed: 10960/28800, Skipped: 0
            Processed: 10970/28800, Skipped: 0
            Processed: 10980/28800, Skipped: 0
            Processed: 10990/28800, Skipped: 0
            Processed: 11000/28800, Skipped: 0
            Processed: 11010/28800, Skipped: 0
            Processed: 11020/28800, Skipped: 0
            Processed: 11030/28800, Skipped: 0
            Processed: 11040/28800, Skipped: 0
            Processed: 11050/28800, Skipped: 0
            Processed: 11060/28800, Skipped: 0
            Processed: 11070/28800, Skipped: 0
            Processed: 11080/28800, Skipped: 0
            Processed: 11090/28800, Skipped: 0
            Processed: 11100/28800, Skipped: 0
            Processed: 11110/28800, Skipped: 0
            Processed: 11120/28800, Skipped: 0
            Processed: 11130/28800, Skipped: 0
            Processed: 11140/28800, Skipped: 0
            Processed: 11150/28800, Skipped: 0
            Processed: 11160/28800, Skipped: 0
            Processed: 11170/28800, Skipped: 0
            Processed: 11180/28800, Skipped: 0
            Processed: 11190/28800, Skipped: 0
            Processed: 11200/28800, Skipped: 0
        Processing 2048 antennas...
          Processing downscale factor 1...
            Processed: 11210/28800, Skipped: 0
            Processed: 11220/28800, Skipped: 0
            Processed: 11230/28800, Skipped: 0
            Processed: 11240/28800, Skipped: 0
            Processed: 11250/28800, Skipped: 0
            Processed: 11260/28800, Skipped: 0
            Processed: 11270/28800, Skipped: 0
            Processed: 11280/28800, Skipped: 0
            Processed: 11290/28800, Skipped: 0
            Processed: 11300/28800, Skipped: 0
            Processed: 11310/28800, Skipped: 0
            Processed: 11320/28800, Skipped: 0
            Processed: 11330/28800, Skipped: 0
            Processed: 11340/28800, Skipped: 0
            Processed: 11350/28800, Skipped: 0
            Processed: 11360/28800, Skipped: 0
            Processed: 11370/28800, Skipped: 0
            Processed: 11380/28800, Skipped: 0
            Processed: 11390/28800, Skipped: 0
            Processed: 11400/28800, Skipped: 0
            Processed: 11410/28800, Skipped: 0
            Processed: 11420/28800, Skipped: 0
            Processed: 11430/28800, Skipped: 0
            Processed: 11440/28800, Skipped: 0
            Processed: 11450/28800, Skipped: 0
            Processed: 11460/28800, Skipped: 0
            Processed: 11470/28800, Skipped: 0
            Processed: 11480/28800, Skipped: 0
            Processed: 11490/28800, Skipped: 0
            Processed: 11500/28800, Skipped: 0
            Processed: 11510/28800, Skipped: 0
            Processed: 11520/28800, Skipped: 0
            Processed: 11530/28800, Skipped: 0
            Processed: 11540/28800, Skipped: 0
            Processed: 11550/28800, Skipped: 0
            Processed: 11560/28800, Skipped: 0
            Processed: 11570/28800, Skipped: 0
            Processed: 11580/28800, Skipped: 0
            Processed: 11590/28800, Skipped: 0
            Processed: 11600/28800, Skipped: 0
            Processed: 11610/28800, Skipped: 0
            Processed: 11620/28800, Skipped: 0
            Processed: 11630/28800, Skipped: 0
            Processed: 11640/28800, Skipped: 0
            Processed: 11650/28800, Skipped: 0
            Processed: 11660/28800, Skipped: 0
            Processed: 11670/28800, Skipped: 0
            Processed: 11680/28800, Skipped: 0
            Processed: 11690/28800, Skipped: 0
            Processed: 11700/28800, Skipped: 0
            Processed: 11710/28800, Skipped: 0
            Processed: 11720/28800, Skipped: 0
            Processed: 11730/28800, Skipped: 0
            Processed: 11740/28800, Skipped: 0
            Processed: 11750/28800, Skipped: 0
            Processed: 11760/28800, Skipped: 0
            Processed: 11770/28800, Skipped: 0
            Processed: 11780/28800, Skipped: 0
            Processed: 11790/28800, Skipped: 0
            Processed: 11800/28800, Skipped: 0
            Processed: 11810/28800, Skipped: 0
            Processed: 11820/28800, Skipped: 0
            Processed: 11830/28800, Skipped: 0
            Processed: 11840/28800, Skipped: 0
            Processed: 11850/28800, Skipped: 0
            Processed: 11860/28800, Skipped: 0
            Processed: 11870/28800, Skipped: 0
            Processed: 11880/28800, Skipped: 0
            Processed: 11890/28800, Skipped: 0
            Processed: 11900/28800, Skipped: 0
            Processed: 11910/28800, Skipped: 0
            Processed: 11920/28800, Skipped: 0
            Processed: 11930/28800, Skipped: 0
            Processed: 11940/28800, Skipped: 0
            Processed: 11950/28800, Skipped: 0
            Processed: 11960/28800, Skipped: 0
            Processed: 11970/28800, Skipped: 0
            Processed: 11980/28800, Skipped: 0
            Processed: 11990/28800, Skipped: 0
            Processed: 12000/28800, Skipped: 0
          Processing downscale factor 2...
            Processed: 12010/28800, Skipped: 0
            Processed: 12020/28800, Skipped: 0
            Processed: 12030/28800, Skipped: 0
            Processed: 12040/28800, Skipped: 0
            Processed: 12050/28800, Skipped: 0
            Processed: 12060/28800, Skipped: 0
            Processed: 12070/28800, Skipped: 0
            Processed: 12080/28800, Skipped: 0
            Processed: 12090/28800, Skipped: 0
            Processed: 12100/28800, Skipped: 0
            Processed: 12110/28800, Skipped: 0
            Processed: 12120/28800, Skipped: 0
            Processed: 12130/28800, Skipped: 0
            Processed: 12140/28800, Skipped: 0
            Processed: 12150/28800, Skipped: 0
            Processed: 12160/28800, Skipped: 0
            Processed: 12170/28800, Skipped: 0
            Processed: 12180/28800, Skipped: 0
            Processed: 12190/28800, Skipped: 0
            Processed: 12200/28800, Skipped: 0
            Processed: 12210/28800, Skipped: 0
            Processed: 12220/28800, Skipped: 0
            Processed: 12230/28800, Skipped: 0
            Processed: 12240/28800, Skipped: 0
            Processed: 12250/28800, Skipped: 0
            Processed: 12260/28800, Skipped: 0
            Processed: 12270/28800, Skipped: 0
            Processed: 12280/28800, Skipped: 0
            Processed: 12290/28800, Skipped: 0
            Processed: 12300/28800, Skipped: 0
            Processed: 12310/28800, Skipped: 0
            Processed: 12320/28800, Skipped: 0
            Processed: 12330/28800, Skipped: 0
            Processed: 12340/28800, Skipped: 0
            Processed: 12350/28800, Skipped: 0
            Processed: 12360/28800, Skipped: 0
            Processed: 12370/28800, Skipped: 0
            Processed: 12380/28800, Skipped: 0
            Processed: 12390/28800, Skipped: 0
            Processed: 12400/28800, Skipped: 0
            Processed: 12410/28800, Skipped: 0
            Processed: 12420/28800, Skipped: 0
            Processed: 12430/28800, Skipped: 0
            Processed: 12440/28800, Skipped: 0
            Processed: 12450/28800, Skipped: 0
            Processed: 12460/28800, Skipped: 0
            Processed: 12470/28800, Skipped: 0
            Processed: 12480/28800, Skipped: 0
            Processed: 12490/28800, Skipped: 0
            Processed: 12500/28800, Skipped: 0
            Processed: 12510/28800, Skipped: 0
            Processed: 12520/28800, Skipped: 0
            Processed: 12530/28800, Skipped: 0
            Processed: 12540/28800, Skipped: 0
            Processed: 12550/28800, Skipped: 0
            Processed: 12560/28800, Skipped: 0
            Processed: 12570/28800, Skipped: 0
            Processed: 12580/28800, Skipped: 0
            Processed: 12590/28800, Skipped: 0
            Processed: 12600/28800, Skipped: 0
            Processed: 12610/28800, Skipped: 0
            Processed: 12620/28800, Skipped: 0
            Processed: 12630/28800, Skipped: 0
            Processed: 12640/28800, Skipped: 0
            Processed: 12650/28800, Skipped: 0
            Processed: 12660/28800, Skipped: 0
            Processed: 12670/28800, Skipped: 0
            Processed: 12680/28800, Skipped: 0
            Processed: 12690/28800, Skipped: 0
            Processed: 12700/28800, Skipped: 0
            Processed: 12710/28800, Skipped: 0
            Processed: 12720/28800, Skipped: 0
            Processed: 12730/28800, Skipped: 0
            Processed: 12740/28800, Skipped: 0
            Processed: 12750/28800, Skipped: 0
            Processed: 12760/28800, Skipped: 0
            Processed: 12770/28800, Skipped: 0
            Processed: 12780/28800, Skipped: 0
            Processed: 12790/28800, Skipped: 0
            Processed: 12800/28800, Skipped: 0
      Processing frequency 700MHz...
        Processing 32 antennas...
          Processing downscale factor 1...
            Processed: 12810/28800, Skipped: 0
            Processed: 12820/28800, Skipped: 0
            Processed: 12830/28800, Skipped: 0
            Processed: 12840/28800, Skipped: 0
            Processed: 12850/28800, Skipped: 0
            Processed: 12860/28800, Skipped: 0
            Processed: 12870/28800, Skipped: 0
            Processed: 12880/28800, Skipped: 0
            Processed: 12890/28800, Skipped: 0
            Processed: 12900/28800, Skipped: 0
            Processed: 12910/28800, Skipped: 0
            Processed: 12920/28800, Skipped: 0
            Processed: 12930/28800, Skipped: 0
            Processed: 12940/28800, Skipped: 0
            Processed: 12950/28800, Skipped: 0
            Processed: 12960/28800, Skipped: 0
            Processed: 12970/28800, Skipped: 0
            Processed: 12980/28800, Skipped: 0
            Processed: 12990/28800, Skipped: 0
            Processed: 13000/28800, Skipped: 0
            Processed: 13010/28800, Skipped: 0
            Processed: 13020/28800, Skipped: 0
            Processed: 13030/28800, Skipped: 0
            Processed: 13040/28800, Skipped: 0
            Processed: 13050/28800, Skipped: 0
            Processed: 13060/28800, Skipped: 0
            Processed: 13070/28800, Skipped: 0
            Processed: 13080/28800, Skipped: 0
            Processed: 13090/28800, Skipped: 0
            Processed: 13100/28800, Skipped: 0
            Processed: 13110/28800, Skipped: 0
            Processed: 13120/28800, Skipped: 0
            Processed: 13130/28800, Skipped: 0
            Processed: 13140/28800, Skipped: 0
            Processed: 13150/28800, Skipped: 0
            Processed: 13160/28800, Skipped: 0
            Processed: 13170/28800, Skipped: 0
            Processed: 13180/28800, Skipped: 0
            Processed: 13190/28800, Skipped: 0
            Processed: 13200/28800, Skipped: 0
            Processed: 13210/28800, Skipped: 0
            Processed: 13220/28800, Skipped: 0
            Processed: 13230/28800, Skipped: 0
            Processed: 13240/28800, Skipped: 0
            Processed: 13250/28800, Skipped: 0
            Processed: 13260/28800, Skipped: 0
            Processed: 13270/28800, Skipped: 0
            Processed: 13280/28800, Skipped: 0
            Processed: 13290/28800, Skipped: 0
            Processed: 13300/28800, Skipped: 0
            Processed: 13310/28800, Skipped: 0
            Processed: 13320/28800, Skipped: 0
            Processed: 13330/28800, Skipped: 0
            Processed: 13340/28800, Skipped: 0
            Processed: 13350/28800, Skipped: 0
            Processed: 13360/28800, Skipped: 0
            Processed: 13370/28800, Skipped: 0
            Processed: 13380/28800, Skipped: 0
            Processed: 13390/28800, Skipped: 0
            Processed: 13400/28800, Skipped: 0
            Processed: 13410/28800, Skipped: 0
            Processed: 13420/28800, Skipped: 0
            Processed: 13430/28800, Skipped: 0
            Processed: 13440/28800, Skipped: 0
            Processed: 13450/28800, Skipped: 0
            Processed: 13460/28800, Skipped: 0
            Processed: 13470/28800, Skipped: 0
            Processed: 13480/28800, Skipped: 0
            Processed: 13490/28800, Skipped: 0
            Processed: 13500/28800, Skipped: 0
            Processed: 13510/28800, Skipped: 0
            Processed: 13520/28800, Skipped: 0
            Processed: 13530/28800, Skipped: 0
            Processed: 13540/28800, Skipped: 0
            Processed: 13550/28800, Skipped: 0
            Processed: 13560/28800, Skipped: 0
            Processed: 13570/28800, Skipped: 0
            Processed: 13580/28800, Skipped: 0
            Processed: 13590/28800, Skipped: 0
            Processed: 13600/28800, Skipped: 0
          Processing downscale factor 2...
            Processed: 13610/28800, Skipped: 0
            Processed: 13620/28800, Skipped: 0
            Processed: 13630/28800, Skipped: 0
            Processed: 13640/28800, Skipped: 0
            Processed: 13650/28800, Skipped: 0
            Processed: 13660/28800, Skipped: 0
            Processed: 13670/28800, Skipped: 0
            Processed: 13680/28800, Skipped: 0
            Processed: 13690/28800, Skipped: 0
            Processed: 13700/28800, Skipped: 0
            Processed: 13710/28800, Skipped: 0
            Processed: 13720/28800, Skipped: 0
            Processed: 13730/28800, Skipped: 0
            Processed: 13740/28800, Skipped: 0
            Processed: 13750/28800, Skipped: 0
            Processed: 13760/28800, Skipped: 0
            Processed: 13770/28800, Skipped: 0
            Processed: 13780/28800, Skipped: 0
            Processed: 13790/28800, Skipped: 0
            Processed: 13800/28800, Skipped: 0
            Processed: 13810/28800, Skipped: 0
            Processed: 13820/28800, Skipped: 0
            Processed: 13830/28800, Skipped: 0
            Processed: 13840/28800, Skipped: 0
            Processed: 13850/28800, Skipped: 0
            Processed: 13860/28800, Skipped: 0
            Processed: 13870/28800, Skipped: 0
            Processed: 13880/28800, Skipped: 0
            Processed: 13890/28800, Skipped: 0
            Processed: 13900/28800, Skipped: 0
            Processed: 13910/28800, Skipped: 0
            Processed: 13920/28800, Skipped: 0
            Processed: 13930/28800, Skipped: 0
            Processed: 13940/28800, Skipped: 0
            Processed: 13950/28800, Skipped: 0
            Processed: 13960/28800, Skipped: 0
            Processed: 13970/28800, Skipped: 0
            Processed: 13980/28800, Skipped: 0
            Processed: 13990/28800, Skipped: 0
            Processed: 14000/28800, Skipped: 0
            Processed: 14010/28800, Skipped: 0
            Processed: 14020/28800, Skipped: 0
            Processed: 14030/28800, Skipped: 0
            Processed: 14040/28800, Skipped: 0
            Processed: 14050/28800, Skipped: 0
            Processed: 14060/28800, Skipped: 0
            Processed: 14070/28800, Skipped: 0
            Processed: 14080/28800, Skipped: 0
            Processed: 14090/28800, Skipped: 0
            Processed: 14100/28800, Skipped: 0
            Processed: 14110/28800, Skipped: 0
            Processed: 14120/28800, Skipped: 0
            Processed: 14130/28800, Skipped: 0
            Processed: 14140/28800, Skipped: 0
            Processed: 14150/28800, Skipped: 0
            Processed: 14160/28800, Skipped: 0
            Processed: 14170/28800, Skipped: 0
            Processed: 14180/28800, Skipped: 0
            Processed: 14190/28800, Skipped: 0
            Processed: 14200/28800, Skipped: 0
            Processed: 14210/28800, Skipped: 0
            Processed: 14220/28800, Skipped: 0
            Processed: 14230/28800, Skipped: 0
            Processed: 14240/28800, Skipped: 0
            Processed: 14250/28800, Skipped: 0
            Processed: 14260/28800, Skipped: 0
            Processed: 14270/28800, Skipped: 0
            Processed: 14280/28800, Skipped: 0
            Processed: 14290/28800, Skipped: 0
            Processed: 14300/28800, Skipped: 0
            Processed: 14310/28800, Skipped: 0
            Processed: 14320/28800, Skipped: 0
            Processed: 14330/28800, Skipped: 0
            Processed: 14340/28800, Skipped: 0
            Processed: 14350/28800, Skipped: 0
            Processed: 14360/28800, Skipped: 0
            Processed: 14370/28800, Skipped: 0
            Processed: 14380/28800, Skipped: 0
            Processed: 14390/28800, Skipped: 0
            Processed: 14400/28800, Skipped: 0
        Processing 128 antennas...
          Processing downscale factor 1...
            Processed: 14410/28800, Skipped: 0
            Processed: 14420/28800, Skipped: 0
            Processed: 14430/28800, Skipped: 0
            Processed: 14440/28800, Skipped: 0
            Processed: 14450/28800, Skipped: 0
            Processed: 14460/28800, Skipped: 0
            Processed: 14470/28800, Skipped: 0
            Processed: 14480/28800, Skipped: 0
            Processed: 14490/28800, Skipped: 0
            Processed: 14500/28800, Skipped: 0
            Processed: 14510/28800, Skipped: 0
            Processed: 14520/28800, Skipped: 0
            Processed: 14530/28800, Skipped: 0
            Processed: 14540/28800, Skipped: 0
            Processed: 14550/28800, Skipped: 0
            Processed: 14560/28800, Skipped: 0
            Processed: 14570/28800, Skipped: 0
            Processed: 14580/28800, Skipped: 0
            Processed: 14590/28800, Skipped: 0
            Processed: 14600/28800, Skipped: 0
            Processed: 14610/28800, Skipped: 0
            Processed: 14620/28800, Skipped: 0
            Processed: 14630/28800, Skipped: 0
            Processed: 14640/28800, Skipped: 0
            Processed: 14650/28800, Skipped: 0
            Processed: 14660/28800, Skipped: 0
            Processed: 14670/28800, Skipped: 0
            Processed: 14680/28800, Skipped: 0
            Processed: 14690/28800, Skipped: 0
            Processed: 14700/28800, Skipped: 0
            Processed: 14710/28800, Skipped: 0
            Processed: 14720/28800, Skipped: 0
            Processed: 14730/28800, Skipped: 0
            Processed: 14740/28800, Skipped: 0
            Processed: 14750/28800, Skipped: 0
            Processed: 14760/28800, Skipped: 0
            Processed: 14770/28800, Skipped: 0
            Processed: 14780/28800, Skipped: 0
            Processed: 14790/28800, Skipped: 0
            Processed: 14800/28800, Skipped: 0
            Processed: 14810/28800, Skipped: 0
            Processed: 14820/28800, Skipped: 0
            Processed: 14830/28800, Skipped: 0
            Processed: 14840/28800, Skipped: 0
            Processed: 14850/28800, Skipped: 0
            Processed: 14860/28800, Skipped: 0
            Processed: 14870/28800, Skipped: 0
            Processed: 14880/28800, Skipped: 0
            Processed: 14890/28800, Skipped: 0
            Processed: 14900/28800, Skipped: 0
            Processed: 14910/28800, Skipped: 0
            Processed: 14920/28800, Skipped: 0
            Processed: 14930/28800, Skipped: 0
            Processed: 14940/28800, Skipped: 0
            Processed: 14950/28800, Skipped: 0
            Processed: 14960/28800, Skipped: 0
            Processed: 14970/28800, Skipped: 0
            Processed: 14980/28800, Skipped: 0
            Processed: 14990/28800, Skipped: 0
            Processed: 15000/28800, Skipped: 0
            Processed: 15010/28800, Skipped: 0
            Processed: 15020/28800, Skipped: 0
            Processed: 15030/28800, Skipped: 0
            Processed: 15040/28800, Skipped: 0
            Processed: 15050/28800, Skipped: 0
            Processed: 15060/28800, Skipped: 0
            Processed: 15070/28800, Skipped: 0
            Processed: 15080/28800, Skipped: 0
            Processed: 15090/28800, Skipped: 0
            Processed: 15100/28800, Skipped: 0
            Processed: 15110/28800, Skipped: 0
            Processed: 15120/28800, Skipped: 0
            Processed: 15130/28800, Skipped: 0
            Processed: 15140/28800, Skipped: 0
            Processed: 15150/28800, Skipped: 0
            Processed: 15160/28800, Skipped: 0
            Processed: 15170/28800, Skipped: 0
            Processed: 15180/28800, Skipped: 0
            Processed: 15190/28800, Skipped: 0
            Processed: 15200/28800, Skipped: 0
          Processing downscale factor 2...
            Processed: 15210/28800, Skipped: 0
            Processed: 15220/28800, Skipped: 0
            Processed: 15230/28800, Skipped: 0
            Processed: 15240/28800, Skipped: 0
            Processed: 15250/28800, Skipped: 0
            Processed: 15260/28800, Skipped: 0
            Processed: 15270/28800, Skipped: 0
            Processed: 15280/28800, Skipped: 0
            Processed: 15290/28800, Skipped: 0
            Processed: 15300/28800, Skipped: 0
            Processed: 15310/28800, Skipped: 0
            Processed: 15320/28800, Skipped: 0
            Processed: 15330/28800, Skipped: 0
            Processed: 15340/28800, Skipped: 0
            Processed: 15350/28800, Skipped: 0
            Processed: 15360/28800, Skipped: 0
            Processed: 15370/28800, Skipped: 0
            Processed: 15380/28800, Skipped: 0
            Processed: 15390/28800, Skipped: 0
            Processed: 15400/28800, Skipped: 0
            Processed: 15410/28800, Skipped: 0
            Processed: 15420/28800, Skipped: 0
            Processed: 15430/28800, Skipped: 0
            Processed: 15440/28800, Skipped: 0
            Processed: 15450/28800, Skipped: 0
            Processed: 15460/28800, Skipped: 0
            Processed: 15470/28800, Skipped: 0
            Processed: 15480/28800, Skipped: 0
            Processed: 15490/28800, Skipped: 0
            Processed: 15500/28800, Skipped: 0
            Processed: 15510/28800, Skipped: 0
            Processed: 15520/28800, Skipped: 0
            Processed: 15530/28800, Skipped: 0
            Processed: 15540/28800, Skipped: 0
            Processed: 15550/28800, Skipped: 0
            Processed: 15560/28800, Skipped: 0
            Processed: 15570/28800, Skipped: 0
            Processed: 15580/28800, Skipped: 0
            Processed: 15590/28800, Skipped: 0
            Processed: 15600/28800, Skipped: 0
            Processed: 15610/28800, Skipped: 0
            Processed: 15620/28800, Skipped: 0
            Processed: 15630/28800, Skipped: 0
            Processed: 15640/28800, Skipped: 0
            Processed: 15650/28800, Skipped: 0
            Processed: 15660/28800, Skipped: 0
            Processed: 15670/28800, Skipped: 0
            Processed: 15680/28800, Skipped: 0
            Processed: 15690/28800, Skipped: 0
            Processed: 15700/28800, Skipped: 0
            Processed: 15710/28800, Skipped: 0
            Processed: 15720/28800, Skipped: 0
            Processed: 15730/28800, Skipped: 0
            Processed: 15740/28800, Skipped: 0
            Processed: 15750/28800, Skipped: 0
            Processed: 15760/28800, Skipped: 0
            Processed: 15770/28800, Skipped: 0
            Processed: 15780/28800, Skipped: 0
            Processed: 15790/28800, Skipped: 0
            Processed: 15800/28800, Skipped: 0
            Processed: 15810/28800, Skipped: 0
            Processed: 15820/28800, Skipped: 0
            Processed: 15830/28800, Skipped: 0
            Processed: 15840/28800, Skipped: 0
            Processed: 15850/28800, Skipped: 0
            Processed: 15860/28800, Skipped: 0
            Processed: 15870/28800, Skipped: 0
            Processed: 15880/28800, Skipped: 0
            Processed: 15890/28800, Skipped: 0
            Processed: 15900/28800, Skipped: 0
            Processed: 15910/28800, Skipped: 0
            Processed: 15920/28800, Skipped: 0
            Processed: 15930/28800, Skipped: 0
            Processed: 15940/28800, Skipped: 0
            Processed: 15950/28800, Skipped: 0
            Processed: 15960/28800, Skipped: 0
            Processed: 15970/28800, Skipped: 0
            Processed: 15980/28800, Skipped: 0
            Processed: 15990/28800, Skipped: 0
            Processed: 16000/28800, Skipped: 0
        Processing 512 antennas...
          Processing downscale factor 1...
            Processed: 16010/28800, Skipped: 0
            Processed: 16020/28800, Skipped: 0
            Processed: 16030/28800, Skipped: 0
            Processed: 16040/28800, Skipped: 0
            Processed: 16050/28800, Skipped: 0
            Processed: 16060/28800, Skipped: 0
            Processed: 16070/28800, Skipped: 0
            Processed: 16080/28800, Skipped: 0
            Processed: 16090/28800, Skipped: 0
            Processed: 16100/28800, Skipped: 0
            Processed: 16110/28800, Skipped: 0
            Processed: 16120/28800, Skipped: 0
            Processed: 16130/28800, Skipped: 0
            Processed: 16140/28800, Skipped: 0
            Processed: 16150/28800, Skipped: 0
            Processed: 16160/28800, Skipped: 0
            Processed: 16170/28800, Skipped: 0
            Processed: 16180/28800, Skipped: 0
            Processed: 16190/28800, Skipped: 0
            Processed: 16200/28800, Skipped: 0
            Processed: 16210/28800, Skipped: 0
            Processed: 16220/28800, Skipped: 0
            Processed: 16230/28800, Skipped: 0
            Processed: 16240/28800, Skipped: 0
            Processed: 16250/28800, Skipped: 0
            Processed: 16260/28800, Skipped: 0
            Processed: 16270/28800, Skipped: 0
            Processed: 16280/28800, Skipped: 0
            Processed: 16290/28800, Skipped: 0
            Processed: 16300/28800, Skipped: 0
            Processed: 16310/28800, Skipped: 0
            Processed: 16320/28800, Skipped: 0
            Processed: 16330/28800, Skipped: 0
            Processed: 16340/28800, Skipped: 0
            Processed: 16350/28800, Skipped: 0
            Processed: 16360/28800, Skipped: 0
            Processed: 16370/28800, Skipped: 0
            Processed: 16380/28800, Skipped: 0
            Processed: 16390/28800, Skipped: 0
            Processed: 16400/28800, Skipped: 0
            Processed: 16410/28800, Skipped: 0
            Processed: 16420/28800, Skipped: 0
            Processed: 16430/28800, Skipped: 0
            Processed: 16440/28800, Skipped: 0
            Processed: 16450/28800, Skipped: 0
            Processed: 16460/28800, Skipped: 0
            Processed: 16470/28800, Skipped: 0
            Processed: 16480/28800, Skipped: 0
            Processed: 16490/28800, Skipped: 0
            Processed: 16500/28800, Skipped: 0
            Processed: 16510/28800, Skipped: 0
            Processed: 16520/28800, Skipped: 0
            Processed: 16530/28800, Skipped: 0
            Processed: 16540/28800, Skipped: 0
            Processed: 16550/28800, Skipped: 0
            Processed: 16560/28800, Skipped: 0
            Processed: 16570/28800, Skipped: 0
            Processed: 16580/28800, Skipped: 0
            Processed: 16590/28800, Skipped: 0
            Processed: 16600/28800, Skipped: 0
            Processed: 16610/28800, Skipped: 0
            Processed: 16620/28800, Skipped: 0
            Processed: 16630/28800, Skipped: 0
            Processed: 16640/28800, Skipped: 0
            Processed: 16650/28800, Skipped: 0
            Processed: 16660/28800, Skipped: 0
            Processed: 16670/28800, Skipped: 0
            Processed: 16680/28800, Skipped: 0
            Processed: 16690/28800, Skipped: 0
            Processed: 16700/28800, Skipped: 0
            Processed: 16710/28800, Skipped: 0
            Processed: 16720/28800, Skipped: 0
            Processed: 16730/28800, Skipped: 0
            Processed: 16740/28800, Skipped: 0
            Processed: 16750/28800, Skipped: 0
            Processed: 16760/28800, Skipped: 0
            Processed: 16770/28800, Skipped: 0
            Processed: 16780/28800, Skipped: 0
            Processed: 16790/28800, Skipped: 0
            Processed: 16800/28800, Skipped: 0
          Processing downscale factor 2...
            Processed: 16810/28800, Skipped: 0
            Processed: 16820/28800, Skipped: 0
            Processed: 16830/28800, Skipped: 0
            Processed: 16840/28800, Skipped: 0
            Processed: 16850/28800, Skipped: 0
            Processed: 16860/28800, Skipped: 0
            Processed: 16870/28800, Skipped: 0
            Processed: 16880/28800, Skipped: 0
            Processed: 16890/28800, Skipped: 0
            Processed: 16900/28800, Skipped: 0
            Processed: 16910/28800, Skipped: 0
            Processed: 16920/28800, Skipped: 0
            Processed: 16930/28800, Skipped: 0
            Processed: 16940/28800, Skipped: 0
            Processed: 16950/28800, Skipped: 0
            Processed: 16960/28800, Skipped: 0
            Processed: 16970/28800, Skipped: 0
            Processed: 16980/28800, Skipped: 0
            Processed: 16990/28800, Skipped: 0
            Processed: 17000/28800, Skipped: 0
            Processed: 17010/28800, Skipped: 0
            Processed: 17020/28800, Skipped: 0
            Processed: 17030/28800, Skipped: 0
            Processed: 17040/28800, Skipped: 0
            Processed: 17050/28800, Skipped: 0
            Processed: 17060/28800, Skipped: 0
            Processed: 17070/28800, Skipped: 0
            Processed: 17080/28800, Skipped: 0
            Processed: 17090/28800, Skipped: 0
            Processed: 17100/28800, Skipped: 0
            Processed: 17110/28800, Skipped: 0
            Processed: 17120/28800, Skipped: 0
            Processed: 17130/28800, Skipped: 0
            Processed: 17140/28800, Skipped: 0
            Processed: 17150/28800, Skipped: 0
            Processed: 17160/28800, Skipped: 0
            Processed: 17170/28800, Skipped: 0
            Processed: 17180/28800, Skipped: 0
            Processed: 17190/28800, Skipped: 0
            Processed: 17200/28800, Skipped: 0
            Processed: 17210/28800, Skipped: 0
            Processed: 17220/28800, Skipped: 0
            Processed: 17230/28800, Skipped: 0
            Processed: 17240/28800, Skipped: 0
            Processed: 17250/28800, Skipped: 0
            Processed: 17260/28800, Skipped: 0
            Processed: 17270/28800, Skipped: 0
            Processed: 17280/28800, Skipped: 0
            Processed: 17290/28800, Skipped: 0
            Processed: 17300/28800, Skipped: 0
            Processed: 17310/28800, Skipped: 0
            Processed: 17320/28800, Skipped: 0
            Processed: 17330/28800, Skipped: 0
            Processed: 17340/28800, Skipped: 0
            Processed: 17350/28800, Skipped: 0
            Processed: 17360/28800, Skipped: 0
            Processed: 17370/28800, Skipped: 0
            Processed: 17380/28800, Skipped: 0
            Processed: 17390/28800, Skipped: 0
            Processed: 17400/28800, Skipped: 0
            Processed: 17410/28800, Skipped: 0
            Processed: 17420/28800, Skipped: 0
            Processed: 17430/28800, Skipped: 0
            Processed: 17440/28800, Skipped: 0
            Processed: 17450/28800, Skipped: 0
            Processed: 17460/28800, Skipped: 0
            Processed: 17470/28800, Skipped: 0
            Processed: 17480/28800, Skipped: 0
            Processed: 17490/28800, Skipped: 0
            Processed: 17500/28800, Skipped: 0
            Processed: 17510/28800, Skipped: 0
            Processed: 17520/28800, Skipped: 0
            Processed: 17530/28800, Skipped: 0
            Processed: 17540/28800, Skipped: 0
            Processed: 17550/28800, Skipped: 0
            Processed: 17560/28800, Skipped: 0
            Processed: 17570/28800, Skipped: 0
            Processed: 17580/28800, Skipped: 0
            Processed: 17590/28800, Skipped: 0
            Processed: 17600/28800, Skipped: 0
        Processing 2048 antennas...
          Processing downscale factor 1...
            Processed: 17610/28800, Skipped: 0
            Processed: 17620/28800, Skipped: 0
            Processed: 17630/28800, Skipped: 0
            Processed: 17640/28800, Skipped: 0
            Processed: 17650/28800, Skipped: 0
            Processed: 17660/28800, Skipped: 0
            Processed: 17670/28800, Skipped: 0
            Processed: 17680/28800, Skipped: 0
            Processed: 17690/28800, Skipped: 0
            Processed: 17700/28800, Skipped: 0
            Processed: 17710/28800, Skipped: 0
            Processed: 17720/28800, Skipped: 0
            Processed: 17730/28800, Skipped: 0
            Processed: 17740/28800, Skipped: 0
            Processed: 17750/28800, Skipped: 0
            Processed: 17760/28800, Skipped: 0
            Processed: 17770/28800, Skipped: 0
            Processed: 17780/28800, Skipped: 0
            Processed: 17790/28800, Skipped: 0
            Processed: 17800/28800, Skipped: 0
            Processed: 17810/28800, Skipped: 0
            Processed: 17820/28800, Skipped: 0
            Processed: 17830/28800, Skipped: 0
            Processed: 17840/28800, Skipped: 0
            Processed: 17850/28800, Skipped: 0
            Processed: 17860/28800, Skipped: 0
            Processed: 17870/28800, Skipped: 0
            Processed: 17880/28800, Skipped: 0
            Processed: 17890/28800, Skipped: 0
            Processed: 17900/28800, Skipped: 0
            Processed: 17910/28800, Skipped: 0
            Processed: 17920/28800, Skipped: 0
            Processed: 17930/28800, Skipped: 0
            Processed: 17940/28800, Skipped: 0
            Processed: 17950/28800, Skipped: 0
            Processed: 17960/28800, Skipped: 0
            Processed: 17970/28800, Skipped: 0
            Processed: 17980/28800, Skipped: 0
            Processed: 17990/28800, Skipped: 0
            Processed: 18000/28800, Skipped: 0
            Processed: 18010/28800, Skipped: 0
            Processed: 18020/28800, Skipped: 0
            Processed: 18030/28800, Skipped: 0
            Processed: 18040/28800, Skipped: 0
            Processed: 18050/28800, Skipped: 0
            Processed: 18060/28800, Skipped: 0
            Processed: 18070/28800, Skipped: 0
            Processed: 18080/28800, Skipped: 0
            Processed: 18090/28800, Skipped: 0
            Processed: 18100/28800, Skipped: 0
            Processed: 18110/28800, Skipped: 0
            Processed: 18120/28800, Skipped: 0
            Processed: 18130/28800, Skipped: 0
            Processed: 18140/28800, Skipped: 0
            Processed: 18150/28800, Skipped: 0
            Processed: 18160/28800, Skipped: 0
            Processed: 18170/28800, Skipped: 0
            Processed: 18180/28800, Skipped: 0
            Processed: 18190/28800, Skipped: 0
            Processed: 18200/28800, Skipped: 0
            Processed: 18210/28800, Skipped: 0
            Processed: 18220/28800, Skipped: 0
            Processed: 18230/28800, Skipped: 0
            Processed: 18240/28800, Skipped: 0
            Processed: 18250/28800, Skipped: 0
            Processed: 18260/28800, Skipped: 0
            Processed: 18270/28800, Skipped: 0
            Processed: 18280/28800, Skipped: 0
            Processed: 18290/28800, Skipped: 0
            Processed: 18300/28800, Skipped: 0
            Processed: 18310/28800, Skipped: 0
            Processed: 18320/28800, Skipped: 0
            Processed: 18330/28800, Skipped: 0
            Processed: 18340/28800, Skipped: 0
            Processed: 18350/28800, Skipped: 0
            Processed: 18360/28800, Skipped: 0
            Processed: 18370/28800, Skipped: 0
            Processed: 18380/28800, Skipped: 0
            Processed: 18390/28800, Skipped: 0
            Processed: 18400/28800, Skipped: 0
          Processing downscale factor 2...
            Processed: 18410/28800, Skipped: 0
            Processed: 18420/28800, Skipped: 0
            Processed: 18430/28800, Skipped: 0
            Processed: 18440/28800, Skipped: 0
            Processed: 18450/28800, Skipped: 0
            Processed: 18460/28800, Skipped: 0
            Processed: 18470/28800, Skipped: 0
            Processed: 18480/28800, Skipped: 0
            Processed: 18490/28800, Skipped: 0
            Processed: 18500/28800, Skipped: 0
            Processed: 18510/28800, Skipped: 0
            Processed: 18520/28800, Skipped: 0
            Processed: 18530/28800, Skipped: 0
            Processed: 18540/28800, Skipped: 0
            Processed: 18550/28800, Skipped: 0
            Processed: 18560/28800, Skipped: 0
            Processed: 18570/28800, Skipped: 0
            Processed: 18580/28800, Skipped: 0
            Processed: 18590/28800, Skipped: 0
            Processed: 18600/28800, Skipped: 0
            Processed: 18610/28800, Skipped: 0
            Processed: 18620/28800, Skipped: 0
            Processed: 18630/28800, Skipped: 0
            Processed: 18640/28800, Skipped: 0
            Processed: 18650/28800, Skipped: 0
            Processed: 18660/28800, Skipped: 0
            Processed: 18670/28800, Skipped: 0
            Processed: 18680/28800, Skipped: 0
            Processed: 18690/28800, Skipped: 0
            Processed: 18700/28800, Skipped: 0
            Processed: 18710/28800, Skipped: 0
            Processed: 18720/28800, Skipped: 0
            Processed: 18730/28800, Skipped: 0
            Processed: 18740/28800, Skipped: 0
            Processed: 18750/28800, Skipped: 0
            Processed: 18760/28800, Skipped: 0
            Processed: 18770/28800, Skipped: 0
            Processed: 18780/28800, Skipped: 0
            Processed: 18790/28800, Skipped: 0
            Processed: 18800/28800, Skipped: 0
            Processed: 18810/28800, Skipped: 0
            Processed: 18820/28800, Skipped: 0
            Processed: 18830/28800, Skipped: 0
            Processed: 18840/28800, Skipped: 0
            Processed: 18850/28800, Skipped: 0
            Processed: 18860/28800, Skipped: 0
            Processed: 18870/28800, Skipped: 0
            Processed: 18880/28800, Skipped: 0
            Processed: 18890/28800, Skipped: 0
            Processed: 18900/28800, Skipped: 0
            Processed: 18910/28800, Skipped: 0
            Processed: 18920/28800, Skipped: 0
            Processed: 18930/28800, Skipped: 0
            Processed: 18940/28800, Skipped: 0
            Processed: 18950/28800, Skipped: 0
            Processed: 18960/28800, Skipped: 0
            Processed: 18970/28800, Skipped: 0
            Processed: 18980/28800, Skipped: 0
            Processed: 18990/28800, Skipped: 0
            Processed: 19000/28800, Skipped: 0
            Processed: 19010/28800, Skipped: 0
            Processed: 19020/28800, Skipped: 0
            Processed: 19030/28800, Skipped: 0
            Processed: 19040/28800, Skipped: 0
            Processed: 19050/28800, Skipped: 0
            Processed: 19060/28800, Skipped: 0
            Processed: 19070/28800, Skipped: 0
            Processed: 19080/28800, Skipped: 0
            Processed: 19090/28800, Skipped: 0
            Processed: 19100/28800, Skipped: 0
            Processed: 19110/28800, Skipped: 0
            Processed: 19120/28800, Skipped: 0
            Processed: 19130/28800, Skipped: 0
            Processed: 19140/28800, Skipped: 0
            Processed: 19150/28800, Skipped: 0
            Processed: 19160/28800, Skipped: 0
            Processed: 19170/28800, Skipped: 0
            Processed: 19180/28800, Skipped: 0
            Processed: 19190/28800, Skipped: 0
            Processed: 19200/28800, Skipped: 0
      Processing frequency 10MHz...
        Processing 32 antennas...
          Processing downscale factor 1...
            Processed: 19210/28800, Skipped: 0
            Processed: 19220/28800, Skipped: 0
            Processed: 19230/28800, Skipped: 0
            Processed: 19240/28800, Skipped: 0
            Processed: 19250/28800, Skipped: 0
            Processed: 19260/28800, Skipped: 0
            Processed: 19270/28800, Skipped: 0
            Processed: 19280/28800, Skipped: 0
            Processed: 19290/28800, Skipped: 0
            Processed: 19300/28800, Skipped: 0
            Processed: 19310/28800, Skipped: 0
            Processed: 19320/28800, Skipped: 0
            Processed: 19330/28800, Skipped: 0
            Processed: 19340/28800, Skipped: 0
            Processed: 19350/28800, Skipped: 0
            Processed: 19360/28800, Skipped: 0
            Processed: 19370/28800, Skipped: 0
            Processed: 19380/28800, Skipped: 0
            Processed: 19390/28800, Skipped: 0
            Processed: 19400/28800, Skipped: 0
            Processed: 19410/28800, Skipped: 0
            Processed: 19420/28800, Skipped: 0
            Processed: 19430/28800, Skipped: 0
            Processed: 19440/28800, Skipped: 0
            Processed: 19450/28800, Skipped: 0
            Processed: 19460/28800, Skipped: 0
            Processed: 19470/28800, Skipped: 0
            Processed: 19480/28800, Skipped: 0
            Processed: 19490/28800, Skipped: 0
            Processed: 19500/28800, Skipped: 0
            Processed: 19510/28800, Skipped: 0
            Processed: 19520/28800, Skipped: 0
            Processed: 19530/28800, Skipped: 0
            Processed: 19540/28800, Skipped: 0
            Processed: 19550/28800, Skipped: 0
            Processed: 19560/28800, Skipped: 0
            Processed: 19570/28800, Skipped: 0
            Processed: 19580/28800, Skipped: 0
            Processed: 19590/28800, Skipped: 0
            Processed: 19600/28800, Skipped: 0
            Processed: 19610/28800, Skipped: 0
            Processed: 19620/28800, Skipped: 0
            Processed: 19630/28800, Skipped: 0
            Processed: 19640/28800, Skipped: 0
            Processed: 19650/28800, Skipped: 0
            Processed: 19660/28800, Skipped: 0
            Processed: 19670/28800, Skipped: 0
            Processed: 19680/28800, Skipped: 0
            Processed: 19690/28800, Skipped: 0
            Processed: 19700/28800, Skipped: 0
            Processed: 19710/28800, Skipped: 0
            Processed: 19720/28800, Skipped: 0
            Processed: 19730/28800, Skipped: 0
            Processed: 19740/28800, Skipped: 0
            Processed: 19750/28800, Skipped: 0
            Processed: 19760/28800, Skipped: 0
            Processed: 19770/28800, Skipped: 0
            Processed: 19780/28800, Skipped: 0
            Processed: 19790/28800, Skipped: 0
            Processed: 19800/28800, Skipped: 0
            Processed: 19810/28800, Skipped: 0
            Processed: 19820/28800, Skipped: 0
            Processed: 19830/28800, Skipped: 0
            Processed: 19840/28800, Skipped: 0
            Processed: 19850/28800, Skipped: 0
            Processed: 19860/28800, Skipped: 0
            Processed: 19870/28800, Skipped: 0
            Processed: 19880/28800, Skipped: 0
            Processed: 19890/28800, Skipped: 0
            Processed: 19900/28800, Skipped: 0
            Processed: 19910/28800, Skipped: 0
            Processed: 19920/28800, Skipped: 0
            Processed: 19930/28800, Skipped: 0
            Processed: 19940/28800, Skipped: 0
            Processed: 19950/28800, Skipped: 0
            Processed: 19960/28800, Skipped: 0
            Processed: 19970/28800, Skipped: 0
            Processed: 19980/28800, Skipped: 0
            Processed: 19990/28800, Skipped: 0
            Processed: 20000/28800, Skipped: 0
          Processing downscale factor 2...
            Processed: 20010/28800, Skipped: 0
            Processed: 20020/28800, Skipped: 0
            Processed: 20030/28800, Skipped: 0
            Processed: 20040/28800, Skipped: 0
            Processed: 20050/28800, Skipped: 0
            Processed: 20060/28800, Skipped: 0
            Processed: 20070/28800, Skipped: 0
            Processed: 20080/28800, Skipped: 0
            Processed: 20090/28800, Skipped: 0
            Processed: 20100/28800, Skipped: 0
            Processed: 20110/28800, Skipped: 0
            Processed: 20120/28800, Skipped: 0
            Processed: 20130/28800, Skipped: 0
            Processed: 20140/28800, Skipped: 0
            Processed: 20150/28800, Skipped: 0
            Processed: 20160/28800, Skipped: 0
            Processed: 20170/28800, Skipped: 0
            Processed: 20180/28800, Skipped: 0
            Processed: 20190/28800, Skipped: 0
            Processed: 20200/28800, Skipped: 0
            Processed: 20210/28800, Skipped: 0
            Processed: 20220/28800, Skipped: 0
            Processed: 20230/28800, Skipped: 0
            Processed: 20240/28800, Skipped: 0
            Processed: 20250/28800, Skipped: 0
            Processed: 20260/28800, Skipped: 0
            Processed: 20270/28800, Skipped: 0
            Processed: 20280/28800, Skipped: 0
            Processed: 20290/28800, Skipped: 0
            Processed: 20300/28800, Skipped: 0
            Processed: 20310/28800, Skipped: 0
            Processed: 20320/28800, Skipped: 0
            Processed: 20330/28800, Skipped: 0
            Processed: 20340/28800, Skipped: 0
            Processed: 20350/28800, Skipped: 0
            Processed: 20360/28800, Skipped: 0
            Processed: 20370/28800, Skipped: 0
            Processed: 20380/28800, Skipped: 0
            Processed: 20390/28800, Skipped: 0
            Processed: 20400/28800, Skipped: 0
            Processed: 20410/28800, Skipped: 0
            Processed: 20420/28800, Skipped: 0
            Processed: 20430/28800, Skipped: 0
            Processed: 20440/28800, Skipped: 0
            Processed: 20450/28800, Skipped: 0
            Processed: 20460/28800, Skipped: 0
            Processed: 20470/28800, Skipped: 0
            Processed: 20480/28800, Skipped: 0
            Processed: 20490/28800, Skipped: 0
            Processed: 20500/28800, Skipped: 0
            Processed: 20510/28800, Skipped: 0
            Processed: 20520/28800, Skipped: 0
            Processed: 20530/28800, Skipped: 0
            Processed: 20540/28800, Skipped: 0
            Processed: 20550/28800, Skipped: 0
            Processed: 20560/28800, Skipped: 0
            Processed: 20570/28800, Skipped: 0
            Processed: 20580/28800, Skipped: 0
            Processed: 20590/28800, Skipped: 0
            Processed: 20600/28800, Skipped: 0
            Processed: 20610/28800, Skipped: 0
            Processed: 20620/28800, Skipped: 0
            Processed: 20630/28800, Skipped: 0
            Processed: 20640/28800, Skipped: 0
            Processed: 20650/28800, Skipped: 0
            Processed: 20660/28800, Skipped: 0
            Processed: 20670/28800, Skipped: 0
            Processed: 20680/28800, Skipped: 0
            Processed: 20690/28800, Skipped: 0
            Processed: 20700/28800, Skipped: 0
            Processed: 20710/28800, Skipped: 0
            Processed: 20720/28800, Skipped: 0
            Processed: 20730/28800, Skipped: 0
            Processed: 20740/28800, Skipped: 0
            Processed: 20750/28800, Skipped: 0
            Processed: 20760/28800, Skipped: 0
            Processed: 20770/28800, Skipped: 0
            Processed: 20780/28800, Skipped: 0
            Processed: 20790/28800, Skipped: 0
            Processed: 20800/28800, Skipped: 0
        Processing 128 antennas...
          Processing downscale factor 1...
            Processed: 20810/28800, Skipped: 0
            Processed: 20820/28800, Skipped: 0
            Processed: 20830/28800, Skipped: 0
            Processed: 20840/28800, Skipped: 0
            Processed: 20850/28800, Skipped: 0
            Processed: 20860/28800, Skipped: 0
            Processed: 20870/28800, Skipped: 0
            Processed: 20880/28800, Skipped: 0
            Processed: 20890/28800, Skipped: 0
            Processed: 20900/28800, Skipped: 0
            Processed: 20910/28800, Skipped: 0
            Processed: 20920/28800, Skipped: 0
            Processed: 20930/28800, Skipped: 0
            Processed: 20940/28800, Skipped: 0
            Processed: 20950/28800, Skipped: 0
            Processed: 20960/28800, Skipped: 0
            Processed: 20970/28800, Skipped: 0
            Processed: 20980/28800, Skipped: 0
            Processed: 20990/28800, Skipped: 0
            Processed: 21000/28800, Skipped: 0
            Processed: 21010/28800, Skipped: 0
            Processed: 21020/28800, Skipped: 0
            Processed: 21030/28800, Skipped: 0
            Processed: 21040/28800, Skipped: 0
            Processed: 21050/28800, Skipped: 0
            Processed: 21060/28800, Skipped: 0
            Processed: 21070/28800, Skipped: 0
            Processed: 21080/28800, Skipped: 0
            Processed: 21090/28800, Skipped: 0
            Processed: 21100/28800, Skipped: 0
            Processed: 21110/28800, Skipped: 0
            Processed: 21120/28800, Skipped: 0
            Processed: 21130/28800, Skipped: 0
            Processed: 21140/28800, Skipped: 0
            Processed: 21150/28800, Skipped: 0
            Processed: 21160/28800, Skipped: 0
            Processed: 21170/28800, Skipped: 0
            Processed: 21180/28800, Skipped: 0
            Processed: 21190/28800, Skipped: 0
            Processed: 21200/28800, Skipped: 0
            Processed: 21210/28800, Skipped: 0
            Processed: 21220/28800, Skipped: 0
            Processed: 21230/28800, Skipped: 0
            Processed: 21240/28800, Skipped: 0
            Processed: 21250/28800, Skipped: 0
            Processed: 21260/28800, Skipped: 0
            Processed: 21270/28800, Skipped: 0
            Processed: 21280/28800, Skipped: 0
            Processed: 21290/28800, Skipped: 0
            Processed: 21300/28800, Skipped: 0
            Processed: 21310/28800, Skipped: 0
            Processed: 21320/28800, Skipped: 0
            Processed: 21330/28800, Skipped: 0
            Processed: 21340/28800, Skipped: 0
            Processed: 21350/28800, Skipped: 0
            Processed: 21360/28800, Skipped: 0
            Processed: 21370/28800, Skipped: 0
            Processed: 21380/28800, Skipped: 0
            Processed: 21390/28800, Skipped: 0
            Processed: 21400/28800, Skipped: 0
            Processed: 21410/28800, Skipped: 0
            Processed: 21420/28800, Skipped: 0
            Processed: 21430/28800, Skipped: 0
            Processed: 21440/28800, Skipped: 0
            Processed: 21450/28800, Skipped: 0
            Processed: 21460/28800, Skipped: 0
            Processed: 21470/28800, Skipped: 0
            Processed: 21480/28800, Skipped: 0
            Processed: 21490/28800, Skipped: 0
            Processed: 21500/28800, Skipped: 0
            Processed: 21510/28800, Skipped: 0
            Processed: 21520/28800, Skipped: 0
            Processed: 21530/28800, Skipped: 0
            Processed: 21540/28800, Skipped: 0
            Processed: 21550/28800, Skipped: 0
            Processed: 21560/28800, Skipped: 0
            Processed: 21570/28800, Skipped: 0
            Processed: 21580/28800, Skipped: 0
            Processed: 21590/28800, Skipped: 0
            Processed: 21600/28800, Skipped: 0
          Processing downscale factor 2...
            Processed: 21610/28800, Skipped: 0
            Processed: 21620/28800, Skipped: 0
            Processed: 21630/28800, Skipped: 0
            Processed: 21640/28800, Skipped: 0
            Processed: 21650/28800, Skipped: 0
            Processed: 21660/28800, Skipped: 0
            Processed: 21670/28800, Skipped: 0
            Processed: 21680/28800, Skipped: 0
            Processed: 21690/28800, Skipped: 0
            Processed: 21700/28800, Skipped: 0
            Processed: 21710/28800, Skipped: 0
            Processed: 21720/28800, Skipped: 0
            Processed: 21730/28800, Skipped: 0
            Processed: 21740/28800, Skipped: 0
            Processed: 21750/28800, Skipped: 0
            Processed: 21760/28800, Skipped: 0
            Processed: 21770/28800, Skipped: 0
            Processed: 21780/28800, Skipped: 0
            Processed: 21790/28800, Skipped: 0
            Processed: 21800/28800, Skipped: 0
            Processed: 21810/28800, Skipped: 0
            Processed: 21820/28800, Skipped: 0
            Processed: 21830/28800, Skipped: 0
            Processed: 21840/28800, Skipped: 0
            Processed: 21850/28800, Skipped: 0
            Processed: 21860/28800, Skipped: 0
            Processed: 21870/28800, Skipped: 0
            Processed: 21880/28800, Skipped: 0
            Processed: 21890/28800, Skipped: 0
            Processed: 21900/28800, Skipped: 0
            Processed: 21910/28800, Skipped: 0
            Processed: 21920/28800, Skipped: 0
            Processed: 21930/28800, Skipped: 0
            Processed: 21940/28800, Skipped: 0
            Processed: 21950/28800, Skipped: 0
            Processed: 21960/28800, Skipped: 0
            Processed: 21970/28800, Skipped: 0
            Processed: 21980/28800, Skipped: 0
            Processed: 21990/28800, Skipped: 0
            Processed: 22000/28800, Skipped: 0
            Processed: 22010/28800, Skipped: 0
            Processed: 22020/28800, Skipped: 0
            Processed: 22030/28800, Skipped: 0
            Processed: 22040/28800, Skipped: 0
            Processed: 22050/28800, Skipped: 0
            Processed: 22060/28800, Skipped: 0
            Processed: 22070/28800, Skipped: 0
            Processed: 22080/28800, Skipped: 0
            Processed: 22090/28800, Skipped: 0
            Processed: 22100/28800, Skipped: 0
            Processed: 22110/28800, Skipped: 0
            Processed: 22120/28800, Skipped: 0
            Processed: 22130/28800, Skipped: 0
            Processed: 22140/28800, Skipped: 0
            Processed: 22150/28800, Skipped: 0
            Processed: 22160/28800, Skipped: 0
            Processed: 22170/28800, Skipped: 0
            Processed: 22180/28800, Skipped: 0
            Processed: 22190/28800, Skipped: 0
            Processed: 22200/28800, Skipped: 0
            Processed: 22210/28800, Skipped: 0
            Processed: 22220/28800, Skipped: 0
            Processed: 22230/28800, Skipped: 0
            Processed: 22240/28800, Skipped: 0
            Processed: 22250/28800, Skipped: 0
            Processed: 22260/28800, Skipped: 0
            Processed: 22270/28800, Skipped: 0
            Processed: 22280/28800, Skipped: 0
            Processed: 22290/28800, Skipped: 0
            Processed: 22300/28800, Skipped: 0
            Processed: 22310/28800, Skipped: 0
            Processed: 22320/28800, Skipped: 0
            Processed: 22330/28800, Skipped: 0
            Processed: 22340/28800, Skipped: 0
            Processed: 22350/28800, Skipped: 0
            Processed: 22360/28800, Skipped: 0
            Processed: 22370/28800, Skipped: 0
            Processed: 22380/28800, Skipped: 0
            Processed: 22390/28800, Skipped: 0
            Processed: 22400/28800, Skipped: 0
        Processing 512 antennas...
          Processing downscale factor 1...
            Processed: 22410/28800, Skipped: 0
            Processed: 22420/28800, Skipped: 0
            Processed: 22430/28800, Skipped: 0
            Processed: 22440/28800, Skipped: 0
            Processed: 22450/28800, Skipped: 0
            Processed: 22460/28800, Skipped: 0
            Processed: 22470/28800, Skipped: 0
            Processed: 22480/28800, Skipped: 0
            Processed: 22490/28800, Skipped: 0
            Processed: 22500/28800, Skipped: 0
            Processed: 22510/28800, Skipped: 0
            Processed: 22520/28800, Skipped: 0
            Processed: 22530/28800, Skipped: 0
            Processed: 22540/28800, Skipped: 0
            Processed: 22550/28800, Skipped: 0
            Processed: 22560/28800, Skipped: 0
            Processed: 22570/28800, Skipped: 0
            Processed: 22580/28800, Skipped: 0
            Processed: 22590/28800, Skipped: 0
            Processed: 22600/28800, Skipped: 0
            Processed: 22610/28800, Skipped: 0
            Processed: 22620/28800, Skipped: 0
            Processed: 22630/28800, Skipped: 0
            Processed: 22640/28800, Skipped: 0
            Processed: 22650/28800, Skipped: 0
            Processed: 22660/28800, Skipped: 0
            Processed: 22670/28800, Skipped: 0
            Processed: 22680/28800, Skipped: 0
            Processed: 22690/28800, Skipped: 0
            Processed: 22700/28800, Skipped: 0
            Processed: 22710/28800, Skipped: 0
            Processed: 22720/28800, Skipped: 0
            Processed: 22730/28800, Skipped: 0
            Processed: 22740/28800, Skipped: 0
            Processed: 22750/28800, Skipped: 0
            Processed: 22760/28800, Skipped: 0
            Processed: 22770/28800, Skipped: 0
            Processed: 22780/28800, Skipped: 0
            Processed: 22790/28800, Skipped: 0
            Processed: 22800/28800, Skipped: 0
            Processed: 22810/28800, Skipped: 0
            Processed: 22820/28800, Skipped: 0
            Processed: 22830/28800, Skipped: 0
            Processed: 22840/28800, Skipped: 0
            Processed: 22850/28800, Skipped: 0
            Processed: 22860/28800, Skipped: 0
            Processed: 22870/28800, Skipped: 0
            Processed: 22880/28800, Skipped: 0
            Processed: 22890/28800, Skipped: 0
            Processed: 22900/28800, Skipped: 0
            Processed: 22910/28800, Skipped: 0
            Processed: 22920/28800, Skipped: 0
            Processed: 22930/28800, Skipped: 0
            Processed: 22940/28800, Skipped: 0
            Processed: 22950/28800, Skipped: 0
            Processed: 22960/28800, Skipped: 0
            Processed: 22970/28800, Skipped: 0
            Processed: 22980/28800, Skipped: 0
            Processed: 22990/28800, Skipped: 0
            Processed: 23000/28800, Skipped: 0
            Processed: 23010/28800, Skipped: 0
            Processed: 23020/28800, Skipped: 0
            Processed: 23030/28800, Skipped: 0
            Processed: 23040/28800, Skipped: 0
            Processed: 23050/28800, Skipped: 0
            Processed: 23060/28800, Skipped: 0
            Processed: 23070/28800, Skipped: 0
            Processed: 23080/28800, Skipped: 0
            Processed: 23090/28800, Skipped: 0
            Processed: 23100/28800, Skipped: 0
            Processed: 23110/28800, Skipped: 0
            Processed: 23120/28800, Skipped: 0
            Processed: 23130/28800, Skipped: 0
            Processed: 23140/28800, Skipped: 0
            Processed: 23150/28800, Skipped: 0
            Processed: 23160/28800, Skipped: 0
            Processed: 23170/28800, Skipped: 0
            Processed: 23180/28800, Skipped: 0
            Processed: 23190/28800, Skipped: 0
            Processed: 23200/28800, Skipped: 0
          Processing downscale factor 2...
            Processed: 23210/28800, Skipped: 0
            Processed: 23220/28800, Skipped: 0
            Processed: 23230/28800, Skipped: 0
            Processed: 23240/28800, Skipped: 0
            Processed: 23250/28800, Skipped: 0
            Processed: 23260/28800, Skipped: 0
            Processed: 23270/28800, Skipped: 0
            Processed: 23280/28800, Skipped: 0
            Processed: 23290/28800, Skipped: 0
            Processed: 23300/28800, Skipped: 0
            Processed: 23310/28800, Skipped: 0
            Processed: 23320/28800, Skipped: 0
            Processed: 23330/28800, Skipped: 0
            Processed: 23340/28800, Skipped: 0
            Processed: 23350/28800, Skipped: 0
            Processed: 23360/28800, Skipped: 0
            Processed: 23370/28800, Skipped: 0
            Processed: 23380/28800, Skipped: 0
            Processed: 23390/28800, Skipped: 0
            Processed: 23400/28800, Skipped: 0
            Processed: 23410/28800, Skipped: 0
            Processed: 23420/28800, Skipped: 0
            Processed: 23430/28800, Skipped: 0
            Processed: 23440/28800, Skipped: 0
            Processed: 23450/28800, Skipped: 0
            Processed: 23460/28800, Skipped: 0
            Processed: 23470/28800, Skipped: 0
            Processed: 23480/28800, Skipped: 0
            Processed: 23490/28800, Skipped: 0
            Processed: 23500/28800, Skipped: 0
            Processed: 23510/28800, Skipped: 0
            Processed: 23520/28800, Skipped: 0
            Processed: 23530/28800, Skipped: 0
            Processed: 23540/28800, Skipped: 0
            Processed: 23550/28800, Skipped: 0
            Processed: 23560/28800, Skipped: 0
            Processed: 23570/28800, Skipped: 0
            Processed: 23580/28800, Skipped: 0
            Processed: 23590/28800, Skipped: 0
            Processed: 23600/28800, Skipped: 0
            Processed: 23610/28800, Skipped: 0
            Processed: 23620/28800, Skipped: 0
            Processed: 23630/28800, Skipped: 0
            Processed: 23640/28800, Skipped: 0
            Processed: 23650/28800, Skipped: 0
            Processed: 23660/28800, Skipped: 0
            Processed: 23670/28800, Skipped: 0
            Processed: 23680/28800, Skipped: 0
            Processed: 23690/28800, Skipped: 0
            Processed: 23700/28800, Skipped: 0
            Processed: 23710/28800, Skipped: 0
            Processed: 23720/28800, Skipped: 0
            Processed: 23730/28800, Skipped: 0
            Processed: 23740/28800, Skipped: 0
            Processed: 23750/28800, Skipped: 0
            Processed: 23760/28800, Skipped: 0
            Processed: 23770/28800, Skipped: 0
            Processed: 23780/28800, Skipped: 0
            Processed: 23790/28800, Skipped: 0
            Processed: 23800/28800, Skipped: 0
            Processed: 23810/28800, Skipped: 0
            Processed: 23820/28800, Skipped: 0
            Processed: 23830/28800, Skipped: 0
            Processed: 23840/28800, Skipped: 0
            Processed: 23850/28800, Skipped: 0
            Processed: 23860/28800, Skipped: 0
            Processed: 23870/28800, Skipped: 0
            Processed: 23880/28800, Skipped: 0
            Processed: 23890/28800, Skipped: 0
            Processed: 23900/28800, Skipped: 0
            Processed: 23910/28800, Skipped: 0
            Processed: 23920/28800, Skipped: 0
            Processed: 23930/28800, Skipped: 0
            Processed: 23940/28800, Skipped: 0
            Processed: 23950/28800, Skipped: 0
            Processed: 23960/28800, Skipped: 0
            Processed: 23970/28800, Skipped: 0
            Processed: 23980/28800, Skipped: 0
            Processed: 23990/28800, Skipped: 0
            Processed: 24000/28800, Skipped: 0
        Processing 2048 antennas...
          Processing downscale factor 1...
            Processed: 24010/28800, Skipped: 0
            Processed: 24020/28800, Skipped: 0
            Processed: 24030/28800, Skipped: 0
            Processed: 24040/28800, Skipped: 0
            Processed: 24050/28800, Skipped: 0
            Processed: 24060/28800, Skipped: 0
            Processed: 24070/28800, Skipped: 0
            Processed: 24080/28800, Skipped: 0
            Processed: 24090/28800, Skipped: 0
            Processed: 24100/28800, Skipped: 0
            Processed: 24110/28800, Skipped: 0
            Processed: 24120/28800, Skipped: 0
            Processed: 24130/28800, Skipped: 0
            Processed: 24140/28800, Skipped: 0
            Processed: 24150/28800, Skipped: 0
            Processed: 24160/28800, Skipped: 0
            Processed: 24170/28800, Skipped: 0
            Processed: 24180/28800, Skipped: 0
            Processed: 24190/28800, Skipped: 0
            Processed: 24200/28800, Skipped: 0
            Processed: 24210/28800, Skipped: 0
            Processed: 24220/28800, Skipped: 0
            Processed: 24230/28800, Skipped: 0
            Processed: 24240/28800, Skipped: 0
            Processed: 24250/28800, Skipped: 0
            Processed: 24260/28800, Skipped: 0
            Processed: 24270/28800, Skipped: 0
            Processed: 24280/28800, Skipped: 0
            Processed: 24290/28800, Skipped: 0
            Processed: 24300/28800, Skipped: 0
            Processed: 24310/28800, Skipped: 0
            Processed: 24320/28800, Skipped: 0
            Processed: 24330/28800, Skipped: 0
            Processed: 24340/28800, Skipped: 0
            Processed: 24350/28800, Skipped: 0
            Processed: 24360/28800, Skipped: 0
            Processed: 24370/28800, Skipped: 0
            Processed: 24380/28800, Skipped: 0
            Processed: 24390/28800, Skipped: 0
            Processed: 24400/28800, Skipped: 0
            Processed: 24410/28800, Skipped: 0
            Processed: 24420/28800, Skipped: 0
            Processed: 24430/28800, Skipped: 0
            Processed: 24440/28800, Skipped: 0
            Processed: 24450/28800, Skipped: 0
            Processed: 24460/28800, Skipped: 0
            Processed: 24470/28800, Skipped: 0
            Processed: 24480/28800, Skipped: 0
            Processed: 24490/28800, Skipped: 0
            Processed: 24500/28800, Skipped: 0
            Processed: 24510/28800, Skipped: 0
            Processed: 24520/28800, Skipped: 0
            Processed: 24530/28800, Skipped: 0
            Processed: 24540/28800, Skipped: 0
            Processed: 24550/28800, Skipped: 0
            Processed: 24560/28800, Skipped: 0
            Processed: 24570/28800, Skipped: 0
            Processed: 24580/28800, Skipped: 0
            Processed: 24590/28800, Skipped: 0
            Processed: 24600/28800, Skipped: 0
            Processed: 24610/28800, Skipped: 0
            Processed: 24620/28800, Skipped: 0
            Processed: 24630/28800, Skipped: 0
            Processed: 24640/28800, Skipped: 0
            Processed: 24650/28800, Skipped: 0
            Processed: 24660/28800, Skipped: 0
            Processed: 24670/28800, Skipped: 0
            Processed: 24680/28800, Skipped: 0
            Processed: 24690/28800, Skipped: 0
            Processed: 24700/28800, Skipped: 0
            Processed: 24710/28800, Skipped: 0
            Processed: 24720/28800, Skipped: 0
            Processed: 24730/28800, Skipped: 0
            Processed: 24740/28800, Skipped: 0
            Processed: 24750/28800, Skipped: 0
            Processed: 24760/28800, Skipped: 0
            Processed: 24770/28800, Skipped: 0
            Processed: 24780/28800, Skipped: 0
            Processed: 24790/28800, Skipped: 0
            Processed: 24800/28800, Skipped: 0
          Processing downscale factor 2...
            Processed: 24810/28800, Skipped: 0
            Processed: 24820/28800, Skipped: 0
            Processed: 24830/28800, Skipped: 0
            Processed: 24840/28800, Skipped: 0
            Processed: 24850/28800, Skipped: 0
            Processed: 24860/28800, Skipped: 0
            Processed: 24870/28800, Skipped: 0
            Processed: 24880/28800, Skipped: 0
            Processed: 24890/28800, Skipped: 0
            Processed: 24900/28800, Skipped: 0
            Processed: 24910/28800, Skipped: 0
            Processed: 24920/28800, Skipped: 0
            Processed: 24930/28800, Skipped: 0
            Processed: 24940/28800, Skipped: 0
            Processed: 24950/28800, Skipped: 0
            Processed: 24960/28800, Skipped: 0
            Processed: 24970/28800, Skipped: 0
            Processed: 24980/28800, Skipped: 0
            Processed: 24990/28800, Skipped: 0
            Processed: 25000/28800, Skipped: 0
            Processed: 25010/28800, Skipped: 0
            Processed: 25020/28800, Skipped: 0
            Processed: 25030/28800, Skipped: 0
            Processed: 25040/28800, Skipped: 0
            Processed: 25050/28800, Skipped: 0
            Processed: 25060/28800, Skipped: 0
            Processed: 25070/28800, Skipped: 0
            Processed: 25080/28800, Skipped: 0
            Processed: 25090/28800, Skipped: 0
            Processed: 25100/28800, Skipped: 0
            Processed: 25110/28800, Skipped: 0
            Processed: 25120/28800, Skipped: 0
            Processed: 25130/28800, Skipped: 0
            Processed: 25140/28800, Skipped: 0
            Processed: 25150/28800, Skipped: 0
            Processed: 25160/28800, Skipped: 0
            Processed: 25170/28800, Skipped: 0
            Processed: 25180/28800, Skipped: 0
            Processed: 25190/28800, Skipped: 0
            Processed: 25200/28800, Skipped: 0
            Processed: 25210/28800, Skipped: 0
            Processed: 25220/28800, Skipped: 0
            Processed: 25230/28800, Skipped: 0
            Processed: 25240/28800, Skipped: 0
            Processed: 25250/28800, Skipped: 0
            Processed: 25260/28800, Skipped: 0
            Processed: 25270/28800, Skipped: 0
            Processed: 25280/28800, Skipped: 0
            Processed: 25290/28800, Skipped: 0
            Processed: 25300/28800, Skipped: 0
            Processed: 25310/28800, Skipped: 0
            Processed: 25320/28800, Skipped: 0
            Processed: 25330/28800, Skipped: 0
            Processed: 25340/28800, Skipped: 0
            Processed: 25350/28800, Skipped: 0
            Processed: 25360/28800, Skipped: 0
            Processed: 25370/28800, Skipped: 0
            Processed: 25380/28800, Skipped: 0
            Processed: 25390/28800, Skipped: 0
            Processed: 25400/28800, Skipped: 0
            Processed: 25410/28800, Skipped: 0
            Processed: 25420/28800, Skipped: 0
            Processed: 25430/28800, Skipped: 0
            Processed: 25440/28800, Skipped: 0
            Processed: 25450/28800, Skipped: 0
            Processed: 25460/28800, Skipped: 0
            Processed: 25470/28800, Skipped: 0
            Processed: 25480/28800, Skipped: 0
            Processed: 25490/28800, Skipped: 0
            Processed: 25500/28800, Skipped: 0
            Processed: 25510/28800, Skipped: 0
            Processed: 25520/28800, Skipped: 0
            Processed: 25530/28800, Skipped: 0
            Processed: 25540/28800, Skipped: 0
            Processed: 25550/28800, Skipped: 0
            Processed: 25560/28800, Skipped: 0
            Processed: 25570/28800, Skipped: 0
            Processed: 25580/28800, Skipped: 0
            Processed: 25590/28800, Skipped: 0
            Processed: 25600/28800, Skipped: 0
    Processing valid dataset...
      Processing frequency 2000MHz...
        Processing 32 antennas...
          Processing downscale factor 1...
            Processed: 25610/28800, Skipped: 0
            Processed: 25620/28800, Skipped: 0
            Processed: 25630/28800, Skipped: 0
            Processed: 25640/28800, Skipped: 0
            Processed: 25650/28800, Skipped: 0
            Processed: 25660/28800, Skipped: 0
            Processed: 25670/28800, Skipped: 0
            Processed: 25680/28800, Skipped: 0
            Processed: 25690/28800, Skipped: 0
            Processed: 25700/28800, Skipped: 0
          Processing downscale factor 2...
            Processed: 25710/28800, Skipped: 0
            Processed: 25720/28800, Skipped: 0
            Processed: 25730/28800, Skipped: 0
            Processed: 25740/28800, Skipped: 0
            Processed: 25750/28800, Skipped: 0
            Processed: 25760/28800, Skipped: 0
            Processed: 25770/28800, Skipped: 0
            Processed: 25780/28800, Skipped: 0
            Processed: 25790/28800, Skipped: 0
            Processed: 25800/28800, Skipped: 0
        Processing 128 antennas...
          Processing downscale factor 1...
            Processed: 25810/28800, Skipped: 0
            Processed: 25820/28800, Skipped: 0
            Processed: 25830/28800, Skipped: 0
            Processed: 25840/28800, Skipped: 0
            Processed: 25850/28800, Skipped: 0
            Processed: 25860/28800, Skipped: 0
            Processed: 25870/28800, Skipped: 0
            Processed: 25880/28800, Skipped: 0
            Processed: 25890/28800, Skipped: 0
            Processed: 25900/28800, Skipped: 0
          Processing downscale factor 2...
            Processed: 25910/28800, Skipped: 0
            Processed: 25920/28800, Skipped: 0
            Processed: 25930/28800, Skipped: 0
            Processed: 25940/28800, Skipped: 0
            Processed: 25950/28800, Skipped: 0
            Processed: 25960/28800, Skipped: 0
            Processed: 25970/28800, Skipped: 0
            Processed: 25980/28800, Skipped: 0
            Processed: 25990/28800, Skipped: 0
            Processed: 26000/28800, Skipped: 0
        Processing 512 antennas...
          Processing downscale factor 1...
            Processed: 26010/28800, Skipped: 0
            Processed: 26020/28800, Skipped: 0
            Processed: 26030/28800, Skipped: 0
            Processed: 26040/28800, Skipped: 0
            Processed: 26050/28800, Skipped: 0
            Processed: 26060/28800, Skipped: 0
            Processed: 26070/28800, Skipped: 0
            Processed: 26080/28800, Skipped: 0
            Processed: 26090/28800, Skipped: 0
            Processed: 26100/28800, Skipped: 0
          Processing downscale factor 2...
            Processed: 26110/28800, Skipped: 0
            Processed: 26120/28800, Skipped: 0
            Processed: 26130/28800, Skipped: 0
            Processed: 26140/28800, Skipped: 0
            Processed: 26150/28800, Skipped: 0
            Processed: 26160/28800, Skipped: 0
            Processed: 26170/28800, Skipped: 0
            Processed: 26180/28800, Skipped: 0
            Processed: 26190/28800, Skipped: 0
            Processed: 26200/28800, Skipped: 0
        Processing 2048 antennas...
          Processing downscale factor 1...
            Processed: 26210/28800, Skipped: 0
            Processed: 26220/28800, Skipped: 0
            Processed: 26230/28800, Skipped: 0
            Processed: 26240/28800, Skipped: 0
            Processed: 26250/28800, Skipped: 0
            Processed: 26260/28800, Skipped: 0
            Processed: 26270/28800, Skipped: 0
            Processed: 26280/28800, Skipped: 0
            Processed: 26290/28800, Skipped: 0
            Processed: 26300/28800, Skipped: 0
          Processing downscale factor 2...
            Processed: 26310/28800, Skipped: 0
            Processed: 26320/28800, Skipped: 0
            Processed: 26330/28800, Skipped: 0
            Processed: 26340/28800, Skipped: 0
            Processed: 26350/28800, Skipped: 0
            Processed: 26360/28800, Skipped: 0
            Processed: 26370/28800, Skipped: 0
            Processed: 26380/28800, Skipped: 0
            Processed: 26390/28800, Skipped: 0
            Processed: 26400/28800, Skipped: 0
      Processing frequency 1300MHz...
        Processing 32 antennas...
          Processing downscale factor 1...
            Processed: 26410/28800, Skipped: 0
            Processed: 26420/28800, Skipped: 0
            Processed: 26430/28800, Skipped: 0
            Processed: 26440/28800, Skipped: 0
            Processed: 26450/28800, Skipped: 0
            Processed: 26460/28800, Skipped: 0
            Processed: 26470/28800, Skipped: 0
            Processed: 26480/28800, Skipped: 0
            Processed: 26490/28800, Skipped: 0
            Processed: 26500/28800, Skipped: 0
          Processing downscale factor 2...
            Processed: 26510/28800, Skipped: 0
            Processed: 26520/28800, Skipped: 0
            Processed: 26530/28800, Skipped: 0
            Processed: 26540/28800, Skipped: 0
            Processed: 26550/28800, Skipped: 0
            Processed: 26560/28800, Skipped: 0
            Processed: 26570/28800, Skipped: 0
            Processed: 26580/28800, Skipped: 0
            Processed: 26590/28800, Skipped: 0
            Processed: 26600/28800, Skipped: 0
        Processing 128 antennas...
          Processing downscale factor 1...
            Processed: 26610/28800, Skipped: 0
            Processed: 26620/28800, Skipped: 0
            Processed: 26630/28800, Skipped: 0
            Processed: 26640/28800, Skipped: 0
            Processed: 26650/28800, Skipped: 0
            Processed: 26660/28800, Skipped: 0
            Processed: 26670/28800, Skipped: 0
            Processed: 26680/28800, Skipped: 0
            Processed: 26690/28800, Skipped: 0
            Processed: 26700/28800, Skipped: 0
          Processing downscale factor 2...
            Processed: 26710/28800, Skipped: 0
            Processed: 26720/28800, Skipped: 0
            Processed: 26730/28800, Skipped: 0
            Processed: 26740/28800, Skipped: 0
            Processed: 26750/28800, Skipped: 0
            Processed: 26760/28800, Skipped: 0
            Processed: 26770/28800, Skipped: 0
            Processed: 26780/28800, Skipped: 0
            Processed: 26790/28800, Skipped: 0
            Processed: 26800/28800, Skipped: 0
        Processing 512 antennas...
          Processing downscale factor 1...
            Processed: 26810/28800, Skipped: 0
            Processed: 26820/28800, Skipped: 0
            Processed: 26830/28800, Skipped: 0
            Processed: 26840/28800, Skipped: 0
            Processed: 26850/28800, Skipped: 0
            Processed: 26860/28800, Skipped: 0
            Processed: 26870/28800, Skipped: 0
            Processed: 26880/28800, Skipped: 0
            Processed: 26890/28800, Skipped: 0
            Processed: 26900/28800, Skipped: 0
          Processing downscale factor 2...
            Processed: 26910/28800, Skipped: 0
            Processed: 26920/28800, Skipped: 0
            Processed: 26930/28800, Skipped: 0
            Processed: 26940/28800, Skipped: 0
            Processed: 26950/28800, Skipped: 0
            Processed: 26960/28800, Skipped: 0
            Processed: 26970/28800, Skipped: 0
            Processed: 26980/28800, Skipped: 0
            Processed: 26990/28800, Skipped: 0
            Processed: 27000/28800, Skipped: 0
        Processing 2048 antennas...
          Processing downscale factor 1...
            Processed: 27010/28800, Skipped: 0
            Processed: 27020/28800, Skipped: 0
            Processed: 27030/28800, Skipped: 0
            Processed: 27040/28800, Skipped: 0
            Processed: 27050/28800, Skipped: 0
            Processed: 27060/28800, Skipped: 0
            Processed: 27070/28800, Skipped: 0
            Processed: 27080/28800, Skipped: 0
            Processed: 27090/28800, Skipped: 0
            Processed: 27100/28800, Skipped: 0
          Processing downscale factor 2...
            Processed: 27110/28800, Skipped: 0
            Processed: 27120/28800, Skipped: 0
            Processed: 27130/28800, Skipped: 0
            Processed: 27140/28800, Skipped: 0
            Processed: 27150/28800, Skipped: 0
            Processed: 27160/28800, Skipped: 0
            Processed: 27170/28800, Skipped: 0
            Processed: 27180/28800, Skipped: 0
            Processed: 27190/28800, Skipped: 0
            Processed: 27200/28800, Skipped: 0
      Processing frequency 700MHz...
        Processing 32 antennas...
          Processing downscale factor 1...
            Processed: 27210/28800, Skipped: 0
            Processed: 27220/28800, Skipped: 0
            Processed: 27230/28800, Skipped: 0
            Processed: 27240/28800, Skipped: 0
            Processed: 27250/28800, Skipped: 0
            Processed: 27260/28800, Skipped: 0
            Processed: 27270/28800, Skipped: 0
            Processed: 27280/28800, Skipped: 0
            Processed: 27290/28800, Skipped: 0
            Processed: 27300/28800, Skipped: 0
          Processing downscale factor 2...
            Processed: 27310/28800, Skipped: 0
            Processed: 27320/28800, Skipped: 0
            Processed: 27330/28800, Skipped: 0
            Processed: 27340/28800, Skipped: 0
            Processed: 27350/28800, Skipped: 0
            Processed: 27360/28800, Skipped: 0
            Processed: 27370/28800, Skipped: 0
            Processed: 27380/28800, Skipped: 0
            Processed: 27390/28800, Skipped: 0
            Processed: 27400/28800, Skipped: 0
        Processing 128 antennas...
          Processing downscale factor 1...
            Processed: 27410/28800, Skipped: 0
            Processed: 27420/28800, Skipped: 0
            Processed: 27430/28800, Skipped: 0
            Processed: 27440/28800, Skipped: 0
            Processed: 27450/28800, Skipped: 0
            Processed: 27460/28800, Skipped: 0
            Processed: 27470/28800, Skipped: 0
            Processed: 27480/28800, Skipped: 0
            Processed: 27490/28800, Skipped: 0
            Processed: 27500/28800, Skipped: 0
          Processing downscale factor 2...
            Processed: 27510/28800, Skipped: 0
            Processed: 27520/28800, Skipped: 0
            Processed: 27530/28800, Skipped: 0
            Processed: 27540/28800, Skipped: 0
            Processed: 27550/28800, Skipped: 0
            Processed: 27560/28800, Skipped: 0
            Processed: 27570/28800, Skipped: 0
            Processed: 27580/28800, Skipped: 0
            Processed: 27590/28800, Skipped: 0
            Processed: 27600/28800, Skipped: 0
        Processing 512 antennas...
          Processing downscale factor 1...
            Processed: 27610/28800, Skipped: 0
            Processed: 27620/28800, Skipped: 0
            Processed: 27630/28800, Skipped: 0
            Processed: 27640/28800, Skipped: 0
            Processed: 27650/28800, Skipped: 0
            Processed: 27660/28800, Skipped: 0
            Processed: 27670/28800, Skipped: 0
            Processed: 27680/28800, Skipped: 0
            Processed: 27690/28800, Skipped: 0
            Processed: 27700/28800, Skipped: 0
          Processing downscale factor 2...
            Processed: 27710/28800, Skipped: 0
            Processed: 27720/28800, Skipped: 0
            Processed: 27730/28800, Skipped: 0
            Processed: 27740/28800, Skipped: 0
            Processed: 27750/28800, Skipped: 0
            Processed: 27760/28800, Skipped: 0
            Processed: 27770/28800, Skipped: 0
            Processed: 27780/28800, Skipped: 0
            Processed: 27790/28800, Skipped: 0
            Processed: 27800/28800, Skipped: 0
        Processing 2048 antennas...
          Processing downscale factor 1...
            Processed: 27810/28800, Skipped: 0
            Processed: 27820/28800, Skipped: 0
            Processed: 27830/28800, Skipped: 0
            Processed: 27840/28800, Skipped: 0
            Processed: 27850/28800, Skipped: 0
            Processed: 27860/28800, Skipped: 0
            Processed: 27870/28800, Skipped: 0
            Processed: 27880/28800, Skipped: 0
            Processed: 27890/28800, Skipped: 0
            Processed: 27900/28800, Skipped: 0
          Processing downscale factor 2...
            Processed: 27910/28800, Skipped: 0
            Processed: 27920/28800, Skipped: 0
            Processed: 27930/28800, Skipped: 0
            Processed: 27940/28800, Skipped: 0
            Processed: 27950/28800, Skipped: 0
            Processed: 27960/28800, Skipped: 0
            Processed: 27970/28800, Skipped: 0
            Processed: 27980/28800, Skipped: 0
            Processed: 27990/28800, Skipped: 0
            Processed: 28000/28800, Skipped: 0
      Processing frequency 10MHz...
        Processing 32 antennas...
          Processing downscale factor 1...
            Processed: 28010/28800, Skipped: 0
            Processed: 28020/28800, Skipped: 0
            Processed: 28030/28800, Skipped: 0
            Processed: 28040/28800, Skipped: 0
            Processed: 28050/28800, Skipped: 0
            Processed: 28060/28800, Skipped: 0
            Processed: 28070/28800, Skipped: 0
            Processed: 28080/28800, Skipped: 0
            Processed: 28090/28800, Skipped: 0
            Processed: 28100/28800, Skipped: 0
          Processing downscale factor 2...
            Processed: 28110/28800, Skipped: 0
            Processed: 28120/28800, Skipped: 0
            Processed: 28130/28800, Skipped: 0
            Processed: 28140/28800, Skipped: 0
            Processed: 28150/28800, Skipped: 0
            Processed: 28160/28800, Skipped: 0
            Processed: 28170/28800, Skipped: 0
            Processed: 28180/28800, Skipped: 0
            Processed: 28190/28800, Skipped: 0
            Processed: 28200/28800, Skipped: 0
        Processing 128 antennas...
          Processing downscale factor 1...
            Processed: 28210/28800, Skipped: 0
            Processed: 28220/28800, Skipped: 0
            Processed: 28230/28800, Skipped: 0
            Processed: 28240/28800, Skipped: 0
            Processed: 28250/28800, Skipped: 0
            Processed: 28260/28800, Skipped: 0
            Processed: 28270/28800, Skipped: 0
            Processed: 28280/28800, Skipped: 0
            Processed: 28290/28800, Skipped: 0
            Processed: 28300/28800, Skipped: 0
          Processing downscale factor 2...
            Processed: 28310/28800, Skipped: 0
            Processed: 28320/28800, Skipped: 0
            Processed: 28330/28800, Skipped: 0
            Processed: 28340/28800, Skipped: 0
            Processed: 28350/28800, Skipped: 0
            Processed: 28360/28800, Skipped: 0
            Processed: 28370/28800, Skipped: 0
            Processed: 28380/28800, Skipped: 0
            Processed: 28390/28800, Skipped: 0
            Processed: 28400/28800, Skipped: 0
        Processing 512 antennas...
          Processing downscale factor 1...
            Processed: 28410/28800, Skipped: 0
            Processed: 28420/28800, Skipped: 0
            Processed: 28430/28800, Skipped: 0
            Processed: 28440/28800, Skipped: 0
            Processed: 28450/28800, Skipped: 0
            Processed: 28460/28800, Skipped: 0
            Processed: 28470/28800, Skipped: 0
            Processed: 28480/28800, Skipped: 0
            Processed: 28490/28800, Skipped: 0
            Processed: 28500/28800, Skipped: 0
          Processing downscale factor 2...
            Processed: 28510/28800, Skipped: 0
            Processed: 28520/28800, Skipped: 0
            Processed: 28530/28800, Skipped: 0
            Processed: 28540/28800, Skipped: 0
            Processed: 28550/28800, Skipped: 0
            Processed: 28560/28800, Skipped: 0
            Processed: 28570/28800, Skipped: 0
            Processed: 28580/28800, Skipped: 0
            Processed: 28590/28800, Skipped: 0
            Processed: 28600/28800, Skipped: 0
        Processing 2048 antennas...
          Processing downscale factor 1...
            Processed: 28610/28800, Skipped: 0
            Processed: 28620/28800, Skipped: 0
            Processed: 28630/28800, Skipped: 0
            Processed: 28640/28800, Skipped: 0
            Processed: 28650/28800, Skipped: 0
            Processed: 28660/28800, Skipped: 0
            Processed: 28670/28800, Skipped: 0
            Processed: 28680/28800, Skipped: 0
            Processed: 28690/28800, Skipped: 0
            Processed: 28700/28800, Skipped: 0
          Processing downscale factor 2...
            Processed: 28710/28800, Skipped: 0
            Processed: 28720/28800, Skipped: 0
            Processed: 28730/28800, Skipped: 0
            Processed: 28740/28800, Skipped: 0
            Processed: 28750/28800, Skipped: 0
            Processed: 28760/28800, Skipped: 0
            Processed: 28770/28800, Skipped: 0
            Processed: 28780/28800, Skipped: 0
            Processed: 28790/28800, Skipped: 0
            Processed: 28800/28800, Skipped: 0
    Dataset generation completed.



```python
# SANITY CHECK
# File paths
file_paths = {
    "dirty_image_full": "./synthetic_data/train/freq_2000MHz/antennas_32_s1/0000_dirty_image_full.npy",
    "fft_full": "./synthetic_data/train/freq_2000MHz/antennas_32_s1/0000_fft_full.npy",
    "masked_fft_full": "./synthetic_data/train/freq_2000MHz/antennas_32_s1/0000_masked_fft_full.npy",
    "noisy_dirty_image_full": "./synthetic_data/train/freq_2000MHz/antennas_32_s1/0000_noisy_dirty_image_full.npy",
    "noisy_fft_full": "./synthetic_data/train/freq_2000MHz/antennas_32_s1/0000_noisy_fft_full.npy",
    "noisy_masked_fft_full": "./synthetic_data/train/freq_2000MHz/antennas_32_s1/0000_noisy_masked_fft_full.npy",
}

# Load data
data = {key: np.load(path) for key, path in file_paths.items()}

# Plot data
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Debugging Visualizations for 0000 Files", fontsize=16)

titles = [
    "Dirty Image Full",
    "FFT Full (Magnitude)",
    "Masked FFT Full (Magnitude)",
    "Noisy Dirty Image Full",
    "Noisy FFT Full (Magnitude)",
    "Noisy Masked FFT Full (Magnitude)",
]

for ax, (key, title) in zip(axes.flat, zip(data.keys(), titles)):
    if "fft" in key:
        # For FFT files, plot the magnitude
        ax.imshow(np.log1p(np.abs(data[key])), cmap="gray", origin="lower")
    else:
        # For image files, plot the actual values
        ax.imshow(data[key], cmap="gray", origin="lower")
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the suptitle
plt.show()
```


    
![png](output_32_0.png)
    


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


```python

```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    Cell In[30], line 1
    ----> 1 assert REGENERATE_DATA, "Not in data generation mode. Stopping this cell from wasting your time..."


    AssertionError: Not in data generation mode. Stopping this cell from wasting your time...

