"""
LOLv1 Dataset with ReDDiT Reflectance Estimation for Low-Light Image Enhancement.

This module extends the standard LOLv1 dataset with Reflectance-Aware Trajectory Refinement (RATR)
from the ReDDiT paper (CVPR 2025): "Efficient Diffusion as Low Light Enhancer"

The reflectance estimation follows the Retinex theory:
    x = (y - z) / h
where:
    - y: low-light input image
    - h: illumination map (estimated as max channel of y)
    - z: noise map (estimated as |y - ψ(y)| where ψ is a denoising operation)
    - x: estimated clean reflectance

Reference:
    Lan et al., "Efficient Diffusion as Low Light Enhancer", CVPR 2025
    https://github.com/lgz-0713/ReDDiT
"""

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
import torch
import torch.nn.functional as F
import os
import glob
from PIL import Image
import cv2
import numpy as np


def pad_to_multiple(tensor, multiple=16, mode='reflect'):
    """
    Pad a tensor (C, H, W) so that H and W are divisible by `multiple`.
    Returns: (padded_tensor, original_h, original_w)
    """
    _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor.unsqueeze(0), (0, pad_w, 0, pad_h), mode=mode).squeeze(0)
    return tensor, h, w


class RetinexEstimator:
    """
    Reflectance-Aware Trajectory Refinement (RATR) module from ReDDiT.

    Implements reflectance estimation based on Retinex theory:
    - Illumination estimation: h' = max_channel(y)
    - Noise estimation: z' = |y - ψ(y)|, where ψ is a non-learning denoising operation
    - Reflectance: x̃₀ = (y - z') / h'

    This estimated reflectance can be used for trajectory refinement in diffusion models.
    """

    def __init__(self,
                 denoise_method='bilateral',
                 bilateral_d=9,
                 bilateral_sigma_color=75,
                 bilateral_sigma_space=75,
                 gaussian_kernel_size=5,
                 gaussian_sigma=1.0,
                 median_kernel_size=5,
                 epsilon=1e-6):
        """
        Args:
            denoise_method: Denoising method for noise estimation.
                Options: 'bilateral', 'gaussian', 'median', 'nlm' (non-local means)
            bilateral_d: Diameter of each pixel neighborhood for bilateral filter
            bilateral_sigma_color: Filter sigma in the color space for bilateral
            bilateral_sigma_space: Filter sigma in the coordinate space for bilateral
            gaussian_kernel_size: Kernel size for Gaussian blur
            gaussian_sigma: Standard deviation for Gaussian kernel
            median_kernel_size: Kernel size for median filter
            epsilon: Small constant to prevent division by zero
        """
        self.denoise_method = denoise_method
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.gaussian_kernel_size = gaussian_kernel_size
        self.gaussian_sigma = gaussian_sigma
        self.median_kernel_size = median_kernel_size
        self.epsilon = epsilon

    def estimate_illumination(self, image):
        """
        Estimate illumination map using maximum channel.

        Following common practice in Retinex-based methods, the illumination
        is estimated as the maximum value across RGB channels for each pixel.

        Args:
            image: Input tensor of shape (C, H, W) or (B, C, H, W) in range [0, 1]

        Returns:
            illumination: Tensor of shape (1, H, W) or (B, 1, H, W)
        """
        if image.dim() == 3:
            # (C, H, W) -> (1, H, W)
            illumination = image.max(dim=0, keepdim=True)[0]
        else:
            # (B, C, H, W) -> (B, 1, H, W)
            illumination = image.max(dim=1, keepdim=True)[0]

        # Clamp to prevent division by zero
        illumination = torch.clamp(illumination, min=self.epsilon)
        return illumination

    def _apply_denoising_numpy(self, image_np):
        """
        Apply non-learning based denoising operation (ψ) to estimate noise.

        Args:
            image_np: NumPy array of shape (H, W, C) in range [0, 255], uint8

        Returns:
            denoised: NumPy array of same shape
        """
        if self.denoise_method == 'bilateral':
            # Bilateral filter preserves edges while smoothing
            denoised = cv2.bilateralFilter(
                image_np,
                self.bilateral_d,
                self.bilateral_sigma_color,
                self.bilateral_sigma_space
            )
        elif self.denoise_method == 'gaussian':
            # Gaussian blur for simple smoothing
            denoised = cv2.GaussianBlur(
                image_np,
                (self.gaussian_kernel_size, self.gaussian_kernel_size),
                self.gaussian_sigma
            )
        elif self.denoise_method == 'median':
            # Median filter for salt-and-pepper noise
            denoised = cv2.medianBlur(image_np, self.median_kernel_size)
        elif self.denoise_method == 'nlm':
            # Non-local means denoising (slower but higher quality)
            denoised = cv2.fastNlMeansDenoisingColored(
                image_np, None, 10, 10, 7, 21
            )
        else:
            raise ValueError(f"Unknown denoise method: {self.denoise_method}")

        return denoised

    def estimate_noise(self, image):
        """
        Estimate noise map as z' = |y - ψ(y)|.

        The noise is modeled as the absolute difference between the input
        image and its denoised version.

        Args:
            image: Input tensor of shape (C, H, W) or (B, C, H, W) in range [0, 1]

        Returns:
            noise: Tensor of same shape as input
        """
        is_batched = image.dim() == 4
        if not is_batched:
            image = image.unsqueeze(0)

        batch_size = image.shape[0]
        noise_maps = []

        for i in range(batch_size):
            # Convert to numpy for OpenCV denoising
            img = image[i]  # (C, H, W)
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            # Apply denoising
            denoised_np = self._apply_denoising_numpy(img_np)

            # Convert back to tensor
            denoised = torch.from_numpy(denoised_np).float() / 255.0
            denoised = denoised.permute(2, 0, 1).to(image.device)

            # Compute noise as absolute difference
            noise = torch.abs(img - denoised)
            noise_maps.append(noise)

        noise_batch = torch.stack(noise_maps, dim=0)

        if not is_batched:
            noise_batch = noise_batch.squeeze(0)

        return noise_batch

    def estimate_reflectance(self, image):
        """
        Estimate clean reflectance based on Retinex theory.

        x̃₀ = (y - z') / h'

        This latent clean image estimate can be used for trajectory refinement
        in diffusion-based low-light enhancement.

        Args:
            image: Low-light input tensor of shape (C, H, W) or (B, C, H, W) in range [0, 1]

        Returns:
            reflectance: Estimated clean reflectance tensor of same shape
            illumination: Illumination map tensor
            noise: Noise map tensor
        """
        # Estimate illumination (max channel)
        illumination = self.estimate_illumination(image)

        # Estimate noise
        noise = self.estimate_noise(image)

        # Compute reflectance: x = (y - z) / h
        # Broadcast illumination to match image shape
        if image.dim() == 3:
            # (C, H, W) case
            reflectance = (image - noise) / illumination
        else:
            # (B, C, H, W) case
            reflectance = (image - noise) / illumination

        # Clamp to valid range [0, 1]
        reflectance = torch.clamp(reflectance, 0, 1)

        return reflectance, illumination, noise

    def __call__(self, image):
        """Convenience method to estimate reflectance."""
        return self.estimate_reflectance(image)


class LOLv1ReDDiTDataset(Dataset):
    """
    LOLv1 Dataset with ReDDiT Reflectance Estimation.

    Extends the standard LOLv1 dataset with Retinex-based reflectance estimation
    for use with diffusion models implementing trajectory refinement.

    Supports:
    - Full images or random crops
    - Data augmentation (flips, rotations)
    - Reflection padding for validation
    - Retinex-based reflectance, illumination, and noise map estimation
    """

    def __init__(self,
                 image_folder,
                 patch_size=256,
                 full_image=False,
                 augment=True,
                 is_val=False,
                 compute_reflectance=True,
                 denoise_method='bilateral',
                 **retinex_kwargs):
        """
        Args:
            image_folder: Path to dataset folder containing low/ and high/ subfolders
            patch_size: Size of random crops (ignored if full_image=True)
            full_image: If True, return full images; if False, return random crops
            augment: If True, apply random flips/rotations (only for crops)
            is_val: If True, use reflection padding instead of resize for full images
            compute_reflectance: If True, compute Retinex-based reflectance estimation
            denoise_method: Denoising method for noise estimation
                Options: 'bilateral', 'gaussian', 'median', 'nlm'
            **retinex_kwargs: Additional arguments passed to RetinexEstimator
        """
        super().__init__()
        self.image_folder = image_folder
        self.patch_size = patch_size
        self.full_image = full_image
        self.augment = augment
        self.is_val = is_val
        self.compute_reflectance = compute_reflectance

        # Initialize Retinex estimator
        if compute_reflectance:
            self.retinex = RetinexEstimator(
                denoise_method=denoise_method,
                **retinex_kwargs
            )

        # Get image paths
        lq_dir = os.path.join(image_folder, "low")
        hq_dir = os.path.join(image_folder, "high")

        if not os.path.exists(lq_dir):
            raise FileNotFoundError(f"Low-light folder not found: {lq_dir}")
        if not os.path.exists(hq_dir):
            raise FileNotFoundError(f"High-light folder not found: {hq_dir}")

        # Get all image files (png format in LOLv1)
        self.lq_paths = sorted(glob.glob(os.path.join(lq_dir, "*.png")))
        self.hq_paths = sorted(glob.glob(os.path.join(hq_dir, "*.png")))

        # Verify matching pairs
        if len(self.lq_paths) != len(self.hq_paths):
            raise ValueError(
                f"Mismatch: {len(self.lq_paths)} low-light images vs "
                f"{len(self.hq_paths)} high-light images"
            )

        # Verify filenames match
        for lq_path, hq_path in zip(self.lq_paths, self.hq_paths):
            lq_name = os.path.basename(lq_path)
            hq_name = os.path.basename(hq_path)
            if lq_name != hq_name:
                raise ValueError(f"Filename mismatch: {lq_name} vs {hq_name}")

        # Transform: just convert to tensor
        self.transform = v2.Compose([v2.ToTensor()])

        mode_str = 'full_image' if full_image else f'crop_{patch_size}'
        if is_val:
            mode_str += '_val_padded'
        reflex_str = f'reflectance_{denoise_method}' if compute_reflectance else 'no_reflectance'
        print(f"[LOLv1ReDDiT] Loaded {len(self)} pairs | mode={mode_str} | {reflex_str}")

    def __len__(self):
        return len(self.lq_paths)

    def __getitem__(self, index):
        # Load images
        lq = Image.open(self.lq_paths[index]).convert("RGB")
        hq = Image.open(self.hq_paths[index]).convert("RGB")

        # Convert to tensor (C, H, W)
        lq = self.transform(lq)
        hq = self.transform(hq)

        if self.is_val:
            # Validation mode: pad to multiple of 32 with reflection
            orig_h, orig_w = lq.shape[1], lq.shape[2]
            lq, _, _ = pad_to_multiple(lq, multiple=32, mode='reflect')
            hq, _, _ = pad_to_multiple(hq, multiple=32, mode='reflect')

            if self.compute_reflectance:
                # Compute reflectance estimation
                reflectance, illumination, noise = self.retinex(lq)
                return {
                    'lq': lq,
                    'hq': hq,
                    'reflectance': reflectance,
                    'illumination': illumination,
                    'noise': noise,
                    'orig_size': torch.tensor([orig_h, orig_w])
                }
            else:
                return lq, hq, torch.tensor([orig_h, orig_w])

        elif self.augment:
            # Training mode with augmentation: random crop
            _, H, W = lq.shape
            P = self.patch_size

            if H >= P and W >= P:
                y = torch.randint(0, H - P + 1, (1,)).item()
                x = torch.randint(0, W - P + 1, (1,)).item()
                lq = lq[:, y:y + P, x:x + P]
                hq = hq[:, y:y + P, x:x + P]
            else:
                raise ValueError(
                    f"Image size ({H}x{W}) is smaller than patch size ({P}x{P})"
                )

            # Augmentation using torch operations
            if torch.rand(1).item() < 0.5:
                lq = torch.flip(lq, dims=[2])  # Horizontal flip
                hq = torch.flip(hq, dims=[2])
            if torch.rand(1).item() < 0.5:
                lq = torch.flip(lq, dims=[1])  # Vertical flip
                hq = torch.flip(hq, dims=[1])
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                lq = torch.rot90(lq, k, dims=[1, 2])
                hq = torch.rot90(hq, k, dims=[1, 2])

        if self.compute_reflectance:
            # Compute reflectance estimation
            reflectance, illumination, noise = self.retinex(lq)
            return {
                'lq': lq,
                'hq': hq,
                'reflectance': reflectance,
                'illumination': illumination.squeeze(0),  # Remove channel dim for consistency
                'noise': noise
            }
        else:
            return lq, hq


class LOLv1ReDDiT:
    """Loader factory for LOLv1 dataset with ReDDiT reflectance estimation."""

    def __init__(self):
        pass

    def create_loaders(self, dataset_params):
        """
        Create DataLoader with ReDDiT reflectance estimation.

        Args:
            dataset_params: Dictionary containing:
                - path: Path to dataset folder
                - batch_size: Batch size (default: 32)
                - num_workers: Number of data loading workers (default: 4)
                - patch_size: Size of random crops (default: 256)
                - shuffle: Whether to shuffle data (default: True)
                - full_image: Return full images instead of crops (default: False)
                - augment: Apply data augmentation (default: True)
                - is_val: Validation mode with padding (default: False)
                - compute_reflectance: Compute Retinex estimation (default: True)
                - denoise_method: Denoising method (default: 'bilateral')

        Returns:
            DataLoader instance
        """
        path = dataset_params.get("path")
        batch_size = dataset_params.get("batch_size", 32)
        num_workers = dataset_params.get("num_workers", 4)
        patch_size = dataset_params.get("patch_size", 256)
        shuffle = dataset_params.get("shuffle", True)
        full_image = dataset_params.get("full_image", False)
        augment = dataset_params.get("augment", True)
        is_val = dataset_params.get("is_val", False)
        compute_reflectance = dataset_params.get("compute_reflectance", True)
        denoise_method = dataset_params.get("denoise_method", 'bilateral')

        # Extract additional retinex kwargs
        retinex_kwargs = {}
        for key in ['bilateral_d', 'bilateral_sigma_color', 'bilateral_sigma_space',
                    'gaussian_kernel_size', 'gaussian_sigma', 'median_kernel_size', 'epsilon']:
            if key in dataset_params:
                retinex_kwargs[key] = dataset_params[key]

        dataset = LOLv1ReDDiTDataset(
            image_folder=path,
            patch_size=patch_size,
            full_image=full_image,
            augment=augment,
            is_val=is_val,
            compute_reflectance=compute_reflectance,
            denoise_method=denoise_method,
            **retinex_kwargs
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=not is_val
        )

        return loader


# ============================================================================
# Utility functions for trajectory refinement (as described in ReDDiT paper)
# ============================================================================

def compute_refined_trajectory_start(lq_batch, retinex_estimator=None):
    """
    Compute refined trajectory starting point using reflectance estimation.

    In ReDDiT, the estimated reflectance x̃₀ is used to shift the diffusion
    trajectory from pure Gaussian noise to a reflectance-aware residual space,
    reducing the inference gap.

    Args:
        lq_batch: Low-light image batch of shape (B, C, H, W)
        retinex_estimator: Optional RetinexEstimator instance

    Returns:
        x_tilde_0: Estimated clean image for trajectory refinement
        components: Dictionary with illumination and noise maps
    """
    if retinex_estimator is None:
        retinex_estimator = RetinexEstimator()

    reflectance, illumination, noise = retinex_estimator(lq_batch)

    return reflectance, {
        'illumination': illumination,
        'noise': noise
    }


def apply_trajectory_shift(x_t, x_tilde_0, alpha_t, sigma_t):
    """
    Apply trajectory shift for reflectance-aware diffusion.

    Instead of pure Gaussian noise, the shifted trajectory incorporates
    the estimated reflectance:
        x̃_s = α_s * x̃₀ + σ_s * ε_η

    This reduces the search space and improves distillation efficiency.

    Args:
        x_t: Current noisy sample at timestep t
        x_tilde_0: Estimated clean reflectance
        alpha_t: Noise schedule alpha at time t
        sigma_t: Noise schedule sigma at time t

    Returns:
        x_shifted: Shifted sample in reflectance-aware space
    """
    # The residual between current sample and scaled reflectance estimate
    residual = x_t - alpha_t * x_tilde_0

    # Normalize by sigma to get the shifted noise component
    if sigma_t > 0:
        x_shifted = residual / sigma_t
    else:
        x_shifted = residual

    return x_shifted


# ============================================================================
# Example usage and testing
# ============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("LOLv1 Dataset with ReDDiT Reflectance Estimation")
    print("=" * 60)

    # Test RetinexEstimator with a synthetic image
    print("\n1. Testing RetinexEstimator...")

    # Create a synthetic low-light image
    H, W = 256, 256
    synthetic_lq = torch.rand(3, H, W) * 0.3  # Dark image

    # Initialize estimator
    retinex = RetinexEstimator(denoise_method='bilateral')

    # Estimate reflectance
    reflectance, illumination, noise = retinex(synthetic_lq)

    print(f"   Input shape: {synthetic_lq.shape}")
    print(f"   Reflectance shape: {reflectance.shape}")
    print(f"   Illumination shape: {illumination.shape}")
    print(f"   Noise shape: {noise.shape}")
    print(f"   Reflectance range: [{reflectance.min():.4f}, {reflectance.max():.4f}]")
    print(f"   Illumination range: [{illumination.min():.4f}, {illumination.max():.4f}]")
    print(f"   Noise range: [{noise.min():.4f}, {noise.max():.4f}]")

    # Test with batch
    print("\n2. Testing batch processing...")
    batch_lq = torch.rand(4, 3, H, W) * 0.3
    batch_ref, batch_ill, batch_noise = retinex(batch_lq)
    print(f"   Batch input shape: {batch_lq.shape}")
    print(f"   Batch reflectance shape: {batch_ref.shape}")

    # Test different denoising methods
    print("\n3. Testing different denoising methods...")
    for method in ['bilateral', 'gaussian', 'median']:
        est = RetinexEstimator(denoise_method=method)
        ref, ill, noise = est(synthetic_lq)
        print(f"   {method}: noise mean={noise.mean():.6f}, std={noise.std():.6f}")

    print("\n4. Dataset Usage Example:")
    print("""
    # Training loader with reflectance estimation
    train_params = {
        'path': 'dataset/LOLv1/our485',
        'batch_size': 16,
        'patch_size': 256,
        'augment': True,
        'compute_reflectance': True,
        'denoise_method': 'bilateral'
    }
    loader = LOLv1ReDDiT().create_loaders(train_params)

    for batch in loader:
        lq = batch['lq']           # Low-light input
        hq = batch['hq']           # Ground truth
        reflectance = batch['reflectance']  # Estimated clean reflectance
        illumination = batch['illumination']  # Illumination map
        noise = batch['noise']     # Noise map

        # Use reflectance for trajectory refinement in diffusion model
        # x̃₀ = reflectance can be used to shift the Gaussian flow
        ...
    """)

    print("\n5. Integration with Diffusion Model:")
    print("""
    # In your diffusion training loop:

    from lolv1_reddit import RetinexEstimator, compute_refined_trajectory_start

    retinex = RetinexEstimator()

    for batch in train_loader:
        lq = batch['lq'].cuda()
        hq = batch['hq'].cuda()

        # Get estimated clean reflectance for trajectory refinement
        x_tilde_0, components = compute_refined_trajectory_start(lq, retinex)

        # Instead of starting from pure noise, use reflectance-aware residual
        # This reduces the inference gap as shown in ReDDiT paper
        noise = torch.randn_like(hq)

        # Forward diffusion with trajectory shift
        t = sample_timesteps(batch_size)
        alpha_t, sigma_t = get_noise_schedule(t)

        # Shifted noisy sample (RATR module)
        x_t = alpha_t * x_tilde_0 + sigma_t * noise

        # Predict noise and compute loss
        predicted_noise = model(x_t, lq, t)
        loss = F.mse_loss(predicted_noise, noise)
        ...
    """)

    print("\n" + "=" * 60)
    print("Done! The module is ready for use with ReDDiT-style training.")
    print("=" * 60)