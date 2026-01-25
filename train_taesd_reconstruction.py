#!/usr/bin/env python3
"""
Phase P1: Fine-tune Tiny AutoEncoder (TAESD) on LOLv1 dataset.

This script trains the TAESD encoder-decoder to reconstruct both low-light (LL)
and high-light (HL) images from the LOLv1 dataset.

Modes:
- 'll': Train on low-light images only (LL → E → z → D → LL')
- 'hl': Train on high-light images only (HL → E → z → D → HL')
- 'both': Train on both LL and HL images (default)

New Feature - Latent KL Divergence Loss:
    Regularizes the latent space by measuring KL divergence between:
    - z_train: latent from the trainable encoder
    - z_ref: latent from a frozen reference encoder

    Loss = mse_weight * MSE(recon, target) + latent_kl_weight * KL(z_train || z_ref)
    Default: 0.8 * MSE + 0.2 * KL

Usage:
    # Train with latent KL divergence loss (default weights: 0.8 MSE + 0.2 KL)
    python train_taesd_reconstruction.py --data_path /path/to/lolv1/our485 --use_latent_kl --ref_encoder_checkpoint /path/to/pretrained_taesd.pth

    # Custom weights
    python train_taesd_reconstruction.py --data_path /path/to/lolv1/our485 --use_latent_kl --ref_encoder_checkpoint /path/to/pretrained_taesd.pth --l2_weight 0.9 --latent_kl_weight 0.1
"""

import argparse
import os
import time
import sys
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
import glob
from tqdm import tqdm

# Import TAESD model
from ELIR.models.taesd import TAESD

# Optional: perceptual loss
try:
    import lpips

    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not available. Install with 'pip install lpips' for perceptual loss.")

# Optional: FID loss
try:
    from torchvision.models import inception_v3, Inception_V3_Weights
    from scipy import linalg
    import numpy as np

    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    print("Warning: scipy not available. Install with 'pip install scipy' for FID loss.")


class TeeLogger:
    """Logs to both stdout and a file."""

    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ============================================================================
# Latent KL Divergence Loss
# ============================================================================

class LatentKLDivergenceLoss(nn.Module):
    """
    KL Divergence loss between latent distributions from trainable and reference encoders.

    This loss encourages the trainable encoder to produce latents that are close
    to those produced by a frozen reference encoder.

    Two modes are supported:
    1. 'gaussian': Treat latents as Gaussian distributions and compute analytical KL
       KL(N(mu1, sigma1) || N(mu2, sigma2))
    2. 'mse': Simple MSE between latents (equivalent to KL with unit variance assumption)
    3. 'distribution': Estimate distributions from spatial statistics and compute KL
    """

    def __init__(self, ref_encoder, device='cuda', mode='gaussian', eps=1e-6):
        """
        Args:
            ref_encoder: Frozen reference encoder (just the encoder part of TAESD)
            device: Device to use
            mode: 'gaussian', 'mse', or 'distribution'
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.ref_encoder = ref_encoder
        self.device = device
        self.mode = mode
        self.eps = eps

        # Freeze reference encoder
        self.ref_encoder.eval()
        for param in self.ref_encoder.parameters():
            param.requires_grad = False

    def _compute_spatial_statistics(self, z):
        """
        Compute mean and variance from spatial dimensions of latent.
        Treats each channel independently.

        Args:
            z: Latent tensor of shape (B, C, H, W)
        Returns:
            mu: Mean of shape (B, C)
            var: Variance of shape (B, C)
        """
        # Compute mean and variance over spatial dimensions
        mu = z.mean(dim=[2, 3])  # (B, C)
        var = z.var(dim=[2, 3], unbiased=False) + self.eps  # (B, C)
        return mu, var

    def _kl_divergence_gaussian(self, mu1, var1, mu2, var2):
        """
        Compute KL divergence between two Gaussian distributions.
        KL(N(mu1, var1) || N(mu2, var2))

        For diagonal covariance:
        KL = 0.5 * sum(log(var2/var1) + (var1 + (mu1-mu2)^2)/var2 - 1)
        """
        kl = 0.5 * (
                torch.log(var2 / var1) +
                (var1 + (mu1 - mu2) ** 2) / var2 - 1
        )
        return kl.sum(dim=1).mean()  # Sum over channels, mean over batch

    def forward(self, z_train, input_images):
        """
        Compute KL divergence loss.

        Args:
            z_train: Latent from trainable encoder, shape (B, C, H, W)
            input_images: Original input images to get reference latent

        Returns:
            kl_loss: Scalar KL divergence loss
        """
        # Get reference latent from frozen encoder
        with torch.no_grad():
            z_ref = self.ref_encoder(input_images)

        if self.mode == 'mse':
            # Simple MSE between latents
            # This is equivalent to KL with unit variance assumption
            kl_loss = F.mse_loss(z_train, z_ref)

        elif self.mode == 'gaussian':
            # Compute spatial statistics
            mu_train, var_train = self._compute_spatial_statistics(z_train)
            mu_ref, var_ref = self._compute_spatial_statistics(z_ref)

            # KL divergence
            kl_loss = self._kl_divergence_gaussian(mu_train, var_train, mu_ref, var_ref)

        elif self.mode == 'distribution':
            # Treat entire latent as samples from a distribution
            # Compute KL using histogram/kernel density estimation
            # For simplicity, use a combination of mean matching and variance matching

            mu_train, var_train = self._compute_spatial_statistics(z_train)
            mu_ref, var_ref = self._compute_spatial_statistics(z_ref)

            # Mean matching (L2)
            mean_loss = F.mse_loss(mu_train, mu_ref)

            # Variance matching (L2 on log variance for scale invariance)
            var_loss = F.mse_loss(torch.log(var_train), torch.log(var_ref))

            # Also add direct MSE for fine-grained alignment
            direct_loss = F.mse_loss(z_train, z_ref)

            kl_loss = mean_loss + var_loss + 0.1 * direct_loss

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return kl_loss


# ============================================================================
# FID Components (unchanged from original)
# ============================================================================

class FIDFeatureExtractor(nn.Module):
    """Extract features from Inception v3 for FID computation."""

    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.inception.fc = nn.Identity()
        self.inception.eval()
        self.inception.to(device)
        for param in self.inception.parameters():
            param.requires_grad = False
        self.resize = transforms.Resize((299, 299), antialias=True)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x):
        x = self.resize(x)
        x = self.normalize(x)
        with torch.no_grad():
            features = self.inception(x)
        return features


class FIDReferenceDataset(Dataset):
    """Dataset for loading reference images for FID computation."""

    def __init__(self, image_folder, max_images=None):
        super().__init__()
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(image_folder, ext)))
            self.image_paths.extend(glob.glob(os.path.join(image_folder, '**', ext), recursive=True))
        self.image_paths = sorted(list(set(self.image_paths)))
        if max_images is not None and len(self.image_paths) > max_images:
            import random
            random.seed(42)
            self.image_paths = random.sample(self.image_paths, max_images)
        self.transform = transforms.ToTensor()
        print(f"[FIDReferenceDataset] Loaded {len(self.image_paths)} images from {image_folder}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img)


class FIDLoss(nn.Module):
    """FID-based loss for training."""

    def __init__(self, ref_dataset_path, device='cuda', max_ref_images=1000, mode='batch', momentum=0.1):
        super().__init__()
        self.device = device
        self.mode = mode
        self.momentum = momentum
        self.feature_extractor = FIDFeatureExtractor(device)

        print(f"Computing reference FID statistics from {ref_dataset_path}...")
        ref_dataset = FIDReferenceDataset(ref_dataset_path, max_images=max_ref_images)
        ref_loader = DataLoader(ref_dataset, batch_size=32, shuffle=False, num_workers=4)
        self.ref_mu, self.ref_sigma = self._compute_statistics(ref_loader)
        print(f"Reference statistics computed.")

        if mode == 'running':
            self.register_buffer('running_mu', torch.zeros(2048, device=device))
            self.register_buffer('running_sigma', torch.eye(2048, device=device))
            self.register_buffer('num_batches', torch.tensor(0, device=device))

    @torch.no_grad()
    def _compute_statistics(self, dataloader):
        all_features = []
        for batch in tqdm(dataloader, desc="Extracting reference features"):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(self.device)
            features = self.feature_extractor(batch)
            all_features.append(features.cpu())
        all_features = torch.cat(all_features, dim=0).numpy()
        mu = np.mean(all_features, axis=0)
        sigma = np.cov(all_features, rowvar=False)
        return torch.from_numpy(mu).float().to(self.device), torch.from_numpy(sigma).float().to(self.device)

    def _compute_batch_statistics(self, features):
        mu = features.mean(dim=0)
        centered = features - mu.unsqueeze(0)
        sigma = (centered.T @ centered) / (features.shape[0] - 1) + 1e-6 * torch.eye(features.shape[1],
                                                                                     device=features.device)
        return mu, sigma

    def _compute_fid(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        diff = mu1 - mu2
        product = sigma1 @ sigma2 + eps * torch.eye(product.shape[0], device=product.device)
        eigenvalues, eigenvectors = torch.linalg.eigh(product)
        eigenvalues = torch.clamp(eigenvalues, min=eps)
        sqrt_product = eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T
        fid = torch.sum(diff ** 2) + torch.trace(sigma1 + sigma2 - 2 * sqrt_product)
        return fid

    def forward(self, generated_images):
        features = self.feature_extractor(generated_images)
        if self.mode == 'batch':
            gen_mu, gen_sigma = self._compute_batch_statistics(features)
            fid = self._compute_fid(gen_mu, gen_sigma, self.ref_mu, self.ref_sigma)
        else:
            batch_mu, batch_sigma = self._compute_batch_statistics(features)
            with torch.no_grad():
                if self.num_batches == 0:
                    self.running_mu.copy_(batch_mu)
                    self.running_sigma.copy_(batch_sigma)
                else:
                    self.running_mu.mul_(1 - self.momentum).add_(batch_mu * self.momentum)
                    self.running_sigma.mul_(1 - self.momentum).add_(batch_sigma * self.momentum)
                self.num_batches += 1
            fid = self._compute_fid(self.running_mu, self.running_sigma, self.ref_mu, self.ref_sigma)
        return fid

    @torch.no_grad()
    def compute_fid_score(self, dataloader):
        all_features = []
        for batch in tqdm(dataloader, desc="Computing FID"):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(self.device)
            features = self.feature_extractor(batch)
            all_features.append(features.cpu())
        all_features = torch.cat(all_features, dim=0).numpy()
        gen_mu = np.mean(all_features, axis=0)
        gen_sigma = np.cov(all_features, rowvar=False)
        ref_mu = self.ref_mu.cpu().numpy()
        ref_sigma = self.ref_sigma.cpu().numpy()
        diff = gen_mu - ref_mu
        covmean, _ = linalg.sqrtm(gen_sigma @ ref_sigma, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = np.sum(diff ** 2) + np.trace(gen_sigma + ref_sigma - 2 * covmean)
        return float(fid)


# ============================================================================
# Datasets (unchanged from original)
# ============================================================================

class LOLv1ReconDataset(Dataset):
    """Dataset for TAESD reconstruction training on LOLv1."""

    def __init__(self, image_folder, patch_size=256, augment=True, mode='both'):
        super().__init__()
        self.image_folder = image_folder
        self.patch_size = patch_size
        self.augment = augment
        self.mode = mode

        lq_dir = os.path.join(image_folder, "low")
        hq_dir = os.path.join(image_folder, "high")

        self.lq_paths = sorted(glob.glob(os.path.join(lq_dir, "*.png")))
        self.hq_paths = sorted(glob.glob(os.path.join(hq_dir, "*.png")))

        assert len(self.lq_paths) == len(self.hq_paths), \
            f"Mismatch: {len(self.lq_paths)} LL vs {len(self.hq_paths)} HL images"

        self.transform = transforms.ToTensor()
        print(f"[LOLv1ReconDataset] Loaded {len(self.lq_paths)} pairs | mode={mode}")

    def __len__(self):
        if self.mode == 'both':
            return len(self.lq_paths) * 2
        return len(self.lq_paths)

    def _load_and_crop(self, img_path):
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        if self.patch_size > 0 and (w > self.patch_size or h > self.patch_size):
            left = torch.randint(0, max(1, w - self.patch_size), (1,)).item()
            top = torch.randint(0, max(1, h - self.patch_size), (1,)).item()
            img = img.crop((left, top, left + self.patch_size, top + self.patch_size))
        img_tensor = self.transform(img)
        if self.augment:
            if torch.rand(1) < 0.5:
                img_tensor = torch.flip(img_tensor, dims=[2])
            if torch.rand(1) < 0.5:
                img_tensor = torch.flip(img_tensor, dims=[1])
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                img_tensor = torch.rot90(img_tensor, k, dims=[1, 2])
        return img_tensor

    def __getitem__(self, idx):
        if self.mode == 'both':
            is_ll = idx < len(self.lq_paths)
            actual_idx = idx if is_ll else idx - len(self.lq_paths)
            img_path = self.lq_paths[actual_idx] if is_ll else self.hq_paths[actual_idx]
        elif self.mode == 'll':
            img_path = self.lq_paths[idx]
        else:
            img_path = self.hq_paths[idx]
        img = self._load_and_crop(img_path)
        return img, img


class LOLv1ValDataset(Dataset):
    """Validation dataset - returns full images with padding."""

    def __init__(self, image_folder, mode='both', pad_multiple=8):
        super().__init__()
        self.image_folder = image_folder
        self.mode = mode
        self.pad_multiple = pad_multiple

        lq_dir = os.path.join(image_folder, "low")
        hq_dir = os.path.join(image_folder, "high")

        self.lq_paths = sorted(glob.glob(os.path.join(lq_dir, "*.png")))
        self.hq_paths = sorted(glob.glob(os.path.join(hq_dir, "*.png")))

        self.transform = transforms.ToTensor()
        print(f"[LOLv1ValDataset] Loaded {len(self.lq_paths)} pairs | mode={mode}")

    def __len__(self):
        if self.mode == 'both':
            return len(self.lq_paths) * 2
        return len(self.lq_paths)

    def _pad_to_multiple(self, tensor):
        _, h, w = tensor.shape
        pad_h = (self.pad_multiple - h % self.pad_multiple) % self.pad_multiple
        pad_w = (self.pad_multiple - w % self.pad_multiple) % self.pad_multiple
        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor.unsqueeze(0), (0, pad_w, 0, pad_h), mode='reflect').squeeze(0)
        return tensor, h, w

    def __getitem__(self, idx):
        if self.mode == 'both':
            is_ll = idx < len(self.lq_paths)
            actual_idx = idx if is_ll else idx - len(self.lq_paths)
            img_path = self.lq_paths[actual_idx] if is_ll else self.hq_paths[actual_idx]
            img_type = 'll' if is_ll else 'hl'
        elif self.mode == 'll':
            img_path = self.lq_paths[idx]
            img_type = 'll'
        else:
            img_path = self.hq_paths[idx]
            img_type = 'hl'

        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        img_padded, orig_h, orig_w = self._pad_to_multiple(img_tensor)
        return img_padded, img_tensor, orig_h, orig_w, img_type


class LOLv1TrainEvalDataset(Dataset):
    """Training dataset for evaluation - returns full images with padding (no augmentation)."""

    def __init__(self, image_folder, mode='both', pad_multiple=8):
        super().__init__()
        self.image_folder = image_folder
        self.mode = mode
        self.pad_multiple = pad_multiple

        lq_dir = os.path.join(image_folder, "low")
        hq_dir = os.path.join(image_folder, "high")

        self.lq_paths = sorted(glob.glob(os.path.join(lq_dir, "*.png")))
        self.hq_paths = sorted(glob.glob(os.path.join(hq_dir, "*.png")))

        assert len(self.lq_paths) == len(self.hq_paths)
        self.transform = transforms.ToTensor()
        print(f"[LOLv1TrainEvalDataset] Loaded {len(self.lq_paths)} pairs | mode={mode}")

    def __len__(self):
        if self.mode == 'both':
            return len(self.lq_paths) * 2
        return len(self.lq_paths)

    def _pad_to_multiple(self, tensor):
        _, h, w = tensor.shape
        pad_h = (self.pad_multiple - h % self.pad_multiple) % self.pad_multiple
        pad_w = (self.pad_multiple - w % self.pad_multiple) % self.pad_multiple
        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor.unsqueeze(0), (0, pad_w, 0, pad_h), mode='reflect').squeeze(0)
        return tensor, h, w

    def __getitem__(self, idx):
        if self.mode == 'both':
            is_ll = idx < len(self.lq_paths)
            actual_idx = idx if is_ll else idx - len(self.lq_paths)
            img_path = self.lq_paths[actual_idx] if is_ll else self.hq_paths[actual_idx]
            img_type = 'll' if is_ll else 'hl'
        elif self.mode == 'll':
            img_path = self.lq_paths[idx]
            img_type = 'll'
        else:
            img_path = self.hq_paths[idx]
            img_type = 'hl'

        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        img_padded, orig_h, orig_w = self._pad_to_multiple(img_tensor)
        return img_padded, img_tensor, orig_h, orig_w, img_type


# ============================================================================
# Trainer
# ============================================================================

class TAESDTrainer:
    """Trainer for TAESD reconstruction."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if args.device != 'auto' else
                                   ('cuda' if torch.cuda.is_available() else
                                    ('mps' if torch.backends.mps.is_available() else 'cpu')))

        print(f"Using device: {self.device}")

        # Create output directory
        if args.eval_only and args.resume:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = os.path.splitext(os.path.basename(args.resume))[0]
            self.run_name = f"eval_{checkpoint_name}_{timestamp}"
            self.output_dir = os.path.join(args.output_dir, self.run_name)
            self.checkpoint_dir = None
            self.samples_dir = os.path.join(self.output_dir, "samples")
            os.makedirs(self.samples_dir, exist_ok=True)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"taesd_p1_{args.mode}_{timestamp}"
            self.output_dir = os.path.join(args.output_dir, self.run_name)
            self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
            self.samples_dir = os.path.join(self.output_dir, "samples")
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            os.makedirs(self.samples_dir, exist_ok=True)

        # Setup logging to file
        log_file = os.path.join(self.output_dir, "training.log")
        sys.stdout = TeeLogger(log_file)
        sys.stderr = TeeLogger(log_file)
        print(f"Logging to: {log_file}")

        # Initialize model
        self.model = TAESD(pretrained=True).to(self.device)
        print(f"TAESD model loaded with pretrained weights")

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

        # Perceptual loss (optional)
        self.lpips_loss = None
        if args.use_lpips and LPIPS_AVAILABLE:
            self.lpips_loss = lpips.LPIPS(net='vgg').to(self.device)
            self.lpips_loss.eval()
            for p in self.lpips_loss.parameters():
                p.requires_grad = False
            print("LPIPS perceptual loss enabled")

        # FID loss (optional)
        self.fid_loss = None
        if args.use_fid:
            if not FID_AVAILABLE:
                raise RuntimeError("FID loss requested but scipy is not available.")
            if not args.fid_ref_path:
                raise ValueError("FID loss requires --fid_ref_path")
            self.fid_loss = FIDLoss(
                ref_dataset_path=args.fid_ref_path,
                device=self.device,
                max_ref_images=args.fid_max_ref_images,
                mode=args.fid_mode,
                momentum=args.fid_momentum
            )
            print(f"FID loss enabled with reference: {args.fid_ref_path}")

        # Latent KL Divergence loss (optional)
        self.latent_kl_loss = None
        if args.use_latent_kl:
            if not args.ref_encoder_checkpoint:
                raise ValueError("Latent KL loss requires --ref_encoder_checkpoint")

            # Load reference encoder (frozen)
            print(f"Loading reference encoder from {args.ref_encoder_checkpoint}")
            ref_taesd = TAESD(pretrained=False).to(self.device)
            ref_ckpt = torch.load(args.ref_encoder_checkpoint, map_location=self.device)
            if 'model_state_dict' in ref_ckpt:
                ref_taesd.load_state_dict(ref_ckpt['model_state_dict'])
            else:
                ref_taesd.load_state_dict(ref_ckpt)

            # Use only the encoder part
            ref_encoder = ref_taesd.encoder
            ref_encoder.eval()
            for p in ref_encoder.parameters():
                p.requires_grad = False

            self.latent_kl_loss = LatentKLDivergenceLoss(
                ref_encoder=ref_encoder,
                device=self.device,
                mode=args.latent_kl_mode
            )
            print(f"Latent KL loss enabled (mode: {args.latent_kl_mode}, weight: {args.latent_kl_weight})")

        # Optimizer (only for training)
        if not args.eval_only:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=args.epochs,
                eta_min=args.lr * 0.01
            )
        else:
            self.optimizer = None
            self.scheduler = None

        # Datasets and dataloaders
        if not args.eval_only:
            self.train_dataset = LOLv1ReconDataset(
                args.data_path,
                patch_size=args.patch_size,
                augment=True,
                mode=args.mode
            )
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True
            )

        self.train_eval_dataset = LOLv1TrainEvalDataset(args.data_path, mode=args.mode)
        self.train_eval_loader = DataLoader(self.train_eval_dataset, batch_size=1, shuffle=False,
                                            num_workers=args.num_workers)

        if args.val_path:
            self.val_dataset = LOLv1ValDataset(args.val_path, mode=args.mode)
            self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
        else:
            self.val_loader = None

        # Tensorboard
        if not args.eval_only:
            self.writer = SummaryWriter(os.path.join(self.output_dir, "logs"))
        else:
            self.writer = None

        # Training state
        self.start_epoch = 0
        self.global_step = 0
        self.best_psnr = 0

        if args.resume:
            self.load_checkpoint(args.resume, eval_only=args.eval_only)

    def compute_loss(self, pred, target, inputs=None, z_train=None, compute_fid=True):
        """
        Compute reconstruction loss with optional latent KL divergence.

        Args:
            pred: Reconstructed images
            target: Target images
            inputs: Original input images (needed for latent KL loss)
            z_train: Latent from trainable encoder (needed for latent KL loss)
            compute_fid: Whether to compute FID loss
        """
        # L1 loss
        loss_l1 = self.l1_loss(pred, target)

        # L2/MSE loss
        loss_l2 = self.mse_loss(pred, target)

        # Combined reconstruction loss
        loss = self.args.l1_weight * loss_l1 + self.args.l2_weight * loss_l2

        # Perceptual loss (optional)
        loss_lpips = torch.tensor(0.0, device=self.device)
        if self.lpips_loss is not None:
            pred_lpips = pred * 2 - 1
            target_lpips = target * 2 - 1
            loss_lpips = self.lpips_loss(pred_lpips, target_lpips).mean()
            loss = loss + self.args.lpips_weight * loss_lpips

        # FID loss (optional)
        loss_fid = torch.tensor(0.0, device=self.device)
        if self.fid_loss is not None and compute_fid:
            if self.global_step % self.args.fid_every == 0:
                loss_fid = self.fid_loss(pred)
                loss = loss + self.args.fid_weight * loss_fid

        # Latent KL divergence loss (optional)
        loss_latent_kl = torch.tensor(0.0, device=self.device)
        if self.latent_kl_loss is not None and inputs is not None and z_train is not None:
            loss_latent_kl = self.latent_kl_loss(z_train, inputs)
            loss = loss + self.args.latent_kl_weight * loss_latent_kl

        return loss, {
            'l1': loss_l1.item(),
            'l2': loss_l2.item(),
            'lpips': loss_lpips.item(),
            'fid': loss_fid.item(),
            'latent_kl': loss_latent_kl.item()
        }

    def compute_psnr(self, pred, target):
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return float('inf')
        return 10 * torch.log10(1.0 / mse).item()

    @torch.no_grad()
    def evaluate_dataset(self, dataloader, dataset_name, save_samples=True, epoch=None):
        self.model.eval()
        total_psnr = 0
        total_psnr_ll = 0
        total_psnr_hl = 0
        total_loss = 0
        count = 0
        count_ll = 0
        count_hl = 0
        sample_images_ll = []
        sample_images_hl = []
        all_outputs = []

        pbar = tqdm(dataloader, desc=f"Evaluating {dataset_name}")
        for batch_idx, (padded, original, orig_h, orig_w, img_type) in enumerate(pbar):
            padded = padded.to(self.device)
            original = original.to(self.device)
            output = self.model(padded)
            output = output[:, :, :orig_h, :orig_w].clamp(0, 1)

            psnr = self.compute_psnr(output, original)
            loss, _ = self.compute_loss(output, original, compute_fid=False)

            total_psnr += psnr
            total_loss += loss.item()
            count += 1

            if self.fid_loss is not None:
                all_outputs.append(output.cpu())

            if img_type[0] == 'll':
                total_psnr_ll += psnr
                count_ll += 1
                if save_samples and len(sample_images_ll) < 2:
                    sample_images_ll.append({'input': original[0].cpu(), 'output': output[0].cpu(), 'type': 'll'})
            else:
                total_psnr_hl += psnr
                count_hl += 1
                if save_samples and len(sample_images_hl) < 2:
                    sample_images_hl.append({'input': original[0].cpu(), 'output': output[0].cpu(), 'type': 'hl'})

            pbar.set_postfix({'psnr': f'{psnr:.2f}'})

        metrics = {'psnr': total_psnr / count if count > 0 else 0, 'loss': total_loss / count if count > 0 else 0,
                   'count': count}
        if count_ll > 0:
            metrics['psnr_ll'] = total_psnr_ll / count_ll
            metrics['count_ll'] = count_ll
        if count_hl > 0:
            metrics['psnr_hl'] = total_psnr_hl / count_hl
            metrics['count_hl'] = count_hl

        if self.fid_loss is not None and all_outputs:
            class OutputDataset(Dataset):
                def __init__(self, outputs): self.outputs = torch.cat(outputs, dim=0)

                def __len__(self): return len(self.outputs)

                def __getitem__(self, idx): return self.outputs[idx]

            output_loader = DataLoader(OutputDataset(all_outputs), batch_size=32, shuffle=False)
            fid_score = self.fid_loss.compute_fid_score(output_loader)
            metrics['fid'] = fid_score

        if save_samples:
            sample_images = sample_images_ll + sample_images_hl
            if sample_images:
                epoch_str = f"epoch_{epoch:04d}" if epoch is not None else "eval"
                self.save_samples(sample_images, f"{dataset_name}_{epoch_str}")

        return metrics

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0
        epoch_psnr = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.args.epochs}")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass - get latent and reconstruction separately for KL loss
            z_train = self.model.encoder(inputs)
            outputs = self.model.decoder(z_train)

            # Compute loss with latent KL
            loss, loss_dict = self.compute_loss(outputs, targets, inputs=inputs, z_train=z_train)

            loss.backward()
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()

            with torch.no_grad():
                psnr = self.compute_psnr(outputs.clamp(0, 1), targets)

            epoch_loss += loss.item()
            epoch_psnr += psnr

            self.global_step += 1
            if self.global_step % self.args.log_interval == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/psnr', psnr, self.global_step)
                self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)
                for k, v in loss_dict.items():
                    if v > 0:
                        self.writer.add_scalar(f'train/loss_{k}', v, self.global_step)

            postfix = {'loss': f'{loss.item():.4f}', 'psnr': f'{psnr:.2f}'}
            if loss_dict['latent_kl'] > 0:
                postfix['kl'] = f'{loss_dict["latent_kl"]:.4f}'
            pbar.set_postfix(postfix)

        return epoch_loss / len(self.train_loader), epoch_psnr / len(self.train_loader)

    @torch.no_grad()
    def validate(self, epoch):
        if self.val_loader is None:
            return 0, 0
        self.model.eval()
        total_psnr = 0
        total_loss = 0
        count = 0
        sample_images = []
        all_outputs = []

        for batch_idx, (padded, original, orig_h, orig_w, img_type) in enumerate(self.val_loader):
            padded = padded.to(self.device)
            original = original.to(self.device)
            output = self.model(padded)
            output = output[:, :, :orig_h, :orig_w].clamp(0, 1)

            psnr = self.compute_psnr(output, original)
            loss, _ = self.compute_loss(output, original, compute_fid=False)
            total_psnr += psnr
            total_loss += loss.item()
            count += 1

            if self.fid_loss is not None:
                all_outputs.append(output.cpu())
            if len(sample_images) < 4:
                sample_images.append({'input': original[0].cpu(), 'output': output[0].cpu(), 'type': img_type[0]})

        avg_psnr = total_psnr / count
        avg_loss = total_loss / count

        if self.writer:
            self.writer.add_scalar('val/psnr', avg_psnr, epoch)
            self.writer.add_scalar('val/loss', avg_loss, epoch)
            if self.fid_loss is not None and all_outputs:
                class OutputDataset(Dataset):
                    def __init__(self, outputs): self.outputs = torch.cat(outputs, dim=0)

                    def __len__(self): return len(self.outputs)

                    def __getitem__(self, idx): return self.outputs[idx]

                output_loader = DataLoader(OutputDataset(all_outputs), batch_size=32, shuffle=False)
                fid_score = self.fid_loss.compute_fid_score(output_loader)
                self.writer.add_scalar('val/fid', fid_score, epoch)

        if sample_images:
            self.save_samples(sample_images, f"val_epoch_{epoch:04d}")

        return avg_psnr, avg_loss

    def save_samples(self, samples, name):
        fig_tensors = []
        for s in samples:
            diff = (s['input'] - s['output']).abs()
            row = torch.cat([s['input'], s['output'], diff * 5], dim=2)
            fig_tensors.append(row)
        grid = make_grid(fig_tensors, nrow=1, padding=2, normalize=False)
        save_image(grid, os.path.join(self.samples_dir, f"{name}.png"))
        if self.writer:
            self.writer.add_image(f'samples/{name}', grid, 0)

    def save_checkpoint(self, epoch, is_best=False):
        if self.checkpoint_dir is None:
            return
        checkpoint = {
            'epoch': epoch, 'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_psnr': self.best_psnr, 'args': vars(self.args)
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, "latest.pth"))
        if (epoch + 1) % self.args.save_every == 0:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, f"epoch_{epoch + 1:04d}.pth"))
        if is_best:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, "best.pth"))

    def load_checkpoint(self, path, eval_only=False):
        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if not eval_only:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.global_step = checkpoint['global_step']
            self.best_psnr = checkpoint.get('best_psnr', 0)

    def run_initial_evaluation(self):
        print(f"\n{'=' * 60}\nRunning initial evaluation...\n{'=' * 60}")
        train_metrics = self.evaluate_dataset(self.train_eval_loader, "train", save_samples=True, epoch=-1)
        print(f"  Train - PSNR: {train_metrics['psnr']:.2f} dB")
        if self.val_loader:
            val_metrics = self.evaluate_dataset(self.val_loader, "val", save_samples=True, epoch=-1)
            print(f"  Val - PSNR: {val_metrics['psnr']:.2f} dB")
        print(f"{'=' * 60}\n")

    def run_evaluation_only(self):
        print(f"\n{'=' * 60}\nRunning evaluation...\n{'=' * 60}")
        train_metrics = self.evaluate_dataset(self.train_eval_loader, "train", save_samples=True, epoch=None)
        print(f"  Train - PSNR: {train_metrics['psnr']:.2f} dB")
        if self.val_loader:
            val_metrics = self.evaluate_dataset(self.val_loader, "val", save_samples=True, epoch=None)
            print(f"  Val - PSNR: {val_metrics['psnr']:.2f} dB")
        print(f"{'=' * 60}\n")

    def train(self):
        print(f"\n{'=' * 60}\nStarting training: {self.run_name}\n{'=' * 60}")
        print(
            f"Loss config: L1={self.args.l1_weight}, L2={self.args.l2_weight}, KL={self.args.latent_kl_weight if self.latent_kl_loss else 0}")

        self.run_initial_evaluation()

        for epoch in range(self.start_epoch, self.args.epochs):
            train_loss, train_psnr = self.train_epoch(epoch)
            print(f"Epoch {epoch + 1}/{self.args.epochs} - Loss: {train_loss:.4f}, PSNR: {train_psnr:.2f} dB")

            if self.val_loader and (epoch + 1) % self.args.val_every == 0:
                val_psnr, val_loss = self.validate(epoch)
                print(f"  Val - Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f} dB")
                is_best = val_psnr > self.best_psnr
                if is_best:
                    self.best_psnr = val_psnr
                    print(f"  New best PSNR: {val_psnr:.2f} dB")
                self.save_checkpoint(epoch, is_best)
            else:
                self.save_checkpoint(epoch)

            self.scheduler.step()

        print(f"\nTraining complete! Best PSNR: {self.best_psnr:.2f} dB")
        if self.writer:
            self.writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune TAESD on LOLv1')

    # Data
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--val_path', type=str, default=None)
    parser.add_argument('--mode', type=str, default='both', choices=['ll', 'hl', 'both'])

    # Model
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--eval_only', action='store_true')

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    # Loss weights (default: 0.8 MSE + 0.2 KL when latent KL is enabled)
    parser.add_argument('--l1_weight', type=float, default=0.0, help='Weight for L1 loss')
    parser.add_argument('--l2_weight', type=float, default=1.0,
                        help='Weight for L2/MSE loss')
    parser.add_argument('--use_lpips', action='store_true')
    parser.add_argument('--lpips_weight', type=float, default=0.1)

    # FID loss
    parser.add_argument('--use_fid', action='store_true')
    parser.add_argument('--fid_ref_path', type=str, default=None)
    parser.add_argument('--fid_weight', type=float, default=0.01)
    parser.add_argument('--fid_max_ref_images', type=int, default=1000)
    parser.add_argument('--fid_mode', type=str, default='batch', choices=['batch', 'running'])
    parser.add_argument('--fid_momentum', type=float, default=0.1)
    parser.add_argument('--fid_every', type=int, default=10)

    # Latent KL Divergence loss (NEW)
    parser.add_argument('--use_latent_kl', action='store_true',
                        help='Use latent KL divergence loss with frozen reference encoder')
    parser.add_argument('--ref_encoder_checkpoint', type=str, default=None,
                        help='Path to checkpoint for frozen reference encoder')
    parser.add_argument('--latent_kl_weight', type=float, default=0.0,
                        help='Weight for latent KL divergence loss')
    parser.add_argument('--latent_kl_mode', type=str, default='gaussian',
                        choices=['gaussian', 'mse', 'distribution'],
                        help='Mode for computing latent KL divergence')

    # Misc
    parser.add_argument('--output_dir', type=str, default='./out_p1')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--val_every', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=10)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.eval_only and args.resume is None:
        raise ValueError("--eval_only requires --resume")
    if args.use_fid and args.fid_ref_path is None:
        raise ValueError("--use_fid requires --fid_ref_path")
    if args.use_latent_kl and args.ref_encoder_checkpoint is None:
        raise ValueError("--use_latent_kl requires --ref_encoder_checkpoint")

    trainer = TAESDTrainer(args)
    if args.eval_only:
        trainer.run_evaluation_only()
    else:
        trainer.train()


if __name__ == '__main__':
    main()