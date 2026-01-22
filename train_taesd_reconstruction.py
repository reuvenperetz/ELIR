#!/usr/bin/env python3
"""
Phase P1: Fine-tune Tiny AutoEncoder (TAESD) on LOLv1 dataset.

This script trains the TAESD encoder-decoder to reconstruct both low-light (LL)
and high-light (HL) images from the LOLv1 dataset.

Modes:
- 'll': Train on low-light images only (LL → E → z → D → LL')
- 'hl': Train on high-light images only (HL → E → z → D → HL')
- 'both': Train on both LL and HL images (default)

Usage:
    # Train on both LL and HL images
    python train_taesd_reconstruction.py --data_path /path/to/lolv1/our485 --mode both

    # Train on low-light images only
    python train_taesd_reconstruction.py --data_path /path/to/lolv1/our485 --mode ll

    # Train on high-light images only
    python train_taesd_reconstruction.py --data_path /path/to/lolv1/our485 --mode hl

    # Resume from checkpoint
    python train_taesd_reconstruction.py --data_path /path/to/lolv1/our485 --resume checkpoints/taesd_p1_epoch_50.pth

    # Evaluate a saved checkpoint only (no training)
    python train_taesd_reconstruction.py --data_path /path/to/lolv1/our485 --val_path /path/to/lolv1/eval15 --eval_only --resume checkpoints/best.pth

    # Train with FID loss against a reference dataset
    python train_taesd_reconstruction.py --data_path /path/to/lolv1/our485 --use_fid --fid_ref_path /path/to/lolv2-real/high --fid_weight 0.01
"""

import argparse
import os
import time
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


class FIDFeatureExtractor(nn.Module):
    """
    Extract features from Inception v3 for FID computation.
    Uses the pool3 layer (2048-dimensional features).
    """

    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device

        # Load pretrained Inception v3
        self.inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.inception.fc = nn.Identity()  # Remove final FC layer
        self.inception.eval()
        self.inception.to(device)

        # Freeze all parameters
        for param in self.inception.parameters():
            param.requires_grad = False

        # Preprocessing for Inception (expects 299x299 images, normalized)
        self.resize = transforms.Resize((299, 299), antialias=True)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, x):
        """
        Extract 2048-dim features from images.
        Args:
            x: Tensor of shape (B, 3, H, W) in range [0, 1]
        Returns:
            features: Tensor of shape (B, 2048)
        """
        # Resize to 299x299
        x = self.resize(x)
        # Normalize
        x = self.normalize(x)
        # Extract features
        with torch.no_grad():
            features = self.inception(x)
        return features


class FIDReferenceDataset(Dataset):
    """Dataset for loading reference images for FID computation."""

    def __init__(self, image_folder, max_images=None):
        """
        Args:
            image_folder: Path to folder containing reference images
            max_images: Maximum number of images to use (None for all)
        """
        super().__init__()
        self.image_folder = image_folder

        # Support multiple image formats
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(image_folder, ext)))
            self.image_paths.extend(glob.glob(os.path.join(image_folder, '**', ext), recursive=True))

        self.image_paths = sorted(list(set(self.image_paths)))

        if max_images is not None and len(self.image_paths) > max_images:
            # Randomly sample images
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
    """
    FID-based loss for training.

    This computes a differentiable approximation to FID by comparing
    the feature statistics of generated images to precomputed reference statistics.

    Two modes are supported:
    1. 'batch': Compare batch statistics to reference (faster, noisier)
    2. 'running': Maintain running statistics and compare periodically (more stable)
    """

    def __init__(self, ref_dataset_path, device='cuda', max_ref_images=1000,
                 mode='batch', momentum=0.1):
        """
        Args:
            ref_dataset_path: Path to reference dataset folder
            device: Device to use
            max_ref_images: Maximum reference images to use for statistics
            mode: 'batch' or 'running'
            momentum: Momentum for running statistics (only used in 'running' mode)
        """
        super().__init__()
        self.device = device
        self.mode = mode
        self.momentum = momentum

        # Feature extractor
        self.feature_extractor = FIDFeatureExtractor(device)

        # Load reference dataset and compute statistics
        print(f"Computing reference FID statistics from {ref_dataset_path}...")
        ref_dataset = FIDReferenceDataset(ref_dataset_path, max_images=max_ref_images)
        ref_loader = DataLoader(ref_dataset, batch_size=32, shuffle=False, num_workers=4)

        self.ref_mu, self.ref_sigma = self._compute_statistics(ref_loader)
        print(f"Reference statistics computed: mu shape={self.ref_mu.shape}, sigma shape={self.ref_sigma.shape}")

        # Running statistics for generated images (used in 'running' mode)
        if mode == 'running':
            self.register_buffer('running_mu', torch.zeros(2048, device=device))
            self.register_buffer('running_sigma', torch.eye(2048, device=device))
            self.register_buffer('num_batches', torch.tensor(0, device=device))

    @torch.no_grad()
    def _compute_statistics(self, dataloader):
        """Compute mean and covariance of features from a dataloader."""
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

        return torch.from_numpy(mu).float().to(self.device), \
            torch.from_numpy(sigma).float().to(self.device)

    def _compute_batch_statistics(self, features):
        """Compute mean and covariance from a batch of features."""
        # features: (B, 2048)
        mu = features.mean(dim=0)

        # Compute covariance
        centered = features - mu.unsqueeze(0)
        # Add small regularization for numerical stability
        sigma = (centered.T @ centered) / (features.shape[0] - 1) + 1e-6 * torch.eye(
            features.shape[1], device=features.device)

        return mu, sigma

    def _compute_fid(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        Compute FID between two Gaussians.

        FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1 @ sigma2))

        This is a differentiable approximation using matrix square root.
        """
        diff = mu1 - mu2

        # Compute sqrt(sigma1 @ sigma2) using eigendecomposition for stability
        # This is differentiable through torch
        product = sigma1 @ sigma2

        # Add small regularization
        product = product + eps * torch.eye(product.shape[0], device=product.device)

        # Compute matrix square root via eigendecomposition
        # Note: This is an approximation that's more stable for training
        eigenvalues, eigenvectors = torch.linalg.eigh(product)
        eigenvalues = torch.clamp(eigenvalues, min=eps)  # Ensure positive
        sqrt_product = eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T

        # FID formula
        fid = torch.sum(diff ** 2) + torch.trace(sigma1 + sigma2 - 2 * sqrt_product)

        return fid

    def forward(self, generated_images):
        """
        Compute FID loss for a batch of generated images.

        Args:
            generated_images: Tensor of shape (B, 3, H, W) in range [0, 1]

        Returns:
            fid_loss: Scalar tensor (differentiable w.r.t. generated_images through features)
        """
        # Extract features (this part is not differentiable, but we can still use
        # the statistics comparison as a training signal)
        # For a truly differentiable version, we'd need to backprop through Inception

        # Get features
        features = self.feature_extractor(generated_images)

        if self.mode == 'batch':
            # Compute batch statistics
            gen_mu, gen_sigma = self._compute_batch_statistics(features)

            # Compute FID
            fid = self._compute_fid(gen_mu, gen_sigma, self.ref_mu, self.ref_sigma)

        else:  # 'running' mode
            # Update running statistics
            batch_mu, batch_sigma = self._compute_batch_statistics(features)

            with torch.no_grad():
                if self.num_batches == 0:
                    self.running_mu.copy_(batch_mu)
                    self.running_sigma.copy_(batch_sigma)
                else:
                    self.running_mu.mul_(1 - self.momentum).add_(batch_mu * self.momentum)
                    self.running_sigma.mul_(1 - self.momentum).add_(batch_sigma * self.momentum)
                self.num_batches += 1

            # Compute FID using running statistics
            fid = self._compute_fid(self.running_mu, self.running_sigma,
                                    self.ref_mu, self.ref_sigma)

        return fid

    @torch.no_grad()
    def compute_fid_score(self, dataloader):
        """
        Compute actual FID score (non-differentiable) for evaluation.

        Args:
            dataloader: DataLoader yielding generated images

        Returns:
            fid_score: Float FID score
        """
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

        # Compute FID using scipy for numerical stability
        ref_mu = self.ref_mu.cpu().numpy()
        ref_sigma = self.ref_sigma.cpu().numpy()

        diff = gen_mu - ref_mu

        # Compute sqrt(sigma1 @ sigma2)
        covmean, _ = linalg.sqrtm(gen_sigma @ ref_sigma, disp=False)

        # Handle numerical issues
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = np.sum(diff ** 2) + np.trace(gen_sigma + ref_sigma - 2 * covmean)

        return float(fid)


class LOLv1ReconDataset(Dataset):
    """
    Dataset for TAESD reconstruction training on LOLv1.
    Returns both low-light and high-light images based on mode.
    """

    def __init__(self, image_folder, patch_size=256, augment=True, mode='both'):
        """
        Args:
            image_folder: Path to dataset folder (e.g., our485) containing low/ and high/
            patch_size: Size of random crops
            augment: Apply random flips/rotations
            mode: 'll' (low-light only), 'hl' (high-light only), or 'both'
        """
        super().__init__()
        self.image_folder = image_folder
        self.patch_size = patch_size
        self.augment = augment
        self.mode = mode

        # Get image paths
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
        """Load image, apply random crop and augmentation."""
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        # Random crop
        if self.patch_size > 0 and (w > self.patch_size or h > self.patch_size):
            left = torch.randint(0, max(1, w - self.patch_size), (1,)).item()
            top = torch.randint(0, max(1, h - self.patch_size), (1,)).item()
            img = img.crop((left, top, left + self.patch_size, top + self.patch_size))

        # Convert to tensor
        img_tensor = self.transform(img)

        # Random augmentation
        if self.augment:
            # Random horizontal flip
            if torch.rand(1) < 0.5:
                img_tensor = torch.flip(img_tensor, dims=[2])
            # Random vertical flip
            if torch.rand(1) < 0.5:
                img_tensor = torch.flip(img_tensor, dims=[1])
            # Random 90-degree rotation
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                img_tensor = torch.rot90(img_tensor, k, dims=[1, 2])

        return img_tensor

    def __getitem__(self, idx):
        if self.mode == 'both':
            # First half: LL images, second half: HL images
            is_ll = idx < len(self.lq_paths)
            actual_idx = idx if is_ll else idx - len(self.lq_paths)
            img_path = self.lq_paths[actual_idx] if is_ll else self.hq_paths[actual_idx]
        elif self.mode == 'll':
            img_path = self.lq_paths[idx]
        else:  # 'hl'
            img_path = self.hq_paths[idx]

        img = self._load_and_crop(img_path)
        return img, img  # Return same image as input and target (reconstruction task)


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
        """Pad tensor so H and W are divisible by pad_multiple."""
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
    """
    Training dataset for evaluation - returns full images with padding (no augmentation).
    Used to evaluate reconstruction quality on the training set.
    """

    def __init__(self, image_folder, mode='both', pad_multiple=8):
        super().__init__()
        self.image_folder = image_folder
        self.mode = mode
        self.pad_multiple = pad_multiple

        lq_dir = os.path.join(image_folder, "low")
        hq_dir = os.path.join(image_folder, "high")

        self.lq_paths = sorted(glob.glob(os.path.join(lq_dir, "*.png")))
        self.hq_paths = sorted(glob.glob(os.path.join(hq_dir, "*.png")))

        assert len(self.lq_paths) == len(self.hq_paths), \
            f"Mismatch: {len(self.lq_paths)} LL vs {len(self.hq_paths)} HL images"

        self.transform = transforms.ToTensor()
        print(f"[LOLv1TrainEvalDataset] Loaded {len(self.lq_paths)} pairs | mode={mode}")

    def __len__(self):
        if self.mode == 'both':
            return len(self.lq_paths) * 2
        return len(self.lq_paths)

    def _pad_to_multiple(self, tensor):
        """Pad tensor so H and W are divisible by pad_multiple."""
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


class TAESDTrainer:
    """Trainer for TAESD reconstruction."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if args.device != 'auto' else
                                   ('cuda' if torch.cuda.is_available() else
                                    ('mps' if torch.backends.mps.is_available() else 'cpu')))

        print(f"Using device: {self.device}")

        # Create output directory (only if not eval_only mode or if no resume path)
        if args.eval_only and args.resume:
            # For eval_only mode, create a simple output dir for results
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
                raise RuntimeError("FID loss requested but scipy is not available. "
                                   "Install with 'pip install scipy'")
            if not args.fid_ref_path:
                raise ValueError("FID loss requires --fid_ref_path to specify reference dataset")

            self.fid_loss = FIDLoss(
                ref_dataset_path=args.fid_ref_path,
                device=self.device,
                max_ref_images=args.fid_max_ref_images,
                mode=args.fid_mode,
                momentum=args.fid_momentum
            )
            print(f"FID loss enabled with reference: {args.fid_ref_path}")
            print(f"  Mode: {args.fid_mode}, Weight: {args.fid_weight}")

        # Optimizer (only needed for training)
        if not args.eval_only:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay
            )

            # Learning rate scheduler
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

        # Training evaluation dataset (full images, no augmentation)
        self.train_eval_dataset = LOLv1TrainEvalDataset(
            args.data_path,
            mode=args.mode
        )
        self.train_eval_loader = DataLoader(
            self.train_eval_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers
        )

        # Validation dataset
        if args.val_path:
            self.val_dataset = LOLv1ValDataset(args.val_path, mode=args.mode)
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers
            )
        else:
            self.val_loader = None

        # Tensorboard (only for training or full eval)
        if not args.eval_only:
            self.writer = SummaryWriter(os.path.join(self.output_dir, "logs"))
        else:
            self.writer = None

        # Training state
        self.start_epoch = 0
        self.global_step = 0
        self.best_psnr = 0

        # Resume from checkpoint
        if args.resume:
            self.load_checkpoint(args.resume, eval_only=args.eval_only)

    def compute_loss(self, pred, target, compute_fid=True):
        """Compute reconstruction loss."""
        # L1 loss
        loss_l1 = self.l1_loss(pred, target)

        # L2/MSE loss
        loss_l2 = self.mse_loss(pred, target)

        # Combined loss
        loss = self.args.l1_weight * loss_l1 + self.args.l2_weight * loss_l2

        # Perceptual loss (optional)
        loss_lpips = torch.tensor(0.0, device=self.device)
        if self.lpips_loss is not None:
            # LPIPS expects input in [-1, 1]
            pred_lpips = pred * 2 - 1
            target_lpips = target * 2 - 1
            loss_lpips = self.lpips_loss(pred_lpips, target_lpips).mean()
            loss = loss + self.args.lpips_weight * loss_lpips

        # FID loss (optional)
        loss_fid = torch.tensor(0.0, device=self.device)
        if self.fid_loss is not None and compute_fid:
            # Only compute FID loss every N steps to save computation
            if self.global_step % self.args.fid_every == 0:
                loss_fid = self.fid_loss(pred)
                loss = loss + self.args.fid_weight * loss_fid

        return loss, {
            'l1': loss_l1.item(),
            'l2': loss_l2.item(),
            'lpips': loss_lpips.item(),
            'fid': loss_fid.item()
        }

    def compute_psnr(self, pred, target):
        """Compute PSNR between prediction and target."""
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return float('inf')
        return 10 * torch.log10(1.0 / mse).item()

    @torch.no_grad()
    def evaluate_dataset(self, dataloader, dataset_name, save_samples=True, epoch=None):
        """
        Evaluate reconstruction quality on a dataset.

        Args:
            dataloader: DataLoader for evaluation
            dataset_name: Name of the dataset (for logging)
            save_samples: Whether to save sample images
            epoch: Epoch number (for logging), None for standalone evaluation

        Returns:
            dict: Evaluation metrics
        """
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

        # Collect all outputs for FID computation
        all_outputs = []

        pbar = tqdm(dataloader, desc=f"Evaluating {dataset_name}")
        for batch_idx, (padded, original, orig_h, orig_w, img_type) in enumerate(pbar):
            padded = padded.to(self.device)
            original = original.to(self.device)

            # Forward pass
            output = self.model(padded)

            # Remove padding
            output = output[:, :, :orig_h, :orig_w]
            output = output.clamp(0, 1)

            # Compute metrics (without FID for per-sample evaluation)
            psnr = self.compute_psnr(output, original)
            loss, _ = self.compute_loss(output, original, compute_fid=False)

            total_psnr += psnr
            total_loss += loss.item()
            count += 1

            # Collect outputs for FID
            if self.fid_loss is not None:
                all_outputs.append(output.cpu())

            # Track per-type metrics
            if img_type[0] == 'll':
                total_psnr_ll += psnr
                count_ll += 1
                if save_samples and len(sample_images_ll) < 2:
                    sample_images_ll.append({
                        'input': original[0].cpu(),
                        'output': output[0].cpu(),
                        'type': 'll'
                    })
            else:
                total_psnr_hl += psnr
                count_hl += 1
                if save_samples and len(sample_images_hl) < 2:
                    sample_images_hl.append({
                        'input': original[0].cpu(),
                        'output': output[0].cpu(),
                        'type': 'hl'
                    })

            pbar.set_postfix({'psnr': f'{psnr:.2f}'})

        # Compute averages
        metrics = {
            'psnr': total_psnr / count if count > 0 else 0,
            'loss': total_loss / count if count > 0 else 0,
            'count': count
        }

        if count_ll > 0:
            metrics['psnr_ll'] = total_psnr_ll / count_ll
            metrics['count_ll'] = count_ll
        if count_hl > 0:
            metrics['psnr_hl'] = total_psnr_hl / count_hl
            metrics['count_hl'] = count_hl

        # Compute FID score if enabled
        if self.fid_loss is not None and all_outputs:
            print(f"  Computing FID score for {dataset_name}...")

            # Create a simple dataloader from collected outputs
            class OutputDataset(Dataset):
                def __init__(self, outputs):
                    self.outputs = torch.cat(outputs, dim=0)

                def __len__(self):
                    return len(self.outputs)

                def __getitem__(self, idx):
                    return self.outputs[idx]

            output_loader = DataLoader(OutputDataset(all_outputs), batch_size=32, shuffle=False)
            fid_score = self.fid_loss.compute_fid_score(output_loader)
            metrics['fid'] = fid_score
            print(f"  FID score: {fid_score:.2f}")

        # Save sample images
        if save_samples:
            sample_images = sample_images_ll + sample_images_hl
            if sample_images:
                epoch_str = f"epoch_{epoch:04d}" if epoch is not None else "eval"
                self.save_samples(sample_images, f"{dataset_name}_{epoch_str}")

        return metrics

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0
        epoch_psnr = 0
        epoch_fid = 0
        fid_count = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.args.epochs}")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # Compute loss
            loss, loss_dict = self.compute_loss(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

            self.optimizer.step()

            # Metrics
            with torch.no_grad():
                psnr = self.compute_psnr(outputs.clamp(0, 1), targets)

            epoch_loss += loss.item()
            epoch_psnr += psnr

            if loss_dict['fid'] > 0:
                epoch_fid += loss_dict['fid']
                fid_count += 1

            # Logging
            self.global_step += 1
            if self.global_step % self.args.log_interval == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/psnr', psnr, self.global_step)
                self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)
                for k, v in loss_dict.items():
                    if v > 0:  # Only log non-zero losses
                        self.writer.add_scalar(f'train/loss_{k}', v, self.global_step)

            postfix = {
                'loss': f'{loss.item():.4f}',
                'psnr': f'{psnr:.2f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            }
            if loss_dict['fid'] > 0:
                postfix['fid'] = f'{loss_dict["fid"]:.1f}'
            pbar.set_postfix(postfix)

        avg_loss = epoch_loss / len(self.train_loader)
        avg_psnr = epoch_psnr / len(self.train_loader)
        return avg_loss, avg_psnr

    @torch.no_grad()
    def validate(self, epoch):
        """Validate on validation set."""
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

            # Forward pass
            output = self.model(padded)

            # Remove padding
            output = output[:, :, :orig_h, :orig_w]
            output = output.clamp(0, 1)

            # Compute metrics
            psnr = self.compute_psnr(output, original)
            loss, _ = self.compute_loss(output, original, compute_fid=False)

            total_psnr += psnr
            total_loss += loss.item()
            count += 1

            if self.fid_loss is not None:
                all_outputs.append(output.cpu())

            # Save sample images
            if len(sample_images) < 4:
                sample_images.append({
                    'input': original[0].cpu(),
                    'output': output[0].cpu(),
                    'type': img_type[0]
                })

        avg_psnr = total_psnr / count
        avg_loss = total_loss / count

        # Log validation metrics
        if self.writer:
            self.writer.add_scalar('val/psnr', avg_psnr, epoch)
            self.writer.add_scalar('val/loss', avg_loss, epoch)

            # Compute and log FID if enabled
            if self.fid_loss is not None and all_outputs:
                class OutputDataset(Dataset):
                    def __init__(self, outputs):
                        self.outputs = torch.cat(outputs, dim=0)

                    def __len__(self):
                        return len(self.outputs)

                    def __getitem__(self, idx):
                        return self.outputs[idx]

                output_loader = DataLoader(OutputDataset(all_outputs), batch_size=32, shuffle=False)
                fid_score = self.fid_loss.compute_fid_score(output_loader)
                self.writer.add_scalar('val/fid', fid_score, epoch)
                print(f"  Validation FID: {fid_score:.2f}")

        # Save sample images
        if sample_images:
            self.save_samples(sample_images, f"val_epoch_{epoch:04d}")

        return avg_psnr, avg_loss

    def save_samples(self, samples, name):
        """Save sample reconstruction images."""
        n = len(samples)
        fig_tensors = []
        for s in samples:
            # Stack input and output horizontally
            diff = (s['input'] - s['output']).abs()
            row = torch.cat([s['input'], s['output'], diff * 5], dim=2)  # Amplify diff
            fig_tensors.append(row)

        grid = make_grid(fig_tensors, nrow=1, padding=2, normalize=False)
        save_path = os.path.join(self.samples_dir, f"{name}.png")
        save_image(grid, save_path)

        # Log to tensorboard
        if self.writer:
            self.writer.add_image(f'samples/{name}', grid, 0)

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            return

        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_psnr': self.best_psnr,
            'args': vars(self.args)
        }

        # Save latest
        latest_path = os.path.join(self.checkpoint_dir, "latest.pth")
        torch.save(checkpoint, latest_path)

        # Save periodic checkpoint
        if (epoch + 1) % self.args.save_every == 0:
            epoch_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch + 1:04d}.pth")
            torch.save(checkpoint, epoch_path)

        # Save best
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best.pth")
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, path, eval_only=False):
        """Load model checkpoint."""
        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if not eval_only:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.global_step = checkpoint['global_step']
            self.best_psnr = checkpoint.get('best_psnr', 0)
            print(f"Resumed from epoch {self.start_epoch}")
        else:
            print(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1} for evaluation")

    def run_initial_evaluation(self):
        """Run evaluation at the beginning of training on both train and val sets."""
        print(f"\n{'=' * 60}")
        print("Running initial evaluation before training...")
        print(f"{'=' * 60}\n")

        # Evaluate on training set
        print("Evaluating on training dataset...")
        train_metrics = self.evaluate_dataset(
            self.train_eval_loader,
            dataset_name="train",
            save_samples=True,
            epoch=-1  # Use -1 to indicate pre-training
        )
        print(f"  Training set - PSNR: {train_metrics['psnr']:.2f} dB, Loss: {train_metrics['loss']:.4f}")
        if 'psnr_ll' in train_metrics:
            print(f"    Low-light PSNR: {train_metrics['psnr_ll']:.2f} dB ({train_metrics['count_ll']} images)")
        if 'psnr_hl' in train_metrics:
            print(f"    High-light PSNR: {train_metrics['psnr_hl']:.2f} dB ({train_metrics['count_hl']} images)")
        if 'fid' in train_metrics:
            print(f"    FID: {train_metrics['fid']:.2f}")

        # Log to tensorboard
        if self.writer:
            self.writer.add_scalar('initial_eval/train_psnr', train_metrics['psnr'], 0)
            self.writer.add_scalar('initial_eval/train_loss', train_metrics['loss'], 0)
            if 'psnr_ll' in train_metrics:
                self.writer.add_scalar('initial_eval/train_psnr_ll', train_metrics['psnr_ll'], 0)
            if 'psnr_hl' in train_metrics:
                self.writer.add_scalar('initial_eval/train_psnr_hl', train_metrics['psnr_hl'], 0)
            if 'fid' in train_metrics:
                self.writer.add_scalar('initial_eval/train_fid', train_metrics['fid'], 0)

        # Evaluate on validation set
        if self.val_loader:
            print("\nEvaluating on validation dataset...")
            val_metrics = self.evaluate_dataset(
                self.val_loader,
                dataset_name="val",
                save_samples=True,
                epoch=-1
            )
            print(f"  Validation set - PSNR: {val_metrics['psnr']:.2f} dB, Loss: {val_metrics['loss']:.4f}")
            if 'psnr_ll' in val_metrics:
                print(f"    Low-light PSNR: {val_metrics['psnr_ll']:.2f} dB ({val_metrics['count_ll']} images)")
            if 'psnr_hl' in val_metrics:
                print(f"    High-light PSNR: {val_metrics['psnr_hl']:.2f} dB ({val_metrics['count_hl']} images)")
            if 'fid' in val_metrics:
                print(f"    FID: {val_metrics['fid']:.2f}")

            # Log to tensorboard
            if self.writer:
                self.writer.add_scalar('initial_eval/val_psnr', val_metrics['psnr'], 0)
                self.writer.add_scalar('initial_eval/val_loss', val_metrics['loss'], 0)
                if 'psnr_ll' in val_metrics:
                    self.writer.add_scalar('initial_eval/val_psnr_ll', val_metrics['psnr_ll'], 0)
                if 'psnr_hl' in val_metrics:
                    self.writer.add_scalar('initial_eval/val_psnr_hl', val_metrics['psnr_hl'], 0)
                if 'fid' in val_metrics:
                    self.writer.add_scalar('initial_eval/val_fid', val_metrics['fid'], 0)

        print(f"\n{'=' * 60}\n")

    def run_evaluation_only(self):
        """Run evaluation only on a saved checkpoint (no training)."""
        print(f"\n{'=' * 60}")
        print(f"Running evaluation on checkpoint")
        print(f"{'=' * 60}\n")

        results = {}

        # Evaluate on training set
        print("Evaluating on training dataset...")
        train_metrics = self.evaluate_dataset(
            self.train_eval_loader,
            dataset_name="train",
            save_samples=True,
            epoch=None
        )
        results['train'] = train_metrics
        print(f"\n  Training set results:")
        print(f"    Overall PSNR: {train_metrics['psnr']:.2f} dB")
        print(f"    Overall Loss: {train_metrics['loss']:.4f}")
        print(f"    Total images: {train_metrics['count']}")
        if 'psnr_ll' in train_metrics:
            print(f"    Low-light PSNR: {train_metrics['psnr_ll']:.2f} dB ({train_metrics['count_ll']} images)")
        if 'psnr_hl' in train_metrics:
            print(f"    High-light PSNR: {train_metrics['psnr_hl']:.2f} dB ({train_metrics['count_hl']} images)")
        if 'fid' in train_metrics:
            print(f"    FID: {train_metrics['fid']:.2f}")

        # Evaluate on validation set
        if self.val_loader:
            print("\nEvaluating on validation dataset...")
            val_metrics = self.evaluate_dataset(
                self.val_loader,
                dataset_name="val",
                save_samples=True,
                epoch=None
            )
            results['val'] = val_metrics
            print(f"\n  Validation set results:")
            print(f"    Overall PSNR: {val_metrics['psnr']:.2f} dB")
            print(f"    Overall Loss: {val_metrics['loss']:.4f}")
            print(f"    Total images: {val_metrics['count']}")
            if 'psnr_ll' in val_metrics:
                print(f"    Low-light PSNR: {val_metrics['psnr_ll']:.2f} dB ({val_metrics['count_ll']} images)")
            if 'psnr_hl' in val_metrics:
                print(f"    High-light PSNR: {val_metrics['psnr_hl']:.2f} dB ({val_metrics['count_hl']} images)")
            if 'fid' in val_metrics:
                print(f"    FID: {val_metrics['fid']:.2f}")

        # Save results to file
        results_path = os.path.join(self.output_dir, "evaluation_results.txt")
        with open(results_path, 'w') as f:
            f.write(f"Evaluation Results\n")
            f.write(f"==================\n")
            f.write(f"Checkpoint: {self.args.resume}\n")
            f.write(f"Mode: {self.args.mode}\n")
            if self.args.use_fid:
                f.write(f"FID Reference: {self.args.fid_ref_path}\n")
            f.write(f"\n")

            f.write(f"Training Set:\n")
            f.write(f"  PSNR: {train_metrics['psnr']:.2f} dB\n")
            f.write(f"  Loss: {train_metrics['loss']:.4f}\n")
            f.write(f"  Count: {train_metrics['count']}\n")
            if 'psnr_ll' in train_metrics:
                f.write(f"  Low-light PSNR: {train_metrics['psnr_ll']:.2f} dB ({train_metrics['count_ll']} images)\n")
            if 'psnr_hl' in train_metrics:
                f.write(f"  High-light PSNR: {train_metrics['psnr_hl']:.2f} dB ({train_metrics['count_hl']} images)\n")
            if 'fid' in train_metrics:
                f.write(f"  FID: {train_metrics['fid']:.2f}\n")

            if self.val_loader:
                f.write(f"\nValidation Set:\n")
                f.write(f"  PSNR: {val_metrics['psnr']:.2f} dB\n")
                f.write(f"  Loss: {val_metrics['loss']:.4f}\n")
                f.write(f"  Count: {val_metrics['count']}\n")
                if 'psnr_ll' in val_metrics:
                    f.write(f"  Low-light PSNR: {val_metrics['psnr_ll']:.2f} dB ({val_metrics['count_ll']} images)\n")
                if 'psnr_hl' in val_metrics:
                    f.write(f"  High-light PSNR: {val_metrics['psnr_hl']:.2f} dB ({val_metrics['count_hl']} images)\n")
                if 'fid' in val_metrics:
                    f.write(f"  FID: {val_metrics['fid']:.2f}\n")

        print(f"\n{'=' * 60}")
        print(f"Evaluation complete!")
        print(f"Results saved to: {results_path}")
        print(f"Samples saved to: {self.samples_dir}")
        print(f"{'=' * 60}\n")

        return results

    def train(self):
        """Main training loop."""
        print(f"\n{'=' * 60}")
        print(f"Starting training: {self.run_name}")
        print(f"{'=' * 60}")
        print(f"Output directory: {self.output_dir}")
        print(f"Training samples: {len(self.train_dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_dataset)}")
        if self.fid_loss is not None:
            print(f"FID reference: {self.args.fid_ref_path}")
        print(f"{'=' * 60}\n")

        # Run initial evaluation before training
        self.run_initial_evaluation()

        for epoch in range(self.start_epoch, self.args.epochs):
            # Train
            train_loss, train_psnr = self.train_epoch(epoch)
            print(
                f"Epoch {epoch + 1}/{self.args.epochs} - Train Loss: {train_loss:.4f}, Train PSNR: {train_psnr:.2f} dB")

            # Validate
            if self.val_loader and (epoch + 1) % self.args.val_every == 0:
                val_psnr, val_loss = self.validate(epoch)
                print(f"  Validation - Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f} dB")

                # Check for best model
                is_best = val_psnr > self.best_psnr
                if is_best:
                    self.best_psnr = val_psnr
                    print(f"  New best PSNR: {val_psnr:.2f} dB")

                self.save_checkpoint(epoch, is_best)
            else:
                self.save_checkpoint(epoch)

            # Update learning rate
            self.scheduler.step()

        # Final save
        self.save_checkpoint(self.args.epochs - 1)
        print(f"\nTraining complete! Best PSNR: {self.best_psnr:.2f} dB")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")

        if self.writer:
            self.writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Phase P1: Fine-tune TAESD on LOLv1')

    # Data
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to training data (e.g., lolv1/our485)')
    parser.add_argument('--val_path', type=str, default=None,
                        help='Path to validation data (e.g., lolv1/eval15)')
    parser.add_argument('--mode', type=str, default='both', choices=['ll', 'hl', 'both'],
                        help='Training mode: ll (low-light), hl (high-light), both')

    # Model
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Evaluation only mode
    parser.add_argument('--eval_only', action='store_true',
                        help='Run evaluation only on a saved checkpoint (requires --resume)')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='Training patch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping value (0 to disable)')

    # Loss weights
    parser.add_argument('--l1_weight', type=float, default=1.0,
                        help='Weight for L1 loss')
    parser.add_argument('--l2_weight', type=float, default=0.0,
                        help='Weight for L2/MSE loss')
    parser.add_argument('--use_lpips', action='store_true',
                        help='Use LPIPS perceptual loss')
    parser.add_argument('--lpips_weight', type=float, default=0.1,
                        help='Weight for LPIPS loss')

    # FID loss options
    parser.add_argument('--use_fid', action='store_true',
                        help='Use FID loss against a reference dataset')
    parser.add_argument('--fid_ref_path', type=str, default=None,
                        help='Path to reference dataset for FID (e.g., lolv2-real/high)')
    parser.add_argument('--fid_weight', type=float, default=0.01,
                        help='Weight for FID loss (typically small, e.g., 0.001-0.1)')
    parser.add_argument('--fid_max_ref_images', type=int, default=1000,
                        help='Maximum number of reference images for FID statistics')
    parser.add_argument('--fid_mode', type=str, default='batch', choices=['batch', 'running'],
                        help='FID computation mode: batch (per-batch) or running (accumulated)')
    parser.add_argument('--fid_momentum', type=float, default=0.1,
                        help='Momentum for running FID statistics')
    parser.add_argument('--fid_every', type=int, default=10,
                        help='Compute FID loss every N training steps (to save computation)')

    # Misc
    parser.add_argument('--output_dir', type=str, default='./out_p1',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cuda, mps, cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log every N steps')
    parser.add_argument('--val_every', type=int, default=1,
                        help='Validate every N epochs')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate arguments
    if args.eval_only and args.resume is None:
        raise ValueError("--eval_only requires --resume to specify a checkpoint")

    if args.use_fid and args.fid_ref_path is None:
        raise ValueError("--use_fid requires --fid_ref_path to specify reference dataset")

    trainer = TAESDTrainer(args)

    if args.eval_only:
        trainer.run_evaluation_only()
    else:
        trainer.train()


if __name__ == '__main__':
    main()