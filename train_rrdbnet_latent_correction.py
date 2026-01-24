#!/usr/bin/env python3
"""
Phase P2: Train RRDB Latent Correction Network for Low-Light Image Enhancement.

This script trains the RRDB (Residual-in-Residual Dense Block) network to correct
latent representations from a low-light trained encoder to match latent representations
from a high-light trained encoder.

Architecture:
    LL Image → Encoder_LL → z_ll → RRDB → z_corrected → Decoder_HL → Enhanced Image

The RRDB learns to transform latents from the LL domain to the HL domain.

Training Objectives:
1. Latent Loss: ||RRDB(z_ll) - z_hl||  (MSE, MAE, or KL divergence)
2. Image Loss: ||Decoder_HL(RRDB(z_ll)) - HL_image||  (MSE, MAE, LPIPS)
3. FID Loss: Match feature distribution to reference dataset

Usage:
    # Basic training with latent loss only
    python train_rrdb_latent_correction.py \\
        --data_path /path/to/lolv1/our485 \\
        --ll_checkpoint /path/to/taesd_ll.pth \\
        --hl_checkpoint /path/to/taesd_hl.pth \\
        --config configs/rrdb_config.yaml

    # Training with image reconstruction loss
    python train_rrdb_latent_correction.py \\
        --data_path /path/to/lolv1/our485 \\
        --ll_checkpoint /path/to/taesd_ll.pth \\
        --hl_checkpoint /path/to/taesd_hl.pth \\
        --use_image_loss --image_loss_weight 1.0

    # Training with FID loss
    python train_rrdb_latent_correction.py \\
        --data_path /path/to/lolv1/our485 \\
        --ll_checkpoint /path/to/taesd_ll.pth \\
        --hl_checkpoint /path/to/taesd_hl.pth \\
        --use_fid --fid_ref_path /path/to/lolv2-real/high
"""

import argparse
import os
import yaml
from datetime import datetime
from collections import OrderedDict

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
import numpy as np

from ELIR.models.load_model import get_model
# Import TAESD and RRDB models
from ELIR.models.taesd import TAESD

# Try to import RRDB - handle different possible locations
try:
    from ELIR.models.rrdb import RRDBNet
except ImportError:
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
    except ImportError:
        # Define RRDB inline if not available
        RRDBNet = None

# Optional: LPIPS
try:
    import lpips

    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not available. Install with 'pip install lpips'")

# Optional: FID
try:
    from torchvision.models import inception_v3, Inception_V3_Weights
    from scipy import linalg

    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    print("Warning: scipy not available for FID. Install with 'pip install scipy'")


# ============================================================================
# RRDB Network Definition (fallback if not importable)
# ============================================================================

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block for RRDB."""

    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block."""

    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNetLatent(nn.Module):
    """
    RRDB Network for latent space correction.

    This network operates in the latent space of TAESD (4 channels).
    """

    def __init__(self, in_channels=4, out_channels=4, num_feat=64, num_block=6,
                 num_grow_ch=32, scale=1):
        super().__init__()
        self.scale = scale

        # First convolution
        self.conv_first = nn.Conv2d(in_channels, num_feat, 3, 1, 1)

        # RRDB blocks
        self.body = nn.Sequential(*[
            RRDB(num_feat, num_grow_ch) for _ in range(num_block)
        ])

        # After body convolution
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Output convolution
        self.conv_last = nn.Conv2d(num_feat, out_channels, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        out = self.conv_last(self.lrelu(feat))
        # Residual connection in latent space
        return out + x


# ============================================================================
# FID Components
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
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, x):
        x = self.resize(x)
        x = self.normalize(x)
        with torch.no_grad():
            features = self.inception(x)
        return features


class FIDLoss(nn.Module):
    """FID-based loss for training."""

    def __init__(self, ref_dataset_path, device='cuda', max_ref_images=1000):
        super().__init__()
        self.device = device
        self.feature_extractor = FIDFeatureExtractor(device)

        print(f"Computing reference FID statistics from {ref_dataset_path}...")
        ref_dataset = FIDReferenceDataset(ref_dataset_path, max_images=max_ref_images)
        ref_loader = DataLoader(ref_dataset, batch_size=32, shuffle=False, num_workers=4)

        self.ref_mu, self.ref_sigma = self._compute_statistics(ref_loader)
        print(f"Reference statistics computed.")

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

        return torch.from_numpy(mu).float().to(self.device), \
            torch.from_numpy(sigma).float().to(self.device)

    def _compute_batch_statistics(self, features):
        mu = features.mean(dim=0)
        centered = features - mu.unsqueeze(0)
        sigma = (centered.T @ centered) / (features.shape[0] - 1) + 1e-6 * torch.eye(
            features.shape[1], device=features.device)
        return mu, sigma

    def _compute_fid(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        diff = mu1 - mu2
        product = sigma1 @ sigma2 + eps * torch.eye(sigma1.shape[0], device=sigma1.device)
        eigenvalues, eigenvectors = torch.linalg.eigh(product)
        eigenvalues = torch.clamp(eigenvalues, min=eps)
        sqrt_product = eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T
        fid = torch.sum(diff ** 2) + torch.trace(sigma1 + sigma2 - 2 * sqrt_product)
        return fid

    def forward(self, generated_images):
        features = self.feature_extractor(generated_images)
        gen_mu, gen_sigma = self._compute_batch_statistics(features)
        fid = self._compute_fid(gen_mu, gen_sigma, self.ref_mu, self.ref_sigma)
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
        print(f"[FIDReferenceDataset] Loaded {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img)


# ============================================================================
# Dataset
# ============================================================================

class LOLv1PairedDataset(Dataset):
    """
    Dataset for paired LL-HL training.
    Returns low-light images and corresponding high-light images.
    """

    def __init__(self, image_folder, patch_size=256, augment=True):
        super().__init__()
        self.image_folder = image_folder
        self.patch_size = patch_size
        self.augment = augment

        lq_dir = os.path.join(image_folder, "low")
        hq_dir = os.path.join(image_folder, "high")

        self.lq_paths = sorted(glob.glob(os.path.join(lq_dir, "*.png")))
        self.hq_paths = sorted(glob.glob(os.path.join(hq_dir, "*.png")))

        assert len(self.lq_paths) == len(self.hq_paths), \
            f"Mismatch: {len(self.lq_paths)} LL vs {len(self.hq_paths)} HL images"

        self.transform = transforms.ToTensor()
        print(f"[LOLv1PairedDataset] Loaded {len(self.lq_paths)} pairs")

    def __len__(self):
        return len(self.lq_paths)

    def _load_pair(self, ll_path, hl_path):
        ll_img = Image.open(ll_path).convert('RGB')
        hl_img = Image.open(hl_path).convert('RGB')

        assert ll_img.size == hl_img.size, "LL and HL images must have same size"
        w, h = ll_img.size

        # Random crop (same location for both)
        if self.patch_size > 0 and (w > self.patch_size or h > self.patch_size):
            left = torch.randint(0, max(1, w - self.patch_size), (1,)).item()
            top = torch.randint(0, max(1, h - self.patch_size), (1,)).item()
            ll_img = ll_img.crop((left, top, left + self.patch_size, top + self.patch_size))
            hl_img = hl_img.crop((left, top, left + self.patch_size, top + self.patch_size))

        ll_tensor = self.transform(ll_img)
        hl_tensor = self.transform(hl_img)

        # Random augmentation (same for both)
        if self.augment:
            if torch.rand(1) < 0.5:
                ll_tensor = torch.flip(ll_tensor, dims=[2])
                hl_tensor = torch.flip(hl_tensor, dims=[2])
            if torch.rand(1) < 0.5:
                ll_tensor = torch.flip(ll_tensor, dims=[1])
                hl_tensor = torch.flip(hl_tensor, dims=[1])
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                ll_tensor = torch.rot90(ll_tensor, k, dims=[1, 2])
                hl_tensor = torch.rot90(hl_tensor, k, dims=[1, 2])

        return ll_tensor, hl_tensor

    def __getitem__(self, idx):
        return self._load_pair(self.lq_paths[idx], self.hq_paths[idx])


class LOLv1ValDataset(Dataset):
    """Validation dataset - returns full images with padding."""

    def __init__(self, image_folder, pad_multiple=8):
        super().__init__()
        self.image_folder = image_folder
        self.pad_multiple = pad_multiple

        lq_dir = os.path.join(image_folder, "low")
        hq_dir = os.path.join(image_folder, "high")

        self.lq_paths = sorted(glob.glob(os.path.join(lq_dir, "*.png")))
        self.hq_paths = sorted(glob.glob(os.path.join(hq_dir, "*.png")))

        self.transform = transforms.ToTensor()
        print(f"[LOLv1ValDataset] Loaded {len(self.lq_paths)} pairs")

    def __len__(self):
        return len(self.lq_paths)

    def _pad_to_multiple(self, tensor):
        _, h, w = tensor.shape
        pad_h = (self.pad_multiple - h % self.pad_multiple) % self.pad_multiple
        pad_w = (self.pad_multiple - w % self.pad_multiple) % self.pad_multiple
        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor.unsqueeze(0), (0, pad_w, 0, pad_h), mode='reflect').squeeze(0)
        return tensor, h, w

    def __getitem__(self, idx):
        ll_img = Image.open(self.lq_paths[idx]).convert('RGB')
        hl_img = Image.open(self.hq_paths[idx]).convert('RGB')

        ll_tensor = self.transform(ll_img)
        hl_tensor = self.transform(hl_img)

        ll_padded, orig_h, orig_w = self._pad_to_multiple(ll_tensor)
        hl_padded, _, _ = self._pad_to_multiple(hl_tensor)

        return ll_padded, hl_padded, ll_tensor, hl_tensor, orig_h, orig_w, \
            os.path.basename(self.lq_paths[idx])


# ============================================================================
# Trainer
# ============================================================================

class RRDBLatentTrainer:
    """Trainer for RRDB latent correction network."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if args.device != 'auto' else
                                   ('cuda' if torch.cuda.is_available() else
                                    ('mps' if torch.backends.mps.is_available() else 'cpu')))

        print(f"Using device: {self.device}")

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"rrdb_latent_{timestamp}"
        self.output_dir = os.path.join(args.output_dir, self.run_name)
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.samples_dir = os.path.join(self.output_dir, "samples")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)

        # Load config if provided
        self.config = {}
        if args.config and os.path.exists(args.config):
            with open(args.config, 'r') as f:
                self.config = yaml.safe_load(f)
            print(f"Loaded config from {args.config}")

        # Initialize TAESD models (frozen)
        self._init_taesd_models()

        # Initialize RRDB network (trainable)
        self._init_rrdb_model()

        # Loss functions
        self._init_losses()

        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.rrdb.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01
        )

        # Datasets and dataloaders
        self.train_dataset = LOLv1PairedDataset(
            args.data_path,
            patch_size=args.patch_size,
            augment=True
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )

        if args.val_path:
            self.val_dataset = LOLv1ValDataset(args.val_path)
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers
            )
        else:
            self.val_loader = None

        # Tensorboard
        self.writer = SummaryWriter(os.path.join(self.output_dir, "logs"))

        # Training state
        self.start_epoch = 0
        self.global_step = 0
        self.best_psnr = 0

        # Resume from checkpoint
        if args.resume:
            self.load_checkpoint(args.resume)

    def _init_taesd_models(self):
        """Initialize and load TAESD encoder/decoder models."""
        # Load LL encoder
        print(f"Loading LL TAESD from {self.args.ll_checkpoint}")
        self.taesd_ll = TAESD(pretrained=False).to(self.device)
        ll_ckpt = torch.load(self.args.ll_checkpoint, map_location=self.device)
        if 'model_state_dict' in ll_ckpt:
            self.taesd_ll.load_state_dict(ll_ckpt['model_state_dict'])
        else:
            self.taesd_ll.load_state_dict(ll_ckpt)
        self.taesd_ll.eval()
        for p in self.taesd_ll.parameters():
            p.requires_grad = False

        # Load HL encoder/decoder
        print(f"Loading HL TAESD from {self.args.hl_checkpoint}")
        self.taesd_hl = TAESD(pretrained=False).to(self.device)
        hl_ckpt = torch.load(self.args.hl_checkpoint, map_location=self.device)
        if 'model_state_dict' in hl_ckpt:
            self.taesd_hl.load_state_dict(hl_ckpt['model_state_dict'])
        else:
            self.taesd_hl.load_state_dict(hl_ckpt)
        self.taesd_hl.eval()
        for p in self.taesd_hl.parameters():
            p.requires_grad = False

        print("TAESD models loaded and frozen")

    def _init_rrdb_model(self):
        """Initialize RRDB network for latent correction."""
        # Get config from yaml or args
        self.rrdb = get_model(self.config).to(self.device)
        if self.config.get("trainable"):
            for p in self.rrdb.parameters():
                p.requires_grad_(True)

        # Count parameters
        total_params = sum(p.numel() for p in self.rrdb.parameters())
        print(f"RRDB network initialized: {total_params:,} parameters")

    def _init_losses(self):
        """Initialize loss functions."""
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        # LPIPS for image loss
        self.lpips_loss = None
        if self.args.use_lpips and LPIPS_AVAILABLE:
            self.lpips_loss = lpips.LPIPS(net='vgg').to(self.device)
            self.lpips_loss.eval()
            for p in self.lpips_loss.parameters():
                p.requires_grad = False
            print("LPIPS loss enabled")

        # FID loss
        self.fid_loss = None
        if self.args.use_fid:
            if not FID_AVAILABLE:
                raise RuntimeError("FID loss requires scipy")
            if not self.args.fid_ref_path:
                raise ValueError("FID loss requires --fid_ref_path")
            self.fid_loss = FIDLoss(
                ref_dataset_path=self.args.fid_ref_path,
                device=self.device,
                max_ref_images=self.args.fid_max_ref_images
            )
            print(f"FID loss enabled with reference: {self.args.fid_ref_path}")

    def compute_latent_loss(self, z_corrected, z_hl_target):
        """
        Compute latent space loss between corrected and target latents.

        Supports MSE, MAE, and KL divergence.
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=self.device)

        # MSE loss
        if self.args.latent_mse_weight > 0:
            loss_mse = self.mse_loss(z_corrected, z_hl_target)
            total_loss = total_loss + self.args.latent_mse_weight * loss_mse
            loss_dict['latent_mse'] = loss_mse.item()

        # MAE/L1 loss
        if self.args.latent_mae_weight > 0:
            loss_mae = self.l1_loss(z_corrected, z_hl_target)
            total_loss = total_loss + self.args.latent_mae_weight * loss_mae
            loss_dict['latent_mae'] = loss_mae.item()

        # KL divergence (treating latents as distributions)
        if self.args.latent_kl_weight > 0:
            # Compute KL divergence assuming Gaussian with mean=latent, var=1
            # KL(N(mu1, 1) || N(mu2, 1)) = 0.5 * (mu1 - mu2)^2
            # This is equivalent to MSE / 2, but we can also compute proper KL
            # by treating channels as independent Gaussians

            # Simple approximation: treat latent values as means
            loss_kl = 0.5 * torch.mean((z_corrected - z_hl_target) ** 2)
            total_loss = total_loss + self.args.latent_kl_weight * loss_kl
            loss_dict['latent_kl'] = loss_kl.item()

        return total_loss, loss_dict

    def compute_image_loss(self, recon_image, target_image):
        """
        Compute image space loss between reconstructed and target images.
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=self.device)

        # MSE loss
        if self.args.image_mse_weight > 0:
            loss_mse = self.mse_loss(recon_image, target_image)
            total_loss = total_loss + self.args.image_mse_weight * loss_mse
            loss_dict['image_mse'] = loss_mse.item()

        # MAE/L1 loss
        if self.args.image_mae_weight > 0:
            loss_mae = self.l1_loss(recon_image, target_image)
            total_loss = total_loss + self.args.image_mae_weight * loss_mae
            loss_dict['image_mae'] = loss_mae.item()

        # LPIPS perceptual loss
        if self.lpips_loss is not None and self.args.image_lpips_weight > 0:
            pred_lpips = recon_image * 2 - 1
            target_lpips = target_image * 2 - 1
            loss_lpips = self.lpips_loss(pred_lpips, target_lpips).mean()
            total_loss = total_loss + self.args.image_lpips_weight * loss_lpips
            loss_dict['image_lpips'] = loss_lpips.item()

        return total_loss, loss_dict

    def compute_psnr(self, pred, target):
        """Compute PSNR between prediction and target."""
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return float('inf')
        return 10 * torch.log10(1.0 / mse).item()

    @torch.no_grad()
    def compute_lpips_score(self, pred, target):
        """Compute LPIPS score."""
        if self.lpips_loss is None:
            return 0.0
        pred_lpips = pred * 2 - 1
        target_lpips = target * 2 - 1
        return self.lpips_loss(pred_lpips, target_lpips).mean().item()

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.rrdb.train()
        epoch_loss = 0
        epoch_psnr = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.args.epochs}")
        for batch_idx, (ll_images, hl_images) in enumerate(pbar):
            ll_images = ll_images.to(self.device)
            hl_images = hl_images.to(self.device)

            self.optimizer.zero_grad()

            # Get latents from frozen encoders
            with torch.no_grad():
                z_ll = self.taesd_ll.encoder(ll_images)
                z_hl = self.taesd_hl.encoder(hl_images)

            # Correct latent with RRDB
            z_corrected = self.rrdb(z_ll)

            # Compute latent loss
            latent_loss, latent_loss_dict = self.compute_latent_loss(z_corrected, z_hl)
            total_loss = latent_loss

            # Compute image loss if enabled
            image_loss_dict = {}
            if self.args.use_image_loss:
                with torch.no_grad():
                    # Decode corrected latent through HL decoder
                    recon_image = self.taesd_hl.decoder(z_corrected)
                    recon_image = recon_image.clamp(0, 1)

                # For image loss, we need gradients through decoder
                # But decoder is frozen, so we use straight-through estimator
                # or just compute loss for monitoring
                recon_image_grad = self.taesd_hl.decoder(z_corrected)
                image_loss, image_loss_dict = self.compute_image_loss(recon_image_grad, hl_images)
                total_loss = total_loss + self.args.image_loss_weight * image_loss

            # FID loss
            fid_loss_val = 0.0
            if self.fid_loss is not None and self.global_step % self.args.fid_every == 0:
                with torch.no_grad():
                    recon_image = self.taesd_hl.decoder(z_corrected).clamp(0, 1)
                fid_loss_val = self.fid_loss(recon_image)
                total_loss = total_loss + self.args.fid_weight * fid_loss_val

            # Backward pass
            total_loss.backward()

            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.rrdb.parameters(), self.args.grad_clip)

            self.optimizer.step()

            # Compute metrics
            with torch.no_grad():
                recon_image = self.taesd_hl.decoder(z_corrected).clamp(0, 1)
                psnr = self.compute_psnr(recon_image, hl_images)

            epoch_loss += total_loss.item()
            epoch_psnr += psnr

            # Logging
            self.global_step += 1
            if self.global_step % self.args.log_interval == 0:
                self.writer.add_scalar('train/loss', total_loss.item(), self.global_step)
                self.writer.add_scalar('train/psnr', psnr, self.global_step)
                self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)

                for k, v in latent_loss_dict.items():
                    self.writer.add_scalar(f'train/{k}', v, self.global_step)
                for k, v in image_loss_dict.items():
                    self.writer.add_scalar(f'train/{k}', v, self.global_step)

                if fid_loss_val > 0:
                    self.writer.add_scalar('train/fid_loss', fid_loss_val, self.global_step)

            postfix = {
                'loss': f'{total_loss.item():.4f}',
                'psnr': f'{psnr:.2f}',
            }
            pbar.set_postfix(postfix)

        avg_loss = epoch_loss / len(self.train_loader)
        avg_psnr = epoch_psnr / len(self.train_loader)
        return avg_loss, avg_psnr

    @torch.no_grad()
    def validate(self, epoch):
        """Validate and save sample images."""
        if self.val_loader is None:
            return 0, 0, 0

        self.rrdb.eval()

        total_psnr_before = 0
        total_psnr_after = 0
        total_lpips_before = 0
        total_lpips_after = 0
        total_loss = 0
        count = 0

        sample_images = []
        all_outputs_before = []
        all_outputs_after = []

        for batch_idx, (ll_padded, hl_padded, ll_orig, hl_orig, orig_h, orig_w, filename) in enumerate(self.val_loader):
            ll_padded = ll_padded.to(self.device)
            hl_padded = hl_padded.to(self.device)
            hl_orig = hl_orig.to(self.device)

            # Get latents
            z_ll = self.taesd_ll.encoder(ll_padded)
            z_hl = self.taesd_hl.encoder(hl_padded)

            # Reconstruct WITHOUT correction (baseline)
            recon_before = self.taesd_hl.decoder(z_ll)
            recon_before = recon_before[:, :, :orig_h, :orig_w].clamp(0, 1)

            # Correct latent and reconstruct
            z_corrected = self.rrdb(z_ll)
            recon_after = self.taesd_hl.decoder(z_corrected)
            recon_after = recon_after[:, :, :orig_h, :orig_w].clamp(0, 1)

            # Get target
            target = hl_orig[:, :, :orig_h, :orig_w]

            # Compute metrics
            psnr_before = self.compute_psnr(recon_before, target)
            psnr_after = self.compute_psnr(recon_after, target)
            lpips_before = self.compute_lpips_score(recon_before, target)
            lpips_after = self.compute_lpips_score(recon_after, target)

            total_psnr_before += psnr_before
            total_psnr_after += psnr_after
            total_lpips_before += lpips_before
            total_lpips_after += lpips_after
            count += 1

            # Collect for FID
            all_outputs_before.append(recon_before.cpu())
            all_outputs_after.append(recon_after.cpu())

            # Save sample images
            if len(sample_images) < 4:
                ll_display = ll_padded[:, :, :orig_h, :orig_w].cpu()
                sample_images.append({
                    'll_input': ll_display[0],
                    'hl_target': target[0].cpu(),
                    'recon_before': recon_before[0].cpu(),
                    'recon_after': recon_after[0].cpu(),
                    'filename': filename[0]
                })

        # Compute averages
        avg_psnr_before = total_psnr_before / count
        avg_psnr_after = total_psnr_after / count
        avg_lpips_before = total_lpips_before / count
        avg_lpips_after = total_lpips_after / count

        # Log metrics
        self.writer.add_scalar('val/psnr_before', avg_psnr_before, epoch)
        self.writer.add_scalar('val/psnr_after', avg_psnr_after, epoch)
        self.writer.add_scalar('val/psnr_improvement', avg_psnr_after - avg_psnr_before, epoch)
        self.writer.add_scalar('val/lpips_before', avg_lpips_before, epoch)
        self.writer.add_scalar('val/lpips_after', avg_lpips_after, epoch)

        # Compute FID if available
        fid_before = None
        fid_after = None
        if self.fid_loss is not None:
            class OutputDataset(Dataset):
                def __init__(self, outputs):
                    self.outputs = torch.cat(outputs, dim=0)

                def __len__(self):
                    return len(self.outputs)

                def __getitem__(self, idx):
                    return self.outputs[idx]

            print("  Computing FID scores...")
            loader_before = DataLoader(OutputDataset(all_outputs_before), batch_size=32, shuffle=False)
            loader_after = DataLoader(OutputDataset(all_outputs_after), batch_size=32, shuffle=False)

            fid_before = self.fid_loss.compute_fid_score(loader_before)
            fid_after = self.fid_loss.compute_fid_score(loader_after)

            self.writer.add_scalar('val/fid_before', fid_before, epoch)
            self.writer.add_scalar('val/fid_after', fid_after, epoch)

        # Save sample images
        if sample_images:
            self.save_samples(sample_images, f"val_epoch_{epoch:04d}")

        return avg_psnr_before, avg_psnr_after, avg_lpips_before, avg_lpips_after, fid_before, fid_after

    def save_samples(self, samples, name):
        """Save sample images showing before/after correction."""
        fig_tensors = []
        for s in samples:
            # Stack: LL input | Before correction | After correction | HL target | Diff
            diff_before = (s['recon_before'] - s['hl_target']).abs()
            diff_after = (s['recon_after'] - s['hl_target']).abs()

            row = torch.cat([
                s['ll_input'],
                s['recon_before'],
                s['recon_after'],
                s['hl_target'],
                diff_before * 5,
                diff_after * 5
            ], dim=2)
            fig_tensors.append(row)

        grid = make_grid(fig_tensors, nrow=1, padding=2, normalize=False)
        save_path = os.path.join(self.samples_dir, f"{name}.png")
        save_image(grid, save_path)

        # Also save individual comparison
        for i, s in enumerate(samples[:2]):  # Save first 2 as separate files
            comparison = torch.stack([
                s['ll_input'],
                s['recon_before'],
                s['recon_after'],
                s['hl_target']
            ])
            comp_grid = make_grid(comparison, nrow=4, padding=2)
            save_image(comp_grid, os.path.join(self.samples_dir, f"{name}_sample_{i}.png"))

        self.writer.add_image(f'samples/{name}', grid, 0)

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'rrdb_state_dict': self.rrdb.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_psnr': self.best_psnr,
            'args': vars(self.args),
            'config': self.config
        }

        latest_path = os.path.join(self.checkpoint_dir, "latest.pth")
        torch.save(checkpoint, latest_path)

        if (epoch + 1) % self.args.save_every == 0:
            epoch_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch + 1:04d}.pth")
            torch.save(checkpoint, epoch_path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best.pth")
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, path):
        """Load model checkpoint."""
        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.rrdb.load_state_dict(checkpoint['rrdb_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_psnr = checkpoint.get('best_psnr', 0)
        print(f"Resumed from epoch {self.start_epoch}")

    def train(self):
        """Main training loop."""
        print(f"\n{'=' * 70}")
        print(f"Starting training: {self.run_name}")
        print(f"{'=' * 70}")
        print(f"Output directory: {self.output_dir}")
        print(f"Training samples: {len(self.train_dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_dataset)}")
        print(f"\nLoss configuration:")
        print(f"  Latent MSE weight: {self.args.latent_mse_weight}")
        print(f"  Latent MAE weight: {self.args.latent_mae_weight}")
        print(f"  Latent KL weight: {self.args.latent_kl_weight}")
        if self.args.use_image_loss:
            print(f"  Image loss weight: {self.args.image_loss_weight}")
            print(f"    Image MSE weight: {self.args.image_mse_weight}")
            print(f"    Image MAE weight: {self.args.image_mae_weight}")
            print(f"    Image LPIPS weight: {self.args.image_lpips_weight}")
        if self.fid_loss:
            print(f"  FID loss weight: {self.args.fid_weight}")
        print(f"{'=' * 70}\n")

        # Initial validation
        if self.val_loader:
            print("Running initial validation...")
            results = self.validate(-1)
            psnr_before, psnr_after, lpips_before, lpips_after, fid_before, fid_after = results
            print(f"  PSNR - Before: {psnr_before:.2f} dB, After: {psnr_after:.2f} dB")
            print(f"  LPIPS - Before: {lpips_before:.4f}, After: {lpips_after:.4f}")
            if fid_before is not None:
                print(f"  FID - Before: {fid_before:.2f}, After: {fid_after:.2f}")
            print()

        for epoch in range(self.start_epoch, self.args.epochs):
            # Train
            train_loss, train_psnr = self.train_epoch(epoch)
            print(f"Epoch {epoch + 1}/{self.args.epochs} - Train Loss: {train_loss:.4f}, PSNR: {train_psnr:.2f} dB")

            # Validate every epoch by default
            if self.val_loader and (epoch + 1) % self.args.val_every == 0:
                results = self.validate(epoch)
                psnr_before, psnr_after, lpips_before, lpips_after, fid_before, fid_after = results

                print(f"  Validation:")
                print(
                    f"    PSNR - Before: {psnr_before:.2f} dB, After: {psnr_after:.2f} dB (Δ={psnr_after - psnr_before:+.2f})")
                print(
                    f"    LPIPS - Before: {lpips_before:.4f}, After: {lpips_after:.4f} (Δ={lpips_after - lpips_before:+.4f})")
                if fid_before is not None:
                    print(
                        f"    FID - Before: {fid_before:.2f}, After: {fid_after:.2f} (Δ={fid_after - fid_before:+.2f})")

                is_best = psnr_after > self.best_psnr
                if is_best:
                    self.best_psnr = psnr_after
                    print(f"    New best PSNR: {psnr_after:.2f} dB")

                self.save_checkpoint(epoch, is_best)
            else:
                self.save_checkpoint(epoch)

            self.scheduler.step()

        self.save_checkpoint(self.args.epochs - 1)
        print(f"\nTraining complete! Best PSNR: {self.best_psnr:.2f} dB")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")

        self.writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Train RRDB Latent Correction Network')

    # Data
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to training data (e.g., lolv1/our485)')
    parser.add_argument('--val_path', type=str, default=None,
                        help='Path to validation data (e.g., lolv1/eval15)')

    # TAESD checkpoints
    parser.add_argument('--ll_checkpoint', type=str, required=True,
                        help='Path to TAESD checkpoint trained on low-light images')
    parser.add_argument('--hl_checkpoint', type=str, required=True,
                        help='Path to TAESD checkpoint trained on high-light images')

    # Config
    parser.add_argument('--config', type=str, default='configs/rrdbnet_training/rrdbnet_lolv1_from_denoising.yaml',
                        help='Path to YAML config file with model architecture settings')

    # RRDB architecture
    parser.add_argument('--rrdb_num_feat', type=int, default=64,
                        help='Number of features in RRDB')
    parser.add_argument('--rrdb_num_block', type=int, default=6,
                        help='Number of RRDB blocks')
    parser.add_argument('--rrdb_num_grow_ch', type=int, default=32,
                        help='Growth channels in RRDB')

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
                        help='Gradient clipping value')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Latent loss weights
    parser.add_argument('--latent_mse_weight', type=float, default=1.0,
                        help='Weight for latent MSE loss')
    parser.add_argument('--latent_mae_weight', type=float, default=0.0,
                        help='Weight for latent MAE/L1 loss')
    parser.add_argument('--latent_kl_weight', type=float, default=0.0,
                        help='Weight for latent KL divergence loss')

    # Image loss
    parser.add_argument('--use_image_loss', action='store_true',
                        help='Use image reconstruction loss')
    parser.add_argument('--image_loss_weight', type=float, default=1.0,
                        help='Overall weight for image loss')
    parser.add_argument('--image_mse_weight', type=float, default=0.0,
                        help='Weight for image MSE loss')
    parser.add_argument('--image_mae_weight', type=float, default=1.0,
                        help='Weight for image MAE/L1 loss')
    parser.add_argument('--use_lpips', action='store_true',
                        help='Use LPIPS perceptual loss')
    parser.add_argument('--image_lpips_weight', type=float, default=0.1,
                        help='Weight for image LPIPS loss')

    # FID loss
    parser.add_argument('--use_fid', action='store_true',
                        help='Use FID loss')
    parser.add_argument('--fid_ref_path', type=str, default=None,
                        help='Path to reference dataset for FID')
    parser.add_argument('--fid_weight', type=float, default=0.01,
                        help='Weight for FID loss')
    parser.add_argument('--fid_max_ref_images', type=int, default=1000,
                        help='Maximum reference images for FID')
    parser.add_argument('--fid_every', type=int, default=10,
                        help='Compute FID loss every N steps')

    # Misc
    parser.add_argument('--output_dir', type=str, default='./out_rrdb',
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
    if args.use_fid and args.fid_ref_path is None:
        raise ValueError("--use_fid requires --fid_ref_path")

    trainer = RRDBLatentTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()