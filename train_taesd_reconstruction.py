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

    # Train with KL divergence loss against the original encoder distribution
    python train_taesd_reconstruction.py --data_path /path/to/lolv1/our485 --use_kl --kl_weight 0.01
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

# KL divergence is always available (no external dependencies beyond torch)
import copy


class LatentKLLoss(nn.Module):
    """
    KL Divergence loss computed between the trained encoder's latent distribution
    and the original (pretrained) encoder's latent distribution.

    This encourages the fine-tuned encoder to produce latents that remain close
    to the original encoder's distribution, which helps maintain compatibility
    with pretrained decoders and prevents distribution shift.

    Two modes are supported:
    1. 'pointwise': Treat each latent as a sample and compute KL based on
       assuming unit Gaussian prior (simpler, faster)
    2. 'distribution': Estimate mean/variance from batch and compute KL between
       the two Gaussian distributions (more accurate)
    """

    def __init__(self, original_encoder, device='cuda', mode='distribution', eps=1e-6):
        """
        Args:
            original_encoder: The frozen original encoder (with pretrained weights)
            device: Device to use
            mode: 'pointwise' or 'distribution'
            eps: Small value for numerical stability
        """
        super().__init__()
        self.device = device
        self.mode = mode
        self.eps = eps

        # Store the frozen original encoder
        self.original_encoder = original_encoder
        self.original_encoder.eval()
        for param in self.original_encoder.parameters():
            param.requires_grad = False

        print(f"[LatentKLLoss] Initialized with mode='{mode}'")

    def _compute_batch_statistics(self, latents):
        """
        Compute mean and variance statistics from a batch of latents.

        Args:
            latents: Tensor of shape (B, C, H, W)

        Returns:
            mu: Mean tensor of shape (C,)
            var: Variance tensor of shape (C,)
        """
        # Flatten spatial dimensions: (B, C, H, W) -> (B*H*W, C)
        B, C, H, W = latents.shape
        latents_flat = latents.permute(0, 2, 3, 1).reshape(-1, C)

        mu = latents_flat.mean(dim=0)
        var = latents_flat.var(dim=0, unbiased=True) + self.eps

        return mu, var

    def _kl_divergence_gaussian(self, mu1, var1, mu2, var2):
        """
        Compute KL divergence between two diagonal Gaussians.

        KL(N(mu1, var1) || N(mu2, var2)) =
            0.5 * sum(log(var2/var1) + (var1 + (mu1-mu2)^2)/var2 - 1)

        Args:
            mu1, var1: Mean and variance of first distribution (trained encoder)
            mu2, var2: Mean and variance of second distribution (original encoder)

        Returns:
            kl: Scalar KL divergence
        """
        kl = 0.5 * torch.sum(
            torch.log(var2 / var1) +
            (var1 + (mu1 - mu2) ** 2) / var2 - 1
        )
        return kl

    def _kl_to_standard_normal(self, mu, var):
        """
        Compute KL divergence from a Gaussian to standard normal N(0, 1).

        KL(N(mu, var) || N(0, 1)) = 0.5 * sum(mu^2 + var - log(var) - 1)

        Args:
            mu: Mean tensor
            var: Variance tensor

        Returns:
            kl: Scalar KL divergence
        """
        kl = 0.5 * torch.sum(mu ** 2 + var - torch.log(var) - 1)
        return kl

    def forward(self, images, trained_latents):
        """
        Compute KL divergence loss between trained encoder latents and original encoder latents.

        Args:
            images: Input images tensor of shape (B, 3, H, W)
            trained_latents: Latents from the trained encoder of shape (B, C, H', W')

        Returns:
            kl_loss: Scalar KL divergence loss
        """
        # Get latents from the frozen original encoder
        with torch.no_grad():
            original_latents = self.original_encoder(images)

        if self.mode == 'pointwise':
            # Pointwise mode: compute element-wise squared difference
            # This is a simplified version that doesn't require distribution estimation
            # Equivalent to assuming both have unit variance and computing squared Mahalanobis distance
            kl_loss = F.mse_loss(trained_latents, original_latents)

        elif self.mode == 'distribution':
            # Distribution mode: estimate Gaussian parameters and compute KL
            trained_mu, trained_var = self._compute_batch_statistics(trained_latents)
            original_mu, original_var = self._compute_batch_statistics(original_latents)

            # KL(trained || original)
            kl_loss = self._kl_divergence_gaussian(
                trained_mu, trained_var,
                original_mu, original_var
            )

            # Normalize by number of dimensions
            kl_loss = kl_loss / trained_mu.numel()

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return kl_loss

    @torch.no_grad()
    def compute_kl_stats(self, dataloader, trained_encoder):
        """
        Compute KL divergence statistics over an entire dataset.

        Args:
            dataloader: DataLoader yielding images
            trained_encoder: The trained encoder to evaluate

        Returns:
            dict: Statistics including mean KL, per-channel statistics, etc.
        """
        trained_encoder.eval()

        all_trained_latents = []
        all_original_latents = []

        for batch in tqdm(dataloader, desc="Computing KL statistics"):
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            images = images.to(self.device)

            # Get latents from both encoders
            trained_latents = trained_encoder(images)
            original_latents = self.original_encoder(images)

            all_trained_latents.append(trained_latents.cpu())
            all_original_latents.append(original_latents.cpu())

        # Concatenate all latents
        all_trained = torch.cat(all_trained_latents, dim=0)
        all_original = torch.cat(all_original_latents, dim=0)

        # Compute overall statistics
        B, C, H, W = all_trained.shape

        # Per-channel statistics
        trained_flat = all_trained.permute(0, 2, 3, 1).reshape(-1, C)
        original_flat = all_original.permute(0, 2, 3, 1).reshape(-1, C)

        trained_mu = trained_flat.mean(dim=0)
        trained_var = trained_flat.var(dim=0) + self.eps
        original_mu = original_flat.mean(dim=0)
        original_var = original_flat.var(dim=0) + self.eps

        # Per-channel KL
        per_channel_kl = 0.5 * (
                torch.log(original_var / trained_var) +
                (trained_var + (trained_mu - original_mu) ** 2) / original_var - 1
        )

        # Overall KL
        total_kl = self._kl_divergence_gaussian(
            trained_mu, trained_var,
            original_mu, original_var
        )

        # MSE between latents (pointwise)
        mse = F.mse_loss(all_trained, all_original)

        stats = {
            'total_kl': total_kl.item(),
            'mean_kl_per_channel': per_channel_kl.mean().item(),
            'max_kl_per_channel': per_channel_kl.max().item(),
            'min_kl_per_channel': per_channel_kl.min().item(),
            'latent_mse': mse.item(),
            'trained_mu_mean': trained_mu.mean().item(),
            'trained_var_mean': trained_var.mean().item(),
            'original_mu_mean': original_mu.mean().item(),
            'original_var_mean': original_var.mean().item(),
            'mu_diff_mean': (trained_mu - original_mu).abs().mean().item(),
            'var_ratio_mean': (trained_var / original_var).mean().item(),
        }

        return stats


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

        # KL divergence loss (optional) - uses original encoder as reference
        self.kl_loss = None
        if args.use_kl:
            # Create a frozen copy of the original encoder
            original_encoder = copy.deepcopy(self.model.encoder)
            original_encoder.eval()
            for param in original_encoder.parameters():
                param.requires_grad = False

            self.kl_loss = LatentKLLoss(
                original_encoder=original_encoder,
                device=self.device,
                mode=args.kl_mode,
            )
            print(f"KL divergence loss enabled against original encoder")
            print(f"  Mode: {args.kl_mode}, Weight: {args.kl_weight}")

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

    def compute_loss(self, pred, target, inputs=None, latents=None, compute_kl=True):
        """Compute reconstruction loss."""
        # L1 loss
        loss_l1 = self.l1_loss(pred, target)

        # L2/MSE loss
        loss_l2 = self.mse_loss(pred, target)
        # print(f"L2 loss: {loss_l2:.4f}")

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

        # KL divergence loss (optional)
        loss_kl = torch.tensor(0.0, device=self.device)
        if self.kl_loss is not None and compute_kl and inputs is not None and latents is not None:
            # Only compute KL loss every N steps to save computation
            if self.global_step % self.args.kl_every == 0:
                loss_kl = self.kl_loss(inputs, latents)
                # print(f"KL loss: {loss_kl:.4f}")

                loss = loss + self.args.kl_weight * loss_kl

        return loss, {
            'l1': loss_l1.item(),
            'l2': loss_l2.item(),
            'lpips': loss_lpips.item(),
            'kl': loss_kl.item()
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

        # Collect all outputs for KL computation
        all_inputs = []
        all_latents = []

        pbar = tqdm(dataloader, desc=f"Evaluating {dataset_name}")
        for batch_idx, (padded, original, orig_h, orig_w, img_type) in enumerate(pbar):
            padded = padded.to(self.device)
            original = original.to(self.device)

            # Forward pass (get latents for KL computation)
            latents = self.model.encoder(padded)
            output = self.model.decoder(latents)

            # Remove padding
            output = output[:, :, :orig_h, :orig_w]
            output = output.clamp(0, 1)

            # Compute metrics (without KL for per-sample evaluation)
            psnr = self.compute_psnr(output, original)
            loss, _ = self.compute_loss(output, original, compute_kl=False)

            total_psnr += psnr
            total_loss += loss.item()
            count += 1

            # Collect for KL computation
            if self.kl_loss is not None:
                all_inputs.append(padded.cpu())
                all_latents.append(latents.cpu())

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

        # Compute KL statistics if enabled
        if self.kl_loss is not None and all_inputs:
            print(f"  Computing KL statistics for {dataset_name}...")

            # Create a simple dataloader from collected data
            class LatentDataset(Dataset):
                def __init__(self, inputs, latents):
                    self.inputs = torch.cat(inputs, dim=0)
                    self.latents = torch.cat(latents, dim=0)

                def __len__(self):
                    return len(self.inputs)

                def __getitem__(self, idx):
                    return self.inputs[idx], self.latents[idx]

            # Compute batch-level KL
            all_inputs_cat = torch.cat(all_inputs, dim=0).to(self.device)
            all_latents_cat = torch.cat(all_latents, dim=0).to(self.device)

            # Compute in smaller batches to avoid memory issues
            batch_size = 32
            total_kl = 0
            num_batches = (len(all_inputs_cat) + batch_size - 1) // batch_size
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(all_inputs_cat))
                batch_inputs = all_inputs_cat[start_idx:end_idx]
                batch_latents = all_latents_cat[start_idx:end_idx]
                total_kl += self.kl_loss(batch_inputs, batch_latents).item()

            kl_score = total_kl / num_batches
            metrics['kl'] = kl_score
            print(f"  KL divergence: {kl_score:.4f}")

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
        epoch_kl = 0
        kl_count = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.args.epochs}")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass - get latents separately for KL computation
            self.optimizer.zero_grad()
            latents = self.model.encoder(inputs)
            outputs = self.model.decoder(latents)

            # Compute loss (pass inputs and latents for KL computation)
            loss, loss_dict = self.compute_loss(outputs, targets, inputs=inputs, latents=latents)

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

            if loss_dict['kl'] > 0:
                epoch_kl += loss_dict['kl']
                kl_count += 1

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
            if loss_dict['kl'] > 0:
                postfix['kl'] = f'{loss_dict["kl"]:.4f}'
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
        all_inputs = []
        all_latents = []

        for batch_idx, (padded, original, orig_h, orig_w, img_type) in enumerate(self.val_loader):
            padded = padded.to(self.device)
            original = original.to(self.device)

            # Forward pass
            latents = self.model.encoder(padded)
            output = self.model.decoder(latents)

            # Remove padding
            output = output[:, :, :orig_h, :orig_w]
            output = output.clamp(0, 1)

            # Compute metrics
            psnr = self.compute_psnr(output, original)
            loss, _ = self.compute_loss(output, original, compute_kl=False)

            total_psnr += psnr
            total_loss += loss.item()
            count += 1

            if self.kl_loss is not None:
                all_inputs.append(padded.cpu())
                all_latents.append(latents.cpu())

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

            # Compute and log KL if enabled
            if self.kl_loss is not None and all_inputs:
                all_inputs_cat = torch.cat(all_inputs, dim=0).to(self.device)
                all_latents_cat = torch.cat(all_latents, dim=0).to(self.device)

                batch_size = 32
                total_kl = 0
                num_batches = (len(all_inputs_cat) + batch_size - 1) // batch_size
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(all_inputs_cat))
                    batch_inputs = all_inputs_cat[start_idx:end_idx]
                    batch_latents = all_latents_cat[start_idx:end_idx]
                    total_kl += self.kl_loss(batch_inputs, batch_latents).item()

                kl_score = total_kl / num_batches
                self.writer.add_scalar('val/kl', kl_score, epoch)
                print(f"  Validation KL: {kl_score:.4f}")

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
        if 'kl' in train_metrics:
            print(f"    KL divergence: {train_metrics['kl']:.4f}")

        # Log to tensorboard
        if self.writer:
            self.writer.add_scalar('initial_eval/train_psnr', train_metrics['psnr'], 0)
            self.writer.add_scalar('initial_eval/train_loss', train_metrics['loss'], 0)
            if 'psnr_ll' in train_metrics:
                self.writer.add_scalar('initial_eval/train_psnr_ll', train_metrics['psnr_ll'], 0)
            if 'psnr_hl' in train_metrics:
                self.writer.add_scalar('initial_eval/train_psnr_hl', train_metrics['psnr_hl'], 0)
            if 'kl' in train_metrics:
                self.writer.add_scalar('initial_eval/train_kl', train_metrics['kl'], 0)

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
            if 'kl' in val_metrics:
                print(f"    KL divergence: {val_metrics['kl']:.4f}")

            # Log to tensorboard
            if self.writer:
                self.writer.add_scalar('initial_eval/val_psnr', val_metrics['psnr'], 0)
                self.writer.add_scalar('initial_eval/val_loss', val_metrics['loss'], 0)
                if 'psnr_ll' in val_metrics:
                    self.writer.add_scalar('initial_eval/val_psnr_ll', val_metrics['psnr_ll'], 0)
                if 'psnr_hl' in val_metrics:
                    self.writer.add_scalar('initial_eval/val_psnr_hl', val_metrics['psnr_hl'], 0)
                if 'kl' in val_metrics:
                    self.writer.add_scalar('initial_eval/val_kl', val_metrics['kl'], 0)

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
        if 'kl' in train_metrics:
            print(f"    KL divergence: {train_metrics['kl']:.4f}")

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
            if 'kl' in val_metrics:
                print(f"    KL divergence: {val_metrics['kl']:.4f}")

        # Save results to file
        results_path = os.path.join(self.output_dir, "evaluation_results.txt")
        with open(results_path, 'w') as f:
            f.write(f"Evaluation Results\n")
            f.write(f"==================\n")
            f.write(f"Checkpoint: {self.args.resume}\n")
            f.write(f"Mode: {self.args.mode}\n")
            if self.args.use_kl:
                f.write(f"KL Mode: {self.args.kl_mode}\n")
            f.write(f"\n")

            f.write(f"Training Set:\n")
            f.write(f"  PSNR: {train_metrics['psnr']:.2f} dB\n")
            f.write(f"  Loss: {train_metrics['loss']:.4f}\n")
            f.write(f"  Count: {train_metrics['count']}\n")
            if 'psnr_ll' in train_metrics:
                f.write(f"  Low-light PSNR: {train_metrics['psnr_ll']:.2f} dB ({train_metrics['count_ll']} images)\n")
            if 'psnr_hl' in train_metrics:
                f.write(f"  High-light PSNR: {train_metrics['psnr_hl']:.2f} dB ({train_metrics['count_hl']} images)\n")
            if 'kl' in train_metrics:
                f.write(f"  KL divergence: {train_metrics['kl']:.4f}\n")

            if self.val_loader:
                f.write(f"\nValidation Set:\n")
                f.write(f"  PSNR: {val_metrics['psnr']:.2f} dB\n")
                f.write(f"  Loss: {val_metrics['loss']:.4f}\n")
                f.write(f"  Count: {val_metrics['count']}\n")
                if 'psnr_ll' in val_metrics:
                    f.write(f"  Low-light PSNR: {val_metrics['psnr_ll']:.2f} dB ({val_metrics['count_ll']} images)\n")
                if 'psnr_hl' in val_metrics:
                    f.write(f"  High-light PSNR: {val_metrics['psnr_hl']:.2f} dB ({val_metrics['count_hl']} images)\n")
                if 'kl' in val_metrics:
                    f.write(f"  KL divergence: {val_metrics['kl']:.4f}\n")

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
        if self.kl_loss is not None:
            print(f"KL loss enabled (mode: {self.args.kl_mode}, weight: {self.args.kl_weight})")
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
    parser.add_argument('--l1_weight', type=float, default=0.0,
                        help='Weight for L1 loss')
    parser.add_argument('--l2_weight', type=float, default=1.0,
                        help='Weight for L2/MSE loss')
    parser.add_argument('--use_lpips', action='store_true',
                        help='Use LPIPS perceptual loss')
    parser.add_argument('--lpips_weight', type=float, default=0.1,
                        help='Weight for LPIPS loss')

    # KL divergence loss options (against original encoder distribution)
    parser.add_argument('--use_kl', action='store_true',
                        help='Use KL divergence loss against original encoder distribution')
    parser.add_argument('--kl_weight', type=float, default=0.01,
                        help='Weight for KL divergence loss (typically small, e.g., 0.001-0.1)')
    parser.add_argument('--kl_mode', type=str, default='distribution', choices=['pointwise', 'distribution'],
                        help='KL computation mode: pointwise (MSE-like) or distribution (Gaussian KL)')
    parser.add_argument('--kl_every', type=int, default=1,
                        help='Compute KL loss every N training steps (to save computation)')

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

    trainer = TAESDTrainer(args)

    if args.eval_only:
        trainer.run_evaluation_only()
    else:
        trainer.train()


if __name__ == '__main__':
    main()