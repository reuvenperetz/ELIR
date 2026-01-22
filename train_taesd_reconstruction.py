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

    def compute_loss(self, pred, target):
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

        return loss, {'l1': loss_l1.item(), 'l2': loss_l2.item(), 'lpips': loss_lpips.item()}

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

        pbar = tqdm(dataloader, desc=f"Evaluating {dataset_name}")
        for batch_idx, (padded, original, orig_h, orig_w, img_type) in enumerate(pbar):
            padded = padded.to(self.device)
            original = original.to(self.device)

            # Forward pass
            output = self.model(padded)

            # Remove padding
            output = output[:, :, :orig_h, :orig_w]
            output = output.clamp(0, 1)

            # Compute metrics
            psnr = self.compute_psnr(output, original)
            loss, _ = self.compute_loss(output, original)

            total_psnr += psnr
            total_loss += loss.item()
            count += 1

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

            # Logging
            self.global_step += 1
            if self.global_step % self.args.log_interval == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/psnr', psnr, self.global_step)
                self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)
                for k, v in loss_dict.items():
                    self.writer.add_scalar(f'train/loss_{k}', v, self.global_step)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'psnr': f'{psnr:.2f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })

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
            loss, _ = self.compute_loss(output, original)

            total_psnr += psnr
            total_loss += loss.item()
            count += 1

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

        # Log to tensorboard
        if self.writer:
            self.writer.add_scalar('initial_eval/train_psnr', train_metrics['psnr'], 0)
            self.writer.add_scalar('initial_eval/train_loss', train_metrics['loss'], 0)
            if 'psnr_ll' in train_metrics:
                self.writer.add_scalar('initial_eval/train_psnr_ll', train_metrics['psnr_ll'], 0)
            if 'psnr_hl' in train_metrics:
                self.writer.add_scalar('initial_eval/train_psnr_hl', train_metrics['psnr_hl'], 0)

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

            # Log to tensorboard
            if self.writer:
                self.writer.add_scalar('initial_eval/val_psnr', val_metrics['psnr'], 0)
                self.writer.add_scalar('initial_eval/val_loss', val_metrics['loss'], 0)
                if 'psnr_ll' in val_metrics:
                    self.writer.add_scalar('initial_eval/val_psnr_ll', val_metrics['psnr_ll'], 0)
                if 'psnr_hl' in val_metrics:
                    self.writer.add_scalar('initial_eval/val_psnr_hl', val_metrics['psnr_hl'], 0)

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

        # Save results to file
        results_path = os.path.join(self.output_dir, "evaluation_results.txt")
        with open(results_path, 'w') as f:
            f.write(f"Evaluation Results\n")
            f.write(f"==================\n")
            f.write(f"Checkpoint: {self.args.resume}\n")
            f.write(f"Mode: {self.args.mode}\n\n")

            f.write(f"Training Set:\n")
            f.write(f"  PSNR: {train_metrics['psnr']:.2f} dB\n")
            f.write(f"  Loss: {train_metrics['loss']:.4f}\n")
            f.write(f"  Count: {train_metrics['count']}\n")
            if 'psnr_ll' in train_metrics:
                f.write(f"  Low-light PSNR: {train_metrics['psnr_ll']:.2f} dB ({train_metrics['count_ll']} images)\n")
            if 'psnr_hl' in train_metrics:
                f.write(f"  High-light PSNR: {train_metrics['psnr_hl']:.2f} dB ({train_metrics['count_hl']} images)\n")

            if self.val_loader:
                f.write(f"\nValidation Set:\n")
                f.write(f"  PSNR: {val_metrics['psnr']:.2f} dB\n")
                f.write(f"  Loss: {val_metrics['loss']:.4f}\n")
                f.write(f"  Count: {val_metrics['count']}\n")
                if 'psnr_ll' in val_metrics:
                    f.write(f"  Low-light PSNR: {val_metrics['psnr_ll']:.2f} dB ({val_metrics['count_ll']} images)\n")
                if 'psnr_hl' in val_metrics:
                    f.write(f"  High-light PSNR: {val_metrics['psnr_hl']:.2f} dB ({val_metrics['count_hl']} images)\n")

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