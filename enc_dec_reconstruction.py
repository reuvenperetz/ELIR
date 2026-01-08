"""
Test Encoder/Decoder Reconstruction Quality

This script checks if the encoder/decoder reconstruction works correctly on clean images.
It passes ground truth HQ images through encoder → decoder (bypassing flow matching
and coarse estimation) to verify the latent space is usable.

If reconstruction looks good: The latent space is working, problem is in flow/coarse estimation.
If reconstruction looks bad: The encoder/decoder or latent space has issues.

Usage:
    # Use encoder/decoder from trained model (config)
    python enc_dec_reconstruction.py --config configs/llie/elir_train_llie_sid.yaml

    # Use pretrained TAESD from diffusers
    python enc_dec_reconstruction.py --config configs/llie/elir_train_llie_sid.yaml --use_pretrained taesd

    # Use pretrained TAESD3 from diffusers
    python enc_dec_reconstruction.py --config configs/llie/elir_train_llie_sid.yaml --use_pretrained taesd3

    # Use pretrained SD VAE from diffusers
    python enc_dec_reconstruction.py --config configs/llie/elir_train_llie_sid.yaml --use_pretrained sdxl-vae

    # Additional options
    python enc_dec_reconstruction.py --config configs/llie/elir_train_llie_sid.yaml --num_samples 5 --save_dir test_reconstruction
"""

import argparse
import os
import torch
import numpy as np
from torchvision.utils import save_image
from hyperpyyaml import load_hyperpyyaml
import pyiqa

from ELIR.models.load_model import get_model
from ELIR.datasets.dataset import get_loader


def parse_args():
    parser = argparse.ArgumentParser(description='Test encoder/decoder reconstruction')
    parser.add_argument('-y', '--config', dest='config_path', required=True,
                        help='Path to config file')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to test (default: 5)')
    parser.add_argument('--save_dir', type=str, default='reconstruction_test',
                        help='Directory to save reconstruction results')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (optional, uses config path if not specified)')
    parser.add_argument('--use_pretrained', type=str, default=None,
                        choices=['taesd', 'taesd3', 'sdxl-vae', 'sd-vae'],
                        help='Use pretrained autoencoder from diffusers instead of config model. '
                             'Options: taesd (Tiny AE for SD), taesd3 (Tiny AE for SD3), '
                             'sdxl-vae (SDXL VAE), sd-vae (SD 1.5 VAE)')
    return parser.parse_args()


def load_model_from_config(conf, checkpoint_path=None):
    """Load the ELIR model from config."""
    model_cfg = conf.get("model_cfg")
    arch_cfg = model_cfg.get("arch_cfg")

    # Override checkpoint path if specified
    if checkpoint_path:
        arch_cfg['path'] = checkpoint_path

    model = get_model(arch_cfg)
    model.eval()

    return model


class PretrainedAutoencoder:
    """Wrapper class for pretrained autoencoders from diffusers."""

    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self._load_model(model_name)

    def _load_model(self, model_name):
        """Load the pretrained autoencoder from diffusers."""
        if model_name == 'taesd':
            from diffusers import AutoencoderTiny
            print("Loading pretrained TAESD (madebyollin/taesd)...")
            self.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd")
            self.scale_factor = 8
            self.is_tiny = True
        elif model_name == 'taesd3':
            from diffusers import AutoencoderTiny
            print("Loading pretrained TAESD3 (madebyollin/taesd3)...")
            self.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd3")
            self.scale_factor = 8
            self.is_tiny = True
        elif model_name == 'sdxl-vae':
            from diffusers import AutoencoderKL
            print("Loading pretrained SDXL VAE (stabilityai/sdxl-vae)...")
            self.vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
            self.scale_factor = 8
            self.is_tiny = False
        elif model_name == 'sd-vae':
            from diffusers import AutoencoderKL
            print("Loading pretrained SD 1.5 VAE (stabilityai/sd-vae-ft-mse)...")
            self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
            self.scale_factor = 8
            self.is_tiny = False
        else:
            raise ValueError(f"Unknown pretrained model: {model_name}")

        self.vae.eval()
        self.vae.to(self.device)

    def to(self, device):
        self.device = device
        self.vae.to(device)
        return self

    def eval(self):
        self.vae.eval()
        return self

    @property
    def enc(self):
        """Return encoder-like callable."""
        return self._encode

    @property
    def dec(self):
        """Return decoder-like callable."""
        return self._decode

    def _encode(self, x):
        """Encode image to latent space."""
        with torch.no_grad():
            if self.is_tiny:
                # AutoencoderTiny uses encode() directly
                latent = self.vae.encode(x).latents
            else:
                # AutoencoderKL uses encode().latent_dist
                latent = self.vae.encode(x).latent_dist.sample()
        return latent

    def _decode(self, latent):
        """Decode latent to image."""
        with torch.no_grad():
            if self.is_tiny:
                # AutoencoderTiny
                decoded = self.vae.decode(latent).sample
            else:
                # AutoencoderKL
                decoded = self.vae.decode(latent).sample
        return decoded


def load_pretrained_autoencoder(model_name, device):
    """Load a pretrained autoencoder from diffusers.

    Args:
        model_name: Name of the pretrained model ('taesd', 'taesd3', 'sdxl-vae', 'sd-vae')
        device: torch device

    Returns:
        PretrainedAutoencoder wrapper with enc and dec attributes
    """
    return PretrainedAutoencoder(model_name, device)


def reconstruction(model, dataloader, num_samples, save_dir, device):
    """
    Test encoder/decoder reconstruction by passing HQ images through encoder → decoder.

    Args:
        model: ELIR model with encoder and decoder
        dataloader: DataLoader with (LQ, HQ) image pairs
        num_samples: Number of samples to test
        save_dir: Directory to save results
        device: torch device
    """
    os.makedirs(save_dir, exist_ok=True)

    # Check if model has encoder and decoder
    if not hasattr(model, 'enc') or not hasattr(model, 'dec'):
        raise ValueError("Model does not have 'enc' (encoder) or 'dec' (decoder) attributes")

    print(f"\n{'='*80}")
    print("Encoder/Decoder Reconstruction Test")
    print(f"{'='*80}")
    print(f"Testing {num_samples} samples...")
    print(f"Results will be saved to: {save_dir}")
    print(f"{'='*80}\n")

    # Initialize PSNR metric
    psnr_metric = pyiqa.create_metric('psnr', device=device)
    psnr_values = []

    model.to(device)
    model.eval()

    sample_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if sample_count >= num_samples:
                break

            # Get LQ and HQ images
            x_lq, x_hq = batch[0], batch[1]
            x_hq = x_hq.to(device)
            # x_lq = x_lq.to(device)

            batch_size = x_hq.shape[0]

            for i in range(batch_size):
                if sample_count >= num_samples:
                    break

                # Get single image
                hq_img = x_hq[i:i+1]  # [1, C, H, W]
                # lq_img = x_lq[i:i+1]  # [1, C, H, W]

                print(f"\nSample {sample_count + 1}/{num_samples}")
                print(f"  Input HQ shape: {hq_img.shape}")

                # Encode HQ image to latent space
                latent = model.enc(hq_img)
                print(f"  Latent shape: {latent.shape}")
                print(f"  Latent stats - min: {latent.min():.4f}, max: {latent.max():.4f}, mean: {latent.mean():.4f}, std: {latent.std():.4f}")

                # Decode latent back to image
                reconstructed = model.dec(latent)
                print(f"  Reconstructed shape: {reconstructed.shape}")

                # Handle size mismatch if any
                if reconstructed.shape != hq_img.shape:
                    print(f"  Warning: Shape mismatch! HQ: {hq_img.shape}, Reconstructed: {reconstructed.shape}")
                    raise ValueError("Shape mismatch!")

                # Clamp to valid range
                reconstructed = reconstructed.clamp(0, 1)

                # Compute PSNR
                psnr_val = psnr_metric(reconstructed, hq_img).item()
                psnr_values.append(psnr_val)
                print(f"  PSNR: {psnr_val:.2f} dB")

                # Save images
                sample_dir = os.path.join(save_dir, f"sample_{sample_count:03d}")
                os.makedirs(sample_dir, exist_ok=True)

                # Save individual images
                save_image(hq_img, os.path.join(sample_dir, "ground_truth_hq.png"))
                save_image(reconstructed, os.path.join(sample_dir, "reconstructed.png"))
                save_image(latent[0, 0:3].unsqueeze(0), os.path.join(sample_dir, "latent_vis.png"), normalize=True)

                # Create side-by-side comparison
                comparison = torch.cat([hq_img, reconstructed, (hq_img - reconstructed).abs()], dim=3)
                save_image(comparison, os.path.join(sample_dir, "comparison_hq_recon_diff.png"))

                # Save difference map with amplification for visibility
                diff = (hq_img - reconstructed).abs()
                diff_amplified = (diff * 5).clamp(0, 1)  # Amplify difference for visibility
                save_image(diff_amplified, os.path.join(sample_dir, "difference_amplified_5x.png"))

                sample_count += 1

    # Print summary
    print(f"\n{'='*80}")
    print("RECONSTRUCTION TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Tested {sample_count} samples")
    print(f"\nPSNR Results:")
    print(f"  Mean:  {np.mean(psnr_values):.2f} dB")
    print(f"  Std:   {np.std(psnr_values):.2f} dB")
    print(f"  Min:   {np.min(psnr_values):.2f} dB")
    print(f"  Max:   {np.max(psnr_values):.2f} dB")

    avg_psnr = np.mean(psnr_values)
    print(f"\nInterpretation:")
    if avg_psnr > 35:
        print("  ✅ EXCELLENT: Encoder/decoder reconstruction is very good!")
        print("     The latent space is working correctly.")
        print("     If final outputs are bad, the problem is in flow matching or coarse estimation.")
    elif avg_psnr > 30:
        print("  ✓ GOOD: Encoder/decoder reconstruction is acceptable.")
        print("     Some loss in the latent space, but should be usable.")
    elif avg_psnr > 25:
        print("  ⚠ MODERATE: Encoder/decoder reconstruction has noticeable loss.")
        print("     The latent space may be losing important information.")
    else:
        print("  ❌ POOR: Encoder/decoder reconstruction is poor!")
        print("     The latent space is losing significant information.")
        print("     Problem is likely in the encoder/decoder, not flow matching.")

    print(f"\nResults saved to: {save_dir}")
    print(f"{'='*80}\n")

    return {'psnr_values': psnr_values, 'psnr_mean': np.mean(psnr_values), 'psnr_std': np.std(psnr_values)}


def main():
    args = parse_args()

    # Load config
    with open(args.config_path, 'r') as f:
        conf = load_hyperpyyaml(f)

    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load model - either from config or pretrained from diffusers
    if args.use_pretrained:
        print(f"\nLoading pretrained autoencoder: {args.use_pretrained}")
        model = load_pretrained_autoencoder(args.use_pretrained, device)
        print(f"\nModel components:")
        print(f"  - Autoencoder: {args.use_pretrained} (from diffusers)")
        print(f"  - Scale factor: {model.scale_factor}")
    else:
        print("Loading model from config...")
        model = load_model_from_config(conf, args.checkpoint)
        # Print model structure info
        print(f"\nModel components:")
        if hasattr(model, 'enc'):
            print(f"  - Encoder: {model.enc.__class__.__name__}")
        if hasattr(model, 'dec'):
            print(f"  - Decoder: {model.dec.__class__.__name__}")
        if hasattr(model, 'fmir'):
            print(f"  - Flow Model (fmir): {model.fmir.__class__.__name__}")
        if hasattr(model, 'mmse'):
            print(f"  - Coarse Model (mmse): {model.mmse.__class__.__name__}")

    # Load validation dataset (use first validation dataset)
    dataset_cfg = conf.get("dataset_cfg")
    val_datasets = dataset_cfg.get('val_datasets', [])
    if not val_datasets:
        val_dataset = dataset_cfg.get('val_dataset')
        val_datasets = [val_dataset] if val_dataset else []

    if not val_datasets:
        raise ValueError("No validation dataset found in config")

    # Use first validation dataset
    val_dataset_cfg = val_datasets[0].copy()
    val_dataset_cfg['phase'] = 'val'
    print(f"\nLoading validation dataset from: {val_dataset_cfg.get('path')}")

    dataloader = get_loader(val_dataset_cfg)

    # Update save_dir to include model type
    save_dir = args.save_dir
    if args.use_pretrained:
        save_dir = f"{args.save_dir}_{args.use_pretrained}"

    # Run reconstruction test
    results = reconstruction(
        model=model,
        dataloader=dataloader,
        num_samples=args.num_samples,
        save_dir=save_dir,
        device=device
    )

    return results


if __name__ == "__main__":
    main()

