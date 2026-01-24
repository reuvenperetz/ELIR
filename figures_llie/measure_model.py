#!/usr/bin/env python3
"""
Script to measure ELIR model statistics:
- Number of parameters (total and trainable)
- Latency (forward pass time)
- Number of MACs (Multiply-Accumulate operations)

Usage:
    python measure_model.py --config configs/llie/elir_train_llie_sid_from_denoising.yaml
    python measure_model.py --config configs/elir_train_sr.yaml --input_size 512
"""

import argparse
import time
import yaml
import torch
import numpy as np
from typing import Tuple, Dict, Any

# Try to import fvcore for MAC counting
try:
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False
    print("Warning: fvcore not installed. Install with 'pip install fvcore' for MAC counting.")

# Try to import thop as an alternative
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        # Handle PyTorch-style YAML tags
        yaml.add_constructor('!name:torch.optim.AdamW', lambda loader, node: torch.optim.AdamW, Loader=yaml.FullLoader)
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def build_model(config: Dict[str, Any]) -> torch.nn.Module:
    """Build ELIR model from configuration."""
    from ELIR.models.load_model import get_model

    model_cfg = config['model_cfg']['arch_cfg']
    model = get_model(model_cfg)
    return model


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters.

    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def count_parameters_by_component(model) -> Dict[str, Tuple[int, int]]:
    """Count parameters for each component of the ELIR model."""
    components = {}

    if hasattr(model, 'fmir'):
        total = sum(p.numel() for p in model.fmir.parameters())
        trainable = sum(p.numel() for p in model.fmir.parameters() if p.requires_grad)
        components['fmir'] = (total, trainable)

    if hasattr(model, 'mmse'):
        total = sum(p.numel() for p in model.mmse.parameters())
        trainable = sum(p.numel() for p in model.mmse.parameters() if p.requires_grad)
        components['mmse'] = (total, trainable)

    if hasattr(model, 'enc'):
        total = sum(p.numel() for p in model.enc.parameters())
        trainable = sum(p.numel() for p in model.enc.parameters() if p.requires_grad)
        components['encoder'] = (total, trainable)

    if hasattr(model, 'dec'):
        total = sum(p.numel() for p in model.dec.parameters())
        trainable = sum(p.numel() for p in model.dec.parameters() if p.requires_grad)
        components['decoder'] = (total, trainable)

    return components


def measure_latency(model: torch.nn.Module, input_tensor: torch.Tensor,
                    warmup_runs: int = 10, measurement_runs: int = 100) -> Dict[str, float]:
    """
    Measure forward pass latency.

    Args:
        model: The model to measure
        input_tensor: Input tensor of shape (B, C, H, W)
        warmup_runs: Number of warmup runs
        measurement_runs: Number of measurement runs

    Returns:
        Dictionary with latency statistics
    """
    model.eval()
    device = input_tensor.device

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()

    # Measurement
    latencies = []
    with torch.no_grad():
        for _ in range(measurement_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            _ = model(input_tensor)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            latencies.append((end_time - start_time) * 1000)  # Convert to ms

    latencies = np.array(latencies)

    return {
        'mean_ms': float(np.mean(latencies)),
        'std_ms': float(np.std(latencies)),
        'min_ms': float(np.min(latencies)),
        'max_ms': float(np.max(latencies)),
        'median_ms': float(np.median(latencies)),
        'fps': float(1000.0 / np.mean(latencies))
    }


def count_macs_fvcore(model: torch.nn.Module, input_tensor: torch.Tensor) -> int:
    """Count MACs using fvcore library."""
    model.eval()
    flops = FlopCountAnalysis(model, input_tensor)
    # FLOPs ≈ 2 * MACs for most operations
    macs = flops.total() // 2
    return macs


def count_macs_thop(model: torch.nn.Module, input_tensor: torch.Tensor) -> Tuple[int, int]:
    """Count MACs using thop library."""
    model.eval()
    macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    return int(macs), int(params)


def format_number(n: int) -> str:
    """Format large numbers with K, M, G suffixes."""
    if n >= 1e9:
        return f"{n/1e9:.2f}G"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    else:
        return str(n)


def main():
    parser = argparse.ArgumentParser(description='Measure ELIR model statistics')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML configuration file')
    parser.add_argument('--input_size', type=int, default=256,
                        help='Input image size (default: 256)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for measurement (default: 1)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Device to use (default: auto)')
    parser.add_argument('--warmup_runs', type=int, default=10,
                        help='Number of warmup runs for latency measurement')
    parser.add_argument('--measurement_runs', type=int, default=100,
                        help='Number of runs for latency measurement')

    args = parser.parse_args()

    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"=" * 60)
    print(f"ELIR Model Measurement")
    print(f"=" * 60)
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Input size: {args.batch_size}x3x{args.input_size}x{args.input_size}")
    print(f"=" * 60)

    # Load configuration and build model
    print("\n[1/4] Loading configuration and building model...")
    config = load_config(args.config)
    model = build_model(config)
    model = model.to(device)
    model.eval()

    # Create random input
    input_tensor = torch.randn(args.batch_size, 3, args.input_size, args.input_size, device=device)

    # Count parameters
    print("\n[2/4] Counting parameters...")
    total_params, trainable_params = count_parameters(model)
    component_params = count_parameters_by_component(model)

    print(f"\n{'='*60}")
    print("PARAMETER COUNT")
    print(f"{'='*60}")
    print(f"Total parameters:     {total_params:>12,} ({format_number(total_params)})")
    print(f"Trainable parameters: {trainable_params:>12,} ({format_number(trainable_params)})")
    print(f"\nBy component:")
    for name, (total, trainable) in component_params.items():
        print(f"  {name:12}: {total:>12,} total, {trainable:>12,} trainable")

    # Count MACs
    print("\n[3/4] Counting MACs...")
    macs = None

    if THOP_AVAILABLE:
        try:
            macs, _ = count_macs_thop(model, input_tensor)
            print(f"\n{'='*60}")
            print("MAC COUNT (via thop)")
            print(f"{'='*60}")
            print(f"MACs: {macs:>15,} ({format_number(macs)})")
            print(f"GMACs: {macs/1e9:.3f}")
        except Exception as e:
            print(f"Warning: Could not count MACs with thop: {e}")

    if FVCORE_AVAILABLE and macs is None:
        try:
            macs = count_macs_fvcore(model, input_tensor)
            print(f"\n{'='*60}")
            print("MAC COUNT (via fvcore)")
            print(f"{'='*60}")
            print(f"MACs: {macs:>15,} ({format_number(macs)})")
            print(f"GMACs: {macs/1e9:.3f}")
        except Exception as e:
            print(f"Warning: Could not count MACs with fvcore: {e}")

    if not THOP_AVAILABLE and not FVCORE_AVAILABLE:
        print("Warning: Install 'thop' or 'fvcore' to count MACs:")
        print("  pip install thop")
        print("  pip install fvcore")

    # Measure latency
    print("\n[4/4] Measuring latency...")
    latency_stats = measure_latency(
        model, input_tensor,
        warmup_runs=args.warmup_runs,
        measurement_runs=args.measurement_runs
    )

    print(f"\n{'='*60}")
    print(f"LATENCY ({args.measurement_runs} runs, batch_size={args.batch_size})")
    print(f"{'='*60}")
    print(f"Mean:   {latency_stats['mean_ms']:>10.2f} ms")
    print(f"Std:    {latency_stats['std_ms']:>10.2f} ms")
    print(f"Min:    {latency_stats['min_ms']:>10.2f} ms")
    print(f"Max:    {latency_stats['max_ms']:>10.2f} ms")
    print(f"Median: {latency_stats['median_ms']:>10.2f} ms")
    print(f"FPS:    {latency_stats['fps']:>10.2f}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Model: ELIR")
    print(f"Input: {args.batch_size}x3x{args.input_size}x{args.input_size}")
    print(f"Device: {device}")
    print(f"Parameters: {format_number(total_params)} ({format_number(trainable_params)} trainable)")
    if macs:
        print(f"MACs: {format_number(macs)} ({macs/1e9:.3f} GMACs)")
    print(f"Latency: {latency_stats['mean_ms']:.2f} ± {latency_stats['std_ms']:.2f} ms")
    print(f"Throughput: {latency_stats['fps']:.2f} FPS")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

