"""
Patch Size Scheduler for Dynamic Patch Size Training

Allows varying patch sizes during training with different modes:
- random: Randomly select from available patch sizes each step
- increasing: Start with smallest, gradually increase to largest
- decreasing: Start with largest, gradually decrease to smallest
- cyclic: Cycle through patch sizes repeatedly

Usage in config:
    dataset_cfg:
      train_dataset:
        patch_size_schedule:
          sizes: [128, 256, 384, 512]  # Available patch sizes
          mode: "random"                # random, increasing, decreasing, cyclic, step
          # For step mode:
          # step_epochs: [10, 20, 30]   # Epochs at which to change size
"""

import numpy as np
import random
from typing import List, Optional


class PatchSizeScheduler:
    """
    Scheduler for dynamic patch sizes during training.

    Args:
        sizes: List of patch sizes to use
        mode: Scheduling mode ('random', 'increasing', 'decreasing', 'cyclic', 'step')
        total_steps: Total number of training steps (for progressive modes)
        total_epochs: Total number of epochs (for epoch-based modes)
        step_epochs: List of epochs at which to change size (for 'step' mode)
        seed: Random seed for reproducibility
    """

    MODES = ['random', 'increasing', 'decreasing', 'cyclic', 'step']

    def __init__(
        self,
        sizes: List[int],
        mode: str = 'random',
        total_steps: Optional[int] = None,
        total_epochs: Optional[int] = None,
        step_epochs: Optional[List[int]] = None,
        seed: Optional[int] = None
    ):
        assert len(sizes) > 0, "Must provide at least one patch size"
        assert mode in self.MODES, f"Mode must be one of {self.MODES}, got {mode}"

        self.sizes = sorted(sizes)  # Sort ascending
        self.mode = mode
        self.total_steps = total_steps
        self.total_epochs = total_epochs
        self.step_epochs = step_epochs or []
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.current_size_idx = 0
        self.current_epoch = 0
        self.current_step = 0

    def get_patch_size(self, step: Optional[int] = None, epoch: Optional[int] = None) -> int:
        """
        Get the patch size for the current step/epoch.

        Args:
            step: Current training step (optional)
            epoch: Current epoch (optional)

        Returns:
            Patch size to use
        """
        if step is not None:
            self.current_step = step
        if epoch is not None:
            self.current_epoch = epoch

        if self.mode == 'random':
            return self._get_random()
        elif self.mode == 'increasing':
            return self._get_increasing()
        elif self.mode == 'decreasing':
            return self._get_decreasing()
        elif self.mode == 'cyclic':
            return self._get_cyclic()
        elif self.mode == 'step':
            return self._get_step()
        else:
            return self.sizes[0]

    def _get_random(self) -> int:
        """Randomly select a patch size."""
        return random.choice(self.sizes)

    def _get_increasing(self) -> int:
        """Progressively increase patch size based on training progress."""
        if self.total_epochs is not None and self.total_epochs > 0:
            progress = self.current_epoch / self.total_epochs
        elif self.total_steps is not None and self.total_steps > 0:
            progress = self.current_step / self.total_steps
        else:
            progress = 0.0

        progress = min(1.0, max(0.0, progress))
        idx = int(progress * (len(self.sizes) - 1) + 0.5)  # Round to nearest
        idx = min(idx, len(self.sizes) - 1)
        return self.sizes[idx]

    def _get_decreasing(self) -> int:
        """Progressively decrease patch size based on training progress."""
        if self.total_epochs is not None and self.total_epochs > 0:
            progress = self.current_epoch / self.total_epochs
        elif self.total_steps is not None and self.total_steps > 0:
            progress = self.current_step / self.total_steps
        else:
            progress = 0.0

        progress = min(1.0, max(0.0, progress))
        idx = int((1.0 - progress) * (len(self.sizes) - 1) + 0.5)  # Round to nearest
        idx = min(idx, len(self.sizes) - 1)
        return self.sizes[idx]

    def _get_cyclic(self) -> int:
        """Cycle through patch sizes each epoch."""
        if len(self.sizes) == 1:
            return self.sizes[0]

        # Simple cycling through sizes
        idx = self.current_epoch % len(self.sizes)
        return self.sizes[idx]

    def _get_step(self) -> int:
        """Change patch size at specific epoch intervals."""
        if not self.step_epochs:
            return self.sizes[0]

        # Find which interval we're in
        size_idx = 0
        for i, epoch_threshold in enumerate(self.step_epochs):
            if self.current_epoch >= epoch_threshold:
                size_idx = min(i + 1, len(self.sizes) - 1)
            else:
                break

        return self.sizes[size_idx]

    def set_epoch(self, epoch: int):
        """Update current epoch."""
        self.current_epoch = epoch

    def set_step(self, step: int):
        """Update current step."""
        self.current_step = step

    def __repr__(self) -> str:
        return (f"PatchSizeScheduler(sizes={self.sizes}, mode='{self.mode}', "
                f"current_epoch={self.current_epoch}, current_step={self.current_step})")


def create_patch_size_scheduler(
    config: dict,
    total_epochs: int = None,
    total_steps: int = None
) -> Optional[PatchSizeScheduler]:
    """
    Create a PatchSizeScheduler from config dictionary.

    Args:
        config: Dictionary containing patch_size_schedule configuration
        total_epochs: Total number of training epochs
        total_steps: Total number of training steps

    Returns:
        PatchSizeScheduler instance or None if not configured
    """
    schedule_cfg = config.get('patch_size_schedule', None)

    if schedule_cfg is None:
        return None

    sizes = schedule_cfg.get('sizes', [])
    if not sizes:
        return None

    mode = schedule_cfg.get('mode', 'random')
    step_epochs = schedule_cfg.get('step_epochs', None)
    seed = schedule_cfg.get('seed', None)

    return PatchSizeScheduler(
        sizes=sizes,
        mode=mode,
        total_epochs=total_epochs,
        total_steps=total_steps,
        step_epochs=step_epochs,
        seed=seed
    )

