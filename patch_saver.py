import torch
import matplotlib.pyplot as plt
import os
from datetime import datetime
from pathlib import Path


class PatchSaver:
    def __init__(self, output_dir="patches"):
        """Initialize with timestamped output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.counter = 0
        print(f"Saving patches to: {self.output_dir}")

    def save_batch(self, low_q, prediction, high_q):
        """
        Save batch of patch triplets.

        Args:
            low_q: Tensor of shape (N, C, H, W)
            prediction: Tensor of shape (N, C, H, W)
            high_q: Tensor of shape (N, C, H, W)
        """
        # Verify batch dimensions match
        assert low_q.shape[0] == prediction.shape[0] == high_q.shape[0], \
            f"Batch dimensions don't match: {low_q.shape[0]}, {prediction.shape[0]}, {high_q.shape[0]}"

        batch_size = low_q.shape[0]

        # Move to CPU and convert to numpy
        low_np = low_q.cpu().detach().numpy().transpose(0, 2, 3, 1)
        pred_np = prediction.cpu().detach().numpy().transpose(0, 2, 3, 1)
        high_np = high_q.cpu().detach().numpy().transpose(0, 2, 3, 1)

        # Normalize to [0, 1] if needed
        for arr in [low_np, pred_np, high_np]:
            if arr.max() > 1.0:
                arr /= 255.0

        # Clip to valid range
        low_np = low_np.clip(0, 1)
        pred_np = pred_np.clip(0, 1)
        high_np = high_np.clip(0, 1)

        for i in range(batch_size):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].imshow(low_np[i])
            axes[0].set_title("Low Q")
            axes[0].axis("off")

            axes[1].imshow(pred_np[i])
            axes[1].set_title("Prediction")
            axes[1].axis("off")

            axes[2].imshow(high_np[i])
            axes[2].set_title("High Q")
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig(self.output_dir / f"patch_{self.counter:05d}.png", dpi=150)
            plt.close()

            self.counter += 1

        # print(f"Saved {batch_size} patches (total: {self.counter})")


# Usage
# saver = PatchSaver("output_patches")
# saver.save_batch(low_q_tensor1, pred_tensor1, high_q_tensor1)
# saver.save_batch(low_q_tensor2, pred_tensor2, high_q_tensor2)  # Won't override
