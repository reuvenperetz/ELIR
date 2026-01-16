from typing import Union, Optional, Callable, Any
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch.optim import Optimizer
import pytorch_lightning as L
import torch
from ELIR.metrics import MetricEval
from ELIR.training.losses import get_loss
from torchvision.utils import save_image
from ELIR.training.ema_timm import ModelEMA
import os
import torch.nn.functional as F
from ELIR.utils import ImageSpliterTh
import math
from PIL import Image
import numpy as np

from patch_saver import PatchSaver


class IRSetup(L.LightningModule):
    def __init__(self, model, fm_cfg={}, optimizer=None, scheduler=None, tmodel=None,
                 ema_decay=None, eval_cfg=None, run_dir=None, save_images=True,
                 val_dataset_names=None, image_logging_mode="local"):
        super().__init__()
        self.model = model
        self.fm_cfg = fm_cfg
        self.eval_cfg = eval_cfg
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = eval_cfg.get("metrics",[])
        self.tmodel = tmodel
        if torch.cuda.is_available():
            self.acc_device = torch.device(torch.cuda.current_device())
        elif torch.backends.mps.is_available():
            self.acc_device = torch.device('mps')
        else:
            self.acc_device = torch.device('cpu')
        self.metric_evals = [MetricEval(metric, self.acc_device, run_dir) for metric in self.metrics]
        self.train_loss = []
        self.ema = None
        if ema_decay:
            self.ema = ModelEMA(model, device=self.acc_device, decay=ema_decay)
        self.samples_dir = None
        if run_dir and save_images:
            self.samples_dir = os.path.join(run_dir, "samples")
            os.makedirs(self.samples_dir, exist_ok=True)  # run folder
            self.samples = []
        self.patch_saver = PatchSaver()

        # For MLflow image logging per validation dataset
        self.val_dataset_names = val_dataset_names or []
        self.val_samples = {}  # Dict: dataloader_idx -> (input, prediction, ground_truth)

        # Image logging mode: "none", "local" (default), or "mlflow"
        self.image_logging_mode = image_logging_mode
        self.logged_reference_images = set()  # Track which dataloaders have logged input/gt

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Union[Optimizer, LightningOptimizer],
        optimizer_closure: Optional[Callable[[], Any]] = None,
    ) -> None:
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        if self.ema:
            self.ema.update(self.model)

    def on_train_epoch_start(self):
        """Called at the start of each training epoch."""
        # Update dataset's epoch for dynamic patch size scheduling
        if self.trainer and self.trainer.train_dataloader:
            train_dataloader = self.trainer.train_dataloader
            dataset = None

            # Get the dataset from the dataloader
            if hasattr(train_dataloader, 'dataset'):
                dataset = train_dataloader.dataset
            elif hasattr(train_dataloader, 'loaders'):
                # CombinedLoader case
                loaders = train_dataloader.loaders
                if isinstance(loaders, list) and len(loaders) > 0:
                    dataset = loaders[0].dataset if hasattr(loaders[0], 'dataset') else None
                elif hasattr(loaders, 'dataset'):
                    dataset = loaders.dataset

            # Update epoch on dataset if it supports set_epoch
            if dataset is not None and hasattr(dataset, 'set_epoch'):
                dataset.set_epoch(self.current_epoch)

    def training_step(self, batch, batch_idx):
        x_lq, x_hq = batch[0], batch[1]
        # Print input shape for each rank (useful for DDP debugging)
        rank = self.global_rank if hasattr(self, 'global_rank') else 0
        print(f"[Rank {rank}] Train batch {batch_idx} - x_lq: {x_lq.shape}, x_hq: {x_hq.shape}")
        # Loss function - now returns a dict of loss components
        loss_dict = get_loss(self.model, x_hq, x_lq, self.fm_cfg, self.tmodel)

        # Log total loss
        self.log("train_loss_total", loss_dict['loss_total'], on_epoch=True, prog_bar=True, logger=True)

        # Log individual loss components
        for name, value in loss_dict.items():
            if name != 'loss_total':
                self.log(f"train_{name}", value, on_epoch=True, prog_bar=False, logger=True)

        return loss_dict['loss_total']


    def compute_metrics(self, x_hq_hat, x_hq, dataloader_idx=0):
        for metric_eval in self.metric_evals:
            metric_eval.compute(x_hq_hat, x_hq, dataloader_idx)


    def save_samples(self, current_epoch):
        save_image(torch.concat(self.samples,dim=0), os.path.join(self.samples_dir,"epoch_"+str(current_epoch)+".png"))

    def infer(self, x):
        if self.ema:
            return self.ema.model.inference(x)
        else:
            return self.model.inference(x)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Handle both formats: (lq, hq) for training and (lq, hq, orig_dims) for validation with padding
        if len(batch) == 3:
            x_lq, y, orig_dims = batch
            # orig_dims is (B, 2) tensor with [orig_h, orig_w] for each sample
        else:
            x_lq, y = batch
            orig_dims = None

        # Print input shape for each rank (useful for DDP debugging)
        rank = self.global_rank if hasattr(self, 'global_rank') else 0
        print(f"[Rank {rank}] Val dataloader {dataloader_idx} batch {batch_idx} - x_lq: {x_lq.shape}, y: {y.shape}")
        chop = self.eval_cfg.get("chop", None)
        if chop:
            sf = chop.get("sf", 4)
            upscale = chop.get("upscale", 4)
            chop_size = chop.get("chop_size", 256)
            chop_stride = chop.get("chop_stride", 224)
            patch_spliter = ImageSpliterTh(x_lq, pch_size=chop_size, stride=chop_stride, sf=sf, extra_bs=1)
            for patch, index_infos in patch_spliter:
                patch_h, patch_w = patch.shape[2:]
                flag_pad = False
                if not (patch_h % 64 == 0 and patch_w % 64 == 0):
                    flag_pad = True
                    pad_h = (math.ceil(patch_h / 64)) * 64 - patch_h
                    pad_w = (math.ceil(patch_w / 64)) * 64 - patch_w
                    patch = F.pad(patch, pad=(0, pad_w, 0, pad_h), mode='reflect')
                pad_patch_h, pad_patch_w = patch.shape[2:]
                patch = F.interpolate(patch, size=(upscale*pad_patch_h, upscale*pad_patch_w), mode='bicubic')
                y_hat = self.infer(patch)
                if flag_pad:
                    y_hat = y_hat[:, :, :patch_h * sf, :patch_w * sf]
                patch_spliter.update(y_hat, index_infos)
            y_hat = patch_spliter.gather()
        else:
            y_hat = self.infer(x_lq)

        # Crop predictions and ground truth back to original dimensions if padded
        if orig_dims is not None:
            # All images in batch should have same original size (LOLv1 is 400x600)
            orig_h, orig_w = orig_dims[0, 0].item(), orig_dims[0, 1].item()
            y_hat = y_hat[:, :, :orig_h, :orig_w]
            y = y[:, :, :orig_h, :orig_w]
            x_lq = x_lq[:, :, :orig_h, :orig_w]

        # Capture one sample per validation dataloader for MLflow image logging
        if batch_idx == 0 and dataloader_idx not in self.val_samples:
            # Store first batch's first image for each validation dataloader
            self.val_samples[dataloader_idx] = (
                x_lq[0:1].detach().cpu(),
                y_hat[0:1].detach().cpu(),
                y[0:1].detach().cpu()
            )

        self.compute_metrics(y_hat, y, dataloader_idx)


    def on_validation_epoch_end(self):
        self.log('global_step', self.global_step)
        for metric_eval in self.metric_evals:
            results = metric_eval.get_final_all()  # Returns dict per dataloader
            for dl_idx, result in results.items():
                suffix = f"_val{dl_idx}" if dl_idx > 0 else ""
                self.log(f"{metric_eval.metric}{suffix}", result.item(), sync_dist=True, prog_bar=True)

        # Log comparison images to MLflow
        self._log_validation_images()

        # Clear samples for next epoch
        self.val_samples.clear()
        torch.cuda.empty_cache()

    def _log_validation_images(self):
        """Log validation images. Input/GT saved only on first epoch, predictions saved every epoch."""
        if self.image_logging_mode == "none":
            return

        if self.image_logging_mode == "local" and self.samples_dir is None:
            return

        if self.image_logging_mode == "mlflow" and self.logger is None:
            return

        for dl_idx, (x_lq, y_hat, y) in self.val_samples.items():
            # Get dataset name
            if dl_idx < len(self.val_dataset_names):
                dataset_name = self.val_dataset_names[dl_idx]
            else:
                dataset_name = f"val_{dl_idx}"

            # Save input and ground truth only once (first epoch for each dataloader)
            if dl_idx not in self.logged_reference_images:
                self._save_single_image(x_lq, dataset_name, "input")
                self._save_single_image(y, dataset_name, "ground_truth")
                self.logged_reference_images.add(dl_idx)

            # Save prediction every epoch
            self._save_single_image(y_hat, dataset_name, f"pred_epoch_{self.current_epoch}")

    def _save_single_image(self, tensor, dataset_name, image_name):
        """Save a single image tensor locally or to MLflow."""
        # Convert tensor to PIL image
        img = tensor[0].clamp(0, 1)  # Take first image, clamp to [0,1]
        img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)

        # Create subfolder for this dataset
        if self.samples_dir:
            dataset_dir = os.path.join(self.samples_dir, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)
            save_path = os.path.join(dataset_dir, f"{image_name}.png")
            pil_image.save(save_path)

            # If mlflow mode, also log to MLflow
            if self.image_logging_mode == "mlflow":
                try:
                    if hasattr(self.logger, 'experiment') and self.logger.experiment is not None:
                        self.logger.experiment.log_artifact(
                            self.logger.run_id,
                            save_path,
                            artifact_path=f"val_images/{dataset_name}"
                        )
                except Exception as e:
                    print(f"Warning: Could not log image to MLflow: {e}")


    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            if hasattr(self.ema.model,"fmir"):
                checkpoint['state_dict_fmir'] = self.ema.model.fmir.state_dict()
            if hasattr(self.ema.model,"mmse"):
                checkpoint['state_dict_mmse'] = self.ema.model.mmse.state_dict()
            if hasattr(self.ema.model,"enc"):
                checkpoint['state_dict_enc'] = self.ema.model.enc.state_dict()
            if hasattr(self.ema.model,"dec"):
                checkpoint['state_dict_dec'] = self.ema.model.dec.state_dict()
        else:
            if hasattr(self.model,"fmir"):
                checkpoint['state_dict_fmir'] = self.model.fmir.state_dict()
            if hasattr(self.model,"mmse"):
                checkpoint['state_dict_mmse'] = self.model.mmse.state_dict()
            if hasattr(self.model,"enc"):
                checkpoint['state_dict_enc'] = self.model.enc.state_dict()
            if hasattr(self.model,"dec"):
                checkpoint['state_dict_dec'] = self.model.dec.state_dict()
        return checkpoint

    def configure_optimizers(self):
        if self.scheduler is None:
            return [self.optimizer]
        return [self.optimizer], [self.scheduler]
