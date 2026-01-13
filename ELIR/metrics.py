import os
import shutil
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import save_image
import pyiqa


class MetricEval(object):
    def __init__(self, metric, device, out_dir=None):
        self.metric = metric
        self.values = {}  # Dict: dataloader_idx -> list of values
        self.count = {}   # Dict: dataloader_idx -> count
        self.image_size = {}  # Dict: dataloader_idx -> image_size
        self.out_dir = out_dir
        self.device = device

        # Per-dataloader directories for metrics that save images
        self.dl_out_dirs = {}  # Dict: dataloader_idx -> output_dir
        self.pred_dirs = {}    # Dict: dataloader_idx -> pred_dir (for fid)
        self.gt_dirs = {}      # Dict: dataloader_idx -> gt_dir (for fid)

        if metric == "fid-g":
            self.evaluater = FrechetInceptionDistance(normalize=True, reset_real_features=False)
            load_fid_statistics(os.path.join(os.path.dirname(__file__), "datasets", "celeba_fid_stat.pt"), self.evaluater, device)
            self.evaluater.to(device)
        elif metric in ["fid", "fid-f"]:
            # Directories will be created per dataloader_idx in _ensure_dl_idx
            self.evaluater = pyiqa.create_metric('fid', device=device, dims=64)
        elif metric == "lpips":
            self.evaluater = pyiqa.create_metric(metric, device=device, net="vgg")
        elif metric == "save":
            # Directories will be created per dataloader_idx in _ensure_dl_idx
            pass
        else:
            self.evaluater = pyiqa.create_metric(metric, device=device)

    def _ensure_dl_idx(self, dataloader_idx):
        if dataloader_idx not in self.values:
            self.values[dataloader_idx] = []
            self.count[dataloader_idx] = 0
            self.image_size[dataloader_idx] = 512

            # Create per-dataloader directories for metrics that save images
            if self.metric in ["fid", "fid-f", "save"]:
                base_dir = "out" if self.out_dir is None else self.out_dir
                if self.out_dir is not None:
                    os.makedirs(self.out_dir, exist_ok=True)

                dl_out_dir = os.path.join(base_dir, f"{self.metric}_dl{dataloader_idx}")
                self.dl_out_dirs[dataloader_idx] = dl_out_dir

                shutil.rmtree(dl_out_dir, ignore_errors=True)

                if self.metric == "fid":
                    # FID needs separate pred and gt directories
                    pred_dir = os.path.join(dl_out_dir, "pred")
                    gt_dir = os.path.join(dl_out_dir, "gt")
                    self.pred_dirs[dataloader_idx] = pred_dir
                    self.gt_dirs[dataloader_idx] = gt_dir
                    os.makedirs(pred_dir, exist_ok=True)
                    os.makedirs(gt_dir, exist_ok=True)
                else:
                    os.makedirs(dl_out_dir, exist_ok=True)

    def compute(self, x_hq_hat, x_hq=None, dataloader_idx=0):
        self._ensure_dl_idx(dataloader_idx)
        bs = x_hq_hat.shape[0]

        if self.metric == "fid-g":
            self.evaluater.to(x_hq_hat.device)
            self.evaluater.update(x_hq_hat, real=False)
            value = torch.zeros((len(x_hq_hat),))
            self.count[dataloader_idx] += bs
        elif self.metric == "fid":
            # FID: save both predictions and ground truth
            self.image_size[dataloader_idx] = x_hq.shape[-1]
            pred_dir = self.pred_dirs[dataloader_idx]
            gt_dir = self.gt_dirs[dataloader_idx]
            for img, gt in zip(x_hq_hat, x_hq):
                save_image(img, os.path.join(pred_dir, str(self.count[dataloader_idx]) + ".png"))
                save_image(gt, os.path.join(gt_dir, str(self.count[dataloader_idx]) + ".png"))
                self.count[dataloader_idx] += 1
            value = torch.zeros((len(x_hq_hat),))
        elif self.metric == "fid-f":
            self.image_size[dataloader_idx] = x_hq.shape[-1]
            dl_out_dir = self.dl_out_dirs[dataloader_idx]
            for img in x_hq_hat:
                save_image(img, os.path.join(dl_out_dir, str(self.count[dataloader_idx]) + ".png"))
                self.count[dataloader_idx] += 1
            value = torch.zeros((len(x_hq_hat),))
        elif self.metric in ["niqe", "clipiqa", "musiq"]:
            value = self.evaluater(x_hq_hat)
            self.count[dataloader_idx] += bs
        elif self.metric == "save":
            dl_out_dir = self.dl_out_dirs[dataloader_idx]
            for img in x_hq_hat:
                save_image(img, os.path.join(dl_out_dir, str(self.count[dataloader_idx]) + ".png"))
                self.count[dataloader_idx] += 1
            value = torch.zeros((len(x_hq_hat),))
        else:
            assert x_hq_hat.shape == x_hq.shape
            value = self.evaluater(x_hq_hat, x_hq)
            self.count[dataloader_idx] += bs

        self.values[dataloader_idx].append(value.reshape(-1))

    def get_final(self, dataloader_idx=0):
        if self.metric == "fid-g":
            final_value = self.evaluater.compute()
            self.evaluater.reset()
        elif self.metric == "fid":
            # FID: compare pred vs gt directories
            pred_dir = self.pred_dirs[dataloader_idx]
            gt_dir = self.gt_dirs[dataloader_idx]
            dl_out_dir = self.dl_out_dirs[dataloader_idx]
            image_size = self.image_size.get(dataloader_idx, 512)

            final_value = self.evaluater(pred_dir, gt_dir, dataset_res=image_size)

            # Clean up and recreate directories for next use
            shutil.rmtree(dl_out_dir, ignore_errors=True)
            os.makedirs(pred_dir, exist_ok=True)
            os.makedirs(gt_dir, exist_ok=True)
        elif self.metric == "fid-f":
            dl_out_dir = self.dl_out_dirs[dataloader_idx]
            image_size = self.image_size.get(dataloader_idx, 512)

            final_value = self.evaluater(dl_out_dir, dataset_name="FFHQ",
                                         dataset_res=image_size,
                                         dataset_split="trainval70k", verbose=False)

            # Clean up and recreate directory for next use
            shutil.rmtree(dl_out_dir, ignore_errors=True)
            os.makedirs(dl_out_dir, exist_ok=True)
        else:
            final_value = torch.mean(torch.concat(self.values.get(dataloader_idx, [torch.tensor([0.0])])))

        self.count[dataloader_idx] = 0
        self.values[dataloader_idx] = []
        return final_value

    def get_final_all(self):
        """Returns dict mapping dataloader_idx -> final metric value."""
        results = {}
        for dl_idx in self.values.keys():
            results[dl_idx] = self.get_final(dl_idx)
        return results

def load_fid_statistics(stat_path, evaluator, device):
    real_features = torch.load(stat_path, map_location=device)
    evaluator.real_features_sum = real_features['real_features_sum']
    evaluator.real_features_cov_sum = real_features['real_features_cov_sum']
    evaluator.real_features_num_samples = real_features['real_features_num_samples']

def save_fid_statistics(evaluater):
    d = {"real_features_sum": evaluater.real_features_sum,
         "real_features_cov_sum": evaluater.real_features_cov_sum,
         "real_features_num_samples": evaluater.real_features_num_samples}
    torch.save(d, "a.pt")
