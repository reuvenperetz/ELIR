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
        self.image_size = 512
        self.out_dir = out_dir
        self.device = device
        if metric == "fid-g":
            self.evaluater = FrechetInceptionDistance(normalize=True, reset_real_features=False)
            load_fid_statistics(os.path.join(os.path.dirname(__file__), "datasets", "celeba_fid_stat.pt"), self.evaluater, device)
            self.evaluater.to(device)
        elif metric == "fid-f":
            self.out_dir = "out" if out_dir is None else os.path.join(self.out_dir, "out")
            shutil.rmtree(self.out_dir, ignore_errors=True)
            os.makedirs(self.out_dir, exist_ok=True)
            self.evaluater = pyiqa.create_metric('fid', device=device)
        elif metric == "lpips":
            self.evaluater = pyiqa.create_metric(metric, device=device, net="vgg")
        elif metric == "save":
            self.out_dir = "out" if out_dir is None else os.path.join(self.out_dir, "out")
            shutil.rmtree(self.out_dir, ignore_errors=True)
            os.makedirs(self.out_dir, exist_ok=True)
        else:
            self.evaluater = pyiqa.create_metric(metric, device=device)

    def _ensure_dl_idx(self, dataloader_idx):
        if dataloader_idx not in self.values:
            self.values[dataloader_idx] = []
            self.count[dataloader_idx] = 0

    def compute(self, x_hq_hat, x_hq=None, dataloader_idx=0):
        self._ensure_dl_idx(dataloader_idx)
        bs = x_hq_hat.shape[0]

        if self.metric == "fid-g":
            self.evaluater.to(x_hq_hat.device)
            self.evaluater.update(x_hq_hat, real=False)
            value = torch.zeros((len(x_hq_hat),))
            self.count[dataloader_idx] += bs
        elif self.metric == "fid-f":
            self.image_size = x_hq.shape[-1]
            for img in x_hq_hat:
                save_image(img, os.path.join(self.out_dir, str(self.count[dataloader_idx]) + ".png"))
                self.count[dataloader_idx] += 1
            value = torch.zeros((len(x_hq_hat),))
        elif self.metric in ["niqe", "clipiqa", "musiq"]:
            value = self.evaluater(x_hq_hat)
            self.count[dataloader_idx] += bs
        elif self.metric == "save":
            for img in x_hq_hat:
                save_image(img, os.path.join(self.out_dir, str(self.count[dataloader_idx]) + ".png"))
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
        elif self.metric == "fid-f":
            final_value = self.evaluater(self.out_dir, dataset_name="FFHQ",
                                         dataset_res=self.image_size,
                                         dataset_split="trainval70k", verbose=False)
            shutil.rmtree(self.out_dir, ignore_errors=True)
            os.makedirs(self.out_dir, exist_ok=True)
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
