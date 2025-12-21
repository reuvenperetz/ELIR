from torchvision.transforms import v2
from ELIR.datasets.dataset import BasicLoader
from torch.utils.data import DataLoader, Dataset
import os
import glob
from PIL import Image
import torch.nn.functional as F


class QRISPDataset(Dataset):
    def __init__(self, image_folder, patch_size):
        super(QRISPDataset, self).__init__()
        self.hq_images_path = self.get_file_paths(image_folder, "1080p/Native/0000")
        self.lq_images_path = self.get_file_paths(image_folder, "540p/Native/0000")

        assert len(self.lq_images_path) == len(self.hq_images_path), \
            "Mismatch between LQ and GT image counts"

        self.transform_LQ, self.transform_HQ = self.preprocess(patch_size)

    def get_file_paths(self, image_folder, subfolder):
        folder = os.path.join(image_folder, subfolder)
        return sorted(glob.glob(os.path.join(folder, "*.png")))

    def __len__(self):
        return len(self.hq_images_path)

    def preprocess(self, patch_size):
        print(f"Warning: Ignoring patch size")
        transform = v2.Compose([
            v2.ToTensor()
        ])
        return transform, transform

    def __getitem__(self, index):
        # LQ
        lq_path = self.lq_images_path[index]
        lq_img = Image.open(lq_path).convert("RGB")
        img_LQ = self.transform_LQ(lq_img)

        # HQ
        hq_path = self.hq_images_path[index]
        hq_img = Image.open(hq_path).convert("RGB")
        img_HQ = self.transform_HQ(hq_img)

        x = img_LQ[:, 0:512, 0:512]

        # x: (3, 512, 512)
        x = x.unsqueeze(0)  # -> (1, 3, 512, 512)

        x_up = F.interpolate(
            x,
            size=(1024, 1024),
            mode="bicubic",
            align_corners=False
        )

        img_LQ = x_up.squeeze(0)  # -> (3, 1024, 1024)

        img_HQ = img_HQ[:, 0:1024, 0:1024]

        return img_LQ, img_HQ


class QRISP(BasicLoader):
    def __init__(self):
        super().__init__()

    def create_loaders(self, dataset_params):
        path = dataset_params.get("path")
        batch_size = dataset_params.get("batch_size", 32)
        num_workers = dataset_params.get("num_workers", 4)
        patch_size = dataset_params.get("patch_size", 64)

        dataset = QRISPDataset(path, patch_size)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )

        return loader
