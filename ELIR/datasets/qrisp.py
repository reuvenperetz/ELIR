from torchvision.transforms import v2
from ELIR.datasets.dataset import BasicLoader
from torch.utils.data import DataLoader, Dataset
import os
import glob
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch



class QRISPDataset(Dataset):
    def __init__(self, image_folder, patch_size):
        super(QRISPDataset, self).__init__()
        self.hq_images_path = self.get_file_paths(image_folder, "hq")
        self.lq_images_path = self.get_file_paths(image_folder, "lq")

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
        # Load Images
        lq_img = Image.open(self.lq_images_path[index]).convert("RGB")
        hq_img = Image.open(self.hq_images_path[index]).convert("RGB")

        img_LQ = self.transform_LQ(lq_img).unsqueeze(0) # (1, C, H, W)
        img_HQ = self.transform_HQ(hq_img)

        # Interpolate LQ to HQ size
        img_LQ = F.interpolate(img_LQ, size=(1080, 1920), mode="bicubic").squeeze(0)

        # Define 4 corner coordinates (y, x) for 1024x1024 patches
        coords = [
            (0, 0),                    # Top-Left
            (0, 1920 - 1024),          # Top-Right
            (1080 - 1024, 0),          # Bottom-Left
            (1080 - 1024, 1920 - 1024) # Bottom-Right
        ]

        patches_LQ = []
        patches_HQ = []

        for y, x in coords:
            patches_LQ.append(img_LQ[:, y:y+1024, x:x+1024])
            patches_HQ.append(img_HQ[:, y:y+1024, x:x+1024])

        # Return as (4, C, 1024, 1024)
        return torch.stack(patches_LQ), torch.stack(patches_HQ)




def collate_patches(batch):
    # batch is [(patches_x, patches_y), ...]
    x, y = zip(*batch)
    # Stack everything into (BatchSize * 4, C, 1024, 1024)
    return torch.cat(x, dim=0), torch.cat(y, dim=0)


import matplotlib.pyplot as plt

def visualize_patches(x_tuple, y_tuple):
    # Extract tensors from tuples
    # Assuming x_tuple = (tensor,) and y_tuple = (tensor,)
    batch_x = x_tuple[0]
    batch_y = y_tuple[0]

    num_patches = batch_x.shape[0] # 4
    fig, axes = plt.subplots(num_patches, 2, figsize=(10, 5 * num_patches))

    for i in range(num_patches):
        # Convert (C, H, W) to (H, W, C) for plotting and move to CPU
        img_x = batch_x[i].permute(1, 2, 0).cpu().detach().numpy()
        img_y = batch_y[i].permute(1, 2, 0).cpu().detach().numpy()

        # Plot X (Low Res / Input)
        axes[i, 0].imshow(img_x)
        axes[i, 0].set_title(f"Patch {i+1}: Input (X)")
        axes[i, 0].axis('off')

        # Plot Y (High Res / Target)
        axes[i, 1].imshow(img_y)
        axes[i, 1].set_title(f"Patch {i+1}: Target (Y)")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

# Usage:
# visualize_patches(out_x, out_y)

class QRISP(BasicLoader):
    def __init__(self):
        super().__init__()

    def create_loaders(self, dataset_params):
        path = dataset_params.get("path")
        batch_size = dataset_params.get("batch_size", 8) # Note: Effective batch will be batch_size * 4
        num_workers = dataset_params.get("num_workers", 4)
        patch_size = dataset_params.get("patch_size", 64)

        dataset = QRISPDataset(path, patch_size)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_patches
        )

        return loader
