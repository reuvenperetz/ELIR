from ELIR.datasets.dataset import BasicLoader
from torch.utils.data import DataLoader
import rawpy
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from torch.utils.data import Dataset
import os
import glob
from PIL import Image

from ELIR.training.patch_tracker import PatchTracker


class SIDDataset(Dataset):
    def __init__(self, image_folder, patch_size):
        super(SIDDataset, self).__init__()
        self.lq_images_path = self.get_file_path(image_folder, "lq")
        self.hq_images_path = self.get_file_path(image_folder, "hq")
        self.transform_LQ, self.transform_HQ = self.preprocess(patch_size)
        self.debug_count = 0


    def get_file_path(self, image_folder, q):
        images_path = []
        image_folder_split = os.path.join(image_folder, q)
        for file in sorted(glob.glob(os.path.join(image_folder_split,"*.ARW"))):
            images_path.append(file)
        return images_path

    def __len__(self):
        return len(self.hq_images_path)

    def preprocess(self, patch_size):
        transform_LQ = v2.Compose([
            v2.ToTensor()])
        transform_HQ = v2.Compose([
            v2.ToTensor()])
        return transform_LQ, transform_HQ

    def __getitem__(self, index, save_debug=False, debug_count=10):
        img_path = self.lq_images_path[index]
        with rawpy.imread(img_path) as raw:
            classic_pp = raw.postprocess()
            if save_debug and self.debug_count < debug_count:
                minimal_pp = raw.postprocess(gamma=(1, 1), no_auto_bright=True, use_camera_wb=False)
                partial_pp = raw.postprocess(gamma=(1, 1), no_auto_bright=True, bright=100.0, use_camera_wb=True)
                fig, axes = plt.subplots(1, 3, figsize=(20, 7))
                titles = ['No Postprocess', 'Partial PP', 'Classic PP']
                images = [minimal_pp, partial_pp, classic_pp]
                for ax, img, title in zip(axes, images, titles):
                    ax.imshow(img)
                    ax.set_title(title)
                    ax.axis('off')
                plt.tight_layout()
                fig.savefig(f'lq_comparison_{index}.png')
                self.debug_count += 1

        img = Image.fromarray(classic_pp)
        img_LQ = self.transform_LQ(img)

        img_path = self.hq_images_path[index]
        with rawpy.imread(img_path) as raw:
            classic_pp = raw.postprocess()
        img = Image.fromarray(classic_pp)
        img_HQ = self.transform_HQ(img)

        return img_LQ, img_HQ



class SIDPatchesDataset(Dataset):
    def __init__(self, image_folder, patch_size):
        super().__init__()

        self.patch_size = patch_size
        self.lq_images_path = self.get_file_path(image_folder, "lq")
        self.hq_images_path = self.get_file_path(image_folder, "hq")

        assert len(self.lq_images_path) == len(self.hq_images_path), \
            "LQ and HQ image count mismatch"

        self.transform_LQ, self.transform_HQ = self.preprocess()

        # Precompute all patch indices
        self.patches = self._compute_patches()


    def get_file_path(self, image_folder, q):
        image_folder_split = os.path.join(image_folder, q)
        return sorted(glob.glob(os.path.join(image_folder_split, "*.ARW")))

    def preprocess(self):
        transform = v2.Compose([
            v2.ToTensor()
        ])
        return transform, transform

    def _compute_patches(self):
        """
        Returns a list of tuples:
        (image_index, top, left)
        """
        patches = []

        for img_idx, img_path in enumerate(self.lq_images_path):
            with rawpy.imread(img_path) as raw:
                img = raw.postprocess()

            H, W, _ = img.shape
            P = self.patch_size

            ys = list(range(0, H - P + 1, P))
            xs = list(range(0, W - P + 1, P))

            # Handle borders (minimal overlap only if needed)
            if ys[-1] != H - P:
                ys.append(H - P)
            if xs[-1] != W - P:
                xs.append(W - P)

            for y in ys:
                for x in xs:
                    patches.append((img_idx, y, x))

        return patches

    def __len__(self):
        return len(self.patches)

    def _load_image(self, path):
        with rawpy.imread(path) as raw:
            img = raw.postprocess()
        return Image.fromarray(img)

    def __getitem__(self, index):
        img_idx, y, x = self.patches[index]
        P = self.patch_size

        # Load LQ and HQ
        img_LQ = self._load_image(self.lq_images_path[img_idx])
        img_HQ = self._load_image(self.hq_images_path[img_idx])

        # Crop aligned patches
        patch_LQ = img_LQ.crop((x, y, x + P, y + P))
        patch_HQ = img_HQ.crop((x, y, x + P, y + P))

        # Transform
        patch_LQ = self.transform_LQ(patch_LQ)
        patch_HQ = self.transform_HQ(patch_HQ)

        return patch_LQ, patch_HQ


class SID(BasicLoader):
    def __init__(self):
        super().__init__()

    def create_loaders(self, dataset_params):
        path = dataset_params.get("path")
        batch_size = dataset_params.get("batch_size", 32)
        num_workers = dataset_params.get("num_workers", 4)
        patch_size = dataset_params.get("patch_size", 64)

        # Dataset
        dataset = SIDPatchesDataset(path, patch_size)
        # Loader
        loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True,
                             drop_last=False)

        return loader
