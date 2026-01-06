from ELIR.datasets.dataset import BasicLoader
from torch.utils.data import DataLoader
import rawpy
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from torch.utils.data import Dataset
import os
import glob
import numpy as np
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

class SIDFullImageDataset(Dataset):
    """
    Dataset that returns full images without patching.
    Used for validation/testing.
    """
    def __init__(self, image_folder):
        super().__init__()

        self.lq_images_path = self.get_file_path(image_folder, "lq")
        self.hq_images_path = self.get_file_path(image_folder, "hq")

        assert len(self.lq_images_path) == len(self.hq_images_path), \
            "LQ and HQ image count mismatch"

        self.transform = v2.Compose([v2.ToTensor()])

    def get_file_path(self, image_folder, q):
        image_folder_split = os.path.join(image_folder, q)
        return sorted(glob.glob(os.path.join(image_folder_split, "*.ARW")))

    def __len__(self):
        return len(self.lq_images_path)

    def _load_image(self, path):
        with rawpy.imread(path) as raw:
            img = raw.postprocess()
        return Image.fromarray(img)

    def __getitem__(self, index):
        img_LQ = self._load_image(self.lq_images_path[index])
        img_HQ = self._load_image(self.hq_images_path[index])

        img_LQ = self.transform(img_LQ)
        img_HQ = self.transform(img_HQ)

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


class SIDRandomCropDataset(Dataset):
    """
    Dataset that returns one random crop per image with optional augmentations.
    Each epoch samples different random crops from the images.
    """
    def __init__(self, image_folder, patch_size, augment=True,
                 hflip=True, vflip=True, rotate=True):
        super().__init__()

        self.patch_size = patch_size
        self.augment = augment
        self.hflip = hflip
        self.vflip = vflip
        self.rotate = rotate

        self.lq_images_path = self.get_file_path(image_folder, "lq")
        self.hq_images_path = self.get_file_path(image_folder, "hq")

        assert len(self.lq_images_path) == len(self.hq_images_path), \
            "LQ and HQ image count mismatch"

        self.transform = v2.Compose([v2.ToTensor()])

    def get_file_path(self, image_folder, q):
        image_folder_split = os.path.join(image_folder, q)
        return sorted(glob.glob(os.path.join(image_folder_split, "*.ARW")))

    def __len__(self):
        return len(self.lq_images_path)

    def _load_image(self, path):
        with rawpy.imread(path) as raw:
            img = raw.postprocess()
        return img  # Return numpy array for easier augmentation

    def _random_crop(self, img_lq, img_hq):
        """Extract the same random crop from both LQ and HQ images."""
        H, W, _ = img_lq.shape
        P = self.patch_size

        if H >= P and W >= P:
            y = np.random.randint(0, H - P + 1)
            x = np.random.randint(0, W - P + 1)
            patch_lq = img_lq[y:y+P, x:x+P]
            patch_hq = img_hq[y:y+P, x:x+P]
        else:
            # If image is smaller than patch_size, resize
            patch_lq = np.array(Image.fromarray(img_lq).resize((P, P), Image.Resampling.BICUBIC))
            patch_hq = np.array(Image.fromarray(img_hq).resize((P, P), Image.Resampling.BICUBIC))

        return patch_lq, patch_hq

    def _augment_pair(self, lq, hq):
        """Apply the same random augmentations to both LQ and HQ patches."""
        # Random horizontal flip
        if self.hflip and np.random.rand() < 0.5:
            lq = np.fliplr(lq).copy()
            hq = np.fliplr(hq).copy()

        # Random vertical flip
        if self.vflip and np.random.rand() < 0.5:
            lq = np.flipud(lq).copy()
            hq = np.flipud(hq).copy()

        # Random 90-degree rotation (0, 90, 180, or 270 degrees)
        if self.rotate:
            k = np.random.randint(0, 4)
            if k > 0:
                lq = np.rot90(lq, k).copy()
                hq = np.rot90(hq, k).copy()

        return lq, hq

    def __getitem__(self, index):
        # Load both images
        img_LQ = self._load_image(self.lq_images_path[index])
        img_HQ = self._load_image(self.hq_images_path[index])

        # Random crop (same location for both)
        patch_LQ, patch_HQ = self._random_crop(img_LQ, img_HQ)

        # Apply augmentations
        if self.augment:
            patch_LQ, patch_HQ = self._augment_pair(patch_LQ, patch_HQ)

        # Ensure correct shape (H, W, C)
        assert patch_LQ.shape == (self.patch_size, self.patch_size, 3), \
            f"Unexpected LQ shape: {patch_LQ.shape}"
        assert patch_HQ.shape == (self.patch_size, self.patch_size, 3), \
            f"Unexpected HQ shape: {patch_HQ.shape}"

        # Convert to PIL and transform to tensor
        patch_LQ = self.transform(Image.fromarray(patch_LQ))
        patch_HQ = self.transform(Image.fromarray(patch_HQ))

        return patch_LQ, patch_HQ


class SID(BasicLoader):
    def __init__(self):
        super().__init__()

    def create_loaders(self, dataset_params):
        path = dataset_params.get("path")
        batch_size = dataset_params.get("batch_size", 32)
        num_workers = dataset_params.get("num_workers", 4)
        patch_size = dataset_params.get("patch_size", 64)
        shuffle = dataset_params.get("shuffle", True)

        # Random crop mode options (default: False = use all patches)
        full_image = dataset_params.get("full_image", False)
        random_crop = dataset_params.get("random_crop", False)
        augment = dataset_params.get("augment", True)
        hflip = dataset_params.get("hflip", True)
        vflip = dataset_params.get("vflip", True)
        rotate = dataset_params.get("rotate", True)

        # Choose dataset based on mode
        if full_image:
            dataset = SIDFullImageDataset(path)
        elif random_crop:
            dataset = SIDRandomCropDataset(
                path, patch_size,
                augment=augment,
                hflip=hflip,
                vflip=vflip,
                rotate=rotate
            )
        else:
            dataset = SIDPatchesDataset(path, patch_size)

        # Loader - use batch_size=1 for full images (different sizes)
        loader = DataLoader(dataset,
                            batch_size=1 if full_image else batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=not full_image)

        return loader
