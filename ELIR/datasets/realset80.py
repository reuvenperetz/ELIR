import glob
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset
from ELIR.datasets.dataset import BasicLoader
import os
from PIL import Image



class RealSet80Dataset(Dataset):
    def __init__(self, image_folder):
        super(RealSet80Dataset, self).__init__()
        self.images_path = self.get_file_path(image_folder)
        self.transform = self.preprocess()

    def preprocess(self):
        transform = v2.Compose([
            v2.ToTensor()])
        return transform

    def get_file_path(self, image_folder):
        images_path = []
        for file in sorted(glob.glob(os.path.join(image_folder,"*.*"))):
            images_path.append(os.path.join(image_folder, file))
        return images_path

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        img_path = self.images_path[index]
        img = Image.open(img_path).convert("RGB")
        lq = self.transform(img)
        return lq, lq


class RealSet80(BasicLoader):
    def __init__(self):
        super().__init__()

    def create_loaders(self, dataset_params):
        path = dataset_params.get("path")
        batch_size = dataset_params.get("batch_size", 32)
        num_workers = dataset_params.get("num_workers", 4)

        # Datasets
        dataset = RealSet80Dataset(path)

        # Loaders
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=False)

        return loader
