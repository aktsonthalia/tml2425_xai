# CUB dataset class from https://github.com/TDeVries/cub2011_dataset
# slightly adapted
import os
import pandas as pd
import torch
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torchvision.datasets import VisionDataset


class Cub2011(VisionDataset):
    base_folder = 'CUB_200_2011/images'
    url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        self.data = None
        self.targets = None

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        meta_data = images.merge(image_class_labels, on='img_id')
        self.meta_data = meta_data.merge(train_test_split, on='img_id')

        if self.train:
            self.meta_data = self.meta_data[self.meta_data.is_training_img == 1]
        else:
            self.meta_data = self.meta_data[self.meta_data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.meta_data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        if self.data is None:
            sample = self.meta_data.iloc[idx]
            path = os.path.join(self.root, self.base_folder, sample.filepath)
            target = sample.target - 1  # Targets start at 1 by default, so shift to 0
            img = self.loader(path)

            if self.transform is not None:
                img = self.transform(img)
        else: 
            img = self.data[idx]
            target = self.targets[idx]

        return img, target