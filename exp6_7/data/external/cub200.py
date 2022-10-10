# https://github.com/TDeVries/cub2011_dataset

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm


class Cub2011(Dataset):
    base_folder = 'exp_datasets/CUB_200_2011/images'
    url = 'https://data.caltech.edu/tindfiles/serve/1239ea37-e132-42ee-8c09-c383bb54e7ff/'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'
    class_mapping = torch.randperm(200)

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True, empty=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train
        self.train_data_file = Path('train_cub200.pik')
        self.test_data_file = Path('test_cub200.pik')
        self.trans_train_data_file = Path('transformed_train_cub200.pik')
        self.trans_test_data_file = Path('transformed_test_cub200.pik')

        if not empty:
            if download:
                self._download()

            if not self._check_integrity():
                raise RuntimeError('Dataset not found or corrupted.' +
                                   ' You can use download=True to download it')

    @classmethod
    def random_order(cls):
        cls.class_mapping = torch.randperm(200)

    def _load_metadata(self):

        # load_path = None
        # if self.transform is not None and self.train:
        #     load_path = self.trans_train_data_file
        # elif self.transform is not None and not self.train:
        #     load_path = self.trans_test_data_file
        # elif self.transform is None and self.train:
        #     load_path = self.train_data_file
        # elif self.transform is None and not self.train:
        #     load_path = self.test_data_file
        #
        # if load_path.exists():
        #     with open(str(load_path), 'rb') as file:
        #         saved_data = pickle.load(file)
        #     self.data = saved_data['data']
        #     self.targets = saved_data['targets']
        #     self.classes = saved_data['classes']
        # else:
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

        targets = []
        images = []
        for index, sample in tqdm(self.data.iterrows()):
            path = os.path.join(self.base_folder, sample.filepath)
            target = sample.target - 1  # Targets start at 1 by default, so shift to 0
            img = self.loader(path)

            input_size = (224, 224)

            self.initial_transforms = transforms.Compose([
                transforms.Resize(input_size)
            ])

            img = self.initial_transforms(img)

            if self.transform is not None:
                img = self.transform(img)

            targets.append(target)
            images.append(np.array(img))

        self.targets = torch.tensor(targets)
        self.data = torch.tensor(np.stack(images))
        self.classes = list(range(0, 200))
        self.targets = self.class_mapping[self.targets]

        # data_to_save = dict(data=self.data, targets=self.targets, classes=self.classes)

        # with open(str(load_path), 'wb') as file:
        #     pickle.dump(data_to_save, file, protocol=4)

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        # for index, row in self.data.iterrows():
        #     filepath = os.path.join(self.base_folder, row.filepath)
        #     if not os.path.isfile(filepath):
        #         print(filepath)
        #         return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return self.data[idx], self.targets[idx]
