import gc
import math
from collections import OrderedDict
from copy import deepcopy

import PIL
import numpy as np
import sklearn
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from config.parser import Configuration
from data.feature_dataset import FeaturesDataset
from data.samplers import UniformBinarySampler
from model.transfer_learning import NetworkTypes, BackendClass


class BaseDataset:
    def __init__(self, config: Configuration, output_result_path, device, model, dataset_class):

        self.network_type = NetworkTypes.get_network_type(config.get_param_value('network_type'))
        self.dataset_name = config.get_param_value('dataset_name')
        self.dataset_path = config.get_param_value('dataset_path')
        self.download = config.get_param_value('download')
        self.train_batch_size = config.get_param_value('train_batch_size')
        self.test_batch_size = config.get_param_value('test_batch_size')
        self.output_path = output_result_path
        self.device = device

        self.dataset_class = dataset_class

        dataset_name = config.get_param_value('dataset_name')
        if dataset_name == 'CUB200':
            self.num_classes = 200
        else:
            raise NotImplementedError
        self.num_channels = 3
        self.classes = list(range(self.num_classes))

        # self.mean = sample_data.mean()
        # self.std = sample_data.std()
        self.data_augmentation = config.get_param_value('data_augmentation/enabled', False)
        self.validate = config.get_param_value('validation/enabled', False) or False
        if self.validate:
            self.validation_split = config.get_param_value('validation/split_per_class')

        network_class = BackendClass.get_backend_name(config.get_param_value('backend_name'))
        if network_class == BackendClass.INCEPTION:
            input_size = (299, 299)
        else:
            input_size = (224, 224)

        if self.dataset_name == "FashionMNIST":
            mean = np.array([0.485, 0.456, 0.406]).mean().tolist()
            std = np.array([0.229, 0.224, 0.225]).mean().tolist()
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        self.test_data_transform = transforms.Compose([
            transforms.Resize(input_size), transforms.ToTensor(),
            transforms.Normalize(mean=mean,
                                 std=std)
        ])

        self.data_transform = transforms.Compose([
            transforms.Resize(input_size), transforms.ToTensor(),
            transforms.Normalize(mean=mean,
                                 std=std)
        ])

        if self.data_augmentation:
            augmentation_probability = config.get_param_value('data_augmentation/augmentation_probability')

            brightness = config.get_param_value('data_augmentation/color_jitter/brightness')
            contrast = config.get_param_value('data_augmentation/color_jitter/contrast')
            degrees = config.get_param_value('data_augmentation/random_affine/degrees')
            scale_min = config.get_param_value('data_augmentation/random_affine/scale_min')
            scale_max = config.get_param_value('data_augmentation/random_affine/scale_max')

            transform_list = torch.nn.ModuleList([])
            if config.get_param_value('data_augmentation/random_crop'):
                transform_list.append(transforms.RandomCrop(input_size))
            if config.get_param_value('data_augmentation/color_jitter'):
                transform_list.append(transforms.ColorJitter(brightness=brightness, contrast=contrast))
            if config.get_param_value('data_augmentation/random_affine'):
                transform_list.append(
                    transforms.RandomAffine(degrees=degrees, scale=(scale_min, scale_max), resample=PIL.Image.BICUBIC,
                                            fillcolor=0))
            if config.get_param_value('data_augmentation/random_horizontal_flip'):
                transform_list.append(transforms.RandomHorizontalFlip())

            self.data_transform = transforms.Compose([
                transforms.Resize(input_size),
                transforms.RandomApply(transform_list, p=augmentation_probability),
                transforms.ToTensor(),
                transforms.Normalize((self.mean / 255,), (self.std / 255,))
            ])

        self.reduction = None

        # current_model = model.cnn
        # class_features = []
        self.features_dataset = self.dataset_class(root=self.dataset_path, download=self.download, train=True,
                                                   transform=self.test_data_transform)

        self.targets = torch.tensor(self.features_dataset.targets, dtype=torch.float)

        # self.features_dataset.data = torch.tensor(self.features_dataset.data, dtype=torch.float32)
        # self.features_dataset.targets = torch.tensor(self.features_dataset.targets, dtype=torch.float32)

        # features_data_loader = DataLoader(self.features_dataset , batch_size=self.train_batch_size, drop_last=False,
        #                                   shuffle=False)
        # for images, labels in tqdm(features_data_loader):
        #     images = images.to(self.device)
        #     if images.shape[1] == 1:
        #         images = images.repeat(1, 3, 1, 1)
        #
        #     class_features.append(current_model(images).cpu())
        #
        # num_elem = torch.tensor(self.features_dataset.data).shape[0]
        # class_features = torch.vstack(class_features).view(num_elem, -1)
        # self.features_dataset = FeaturesDataset(torch.tensor(self.features_dataset.data), torch.tensor(self.features_dataset.targets))
        # self.images_dataset = images_dataset
        gc.collect()

    def gen_train_val_loader(self, images_no_c, images_c, labels_no_c, labels_c, labels_no_c_org, labels_c_org):
        val_data_loader = None
        if self.validate:
            classes_no_c = torch.unique(labels_no_c_org)
            classes_c = torch.unique(labels_c_org)

            if classes_no_c.shape[0] > 0:
                train_idx_no_c = []
                val_idx_no_c = []
                for class_no_c in classes_no_c:
                    current_c_idx = torch.where(class_no_c == labels_no_c_org)[0]
                    val_items = int(current_c_idx.shape[0] * self.validation_split)
                    val_items_idx = torch.tensor(np.linspace(1, current_c_idx.shape[0] - 2, val_items)).long()
                    mask = torch.zeros_like(current_c_idx, dtype=torch.bool)
                    mask[val_items_idx] = True
                    val_idx_no_c.append(current_c_idx[mask])
                    train_idx_no_c.append(current_c_idx[~mask])
                train_idx_no_c = torch.cat(train_idx_no_c)
                val_idx_no_c = torch.cat(val_idx_no_c)

                train_images_no_c = images_no_c[train_idx_no_c]
                train_labels_no_c = labels_no_c[train_idx_no_c]
                val_images_no_c = images_no_c[val_idx_no_c]
                val_labels_no_c = labels_no_c[val_idx_no_c]
            else:  # empty no 0 classes
                train_images_no_c = images_no_c
                train_labels_no_c = labels_no_c
                val_images_no_c = images_no_c
                val_labels_no_c = labels_no_c

            train_idx_c = []
            val_idx_c = []
            for class_c in classes_c:
                current_c_idx = torch.where(class_c == labels_c_org)[0]
                val_items = int(current_c_idx.shape[0] * self.validation_split)
                val_items_idx = torch.tensor(np.linspace(1, current_c_idx.shape[0] - 2, val_items)).long()
                mask = torch.zeros_like(current_c_idx, dtype=torch.bool)
                mask[val_items_idx] = True
                val_idx_c.append(current_c_idx[mask])
                train_idx_c.append(current_c_idx[~mask])
            train_idx_c = torch.cat(train_idx_c)
            val_idx_c = torch.cat(val_idx_c)

            train_images_c = images_c[train_idx_c]
            train_labels_c = labels_c[train_idx_c]
            val_images_c = images_c[val_idx_c]
            val_labels_c = labels_c[val_idx_c]

            val_new_images = torch.cat((val_images_no_c, val_images_c))
            val_new_labels = torch.cat((val_labels_no_c, val_labels_c))
            dataset_val = FeaturesDataset(val_new_images, val_new_labels)
            val_data_loader = DataLoader(dataset_val, batch_size=self.train_batch_size, shuffle=True)


        else:
            train_images_c = images_c
            train_labels_c = labels_c
            train_images_no_c = images_no_c
            train_labels_no_c = labels_no_c

        train_new_images = torch.cat((train_images_no_c, train_images_c))
        train_new_labels = torch.cat((train_labels_no_c, train_labels_c))

        dataset_train = self.dataset_class(root=self.dataset_path, download=self.download, train=True,
                                           transform=self.test_data_transform)

        dataset_train.data = torch.tensor(train_new_images, dtype=torch.float32)
        dataset_train.targets = torch.tensor(train_new_labels, dtype=torch.long)
        uniform_sampler = UniformBinarySampler(train_new_labels, self.train_batch_size)
        train_data_loader = DataLoader(dataset_train, batch_sampler=uniform_sampler)

        return train_data_loader, val_data_loader

    def get_train_loader(self, i_class: int = -1, model=None):
        raise NotImplementedError

    def get_test_loader(self):
        dataset_test = self.dataset_class(root=self.dataset_path, download=self.download, train=False,
                                          transform=self.test_data_transform)
        test_data_loader = DataLoader(dataset_test, batch_size=self.test_batch_size, drop_last=False)

        return test_data_loader

    def get_eval_loader(self):
        dataset_train = self.dataset_class(root=self.dataset_path, download=self.download, train=True,
                                           transform=self.test_data_transform)
        eval_train_data_loader = DataLoader(dataset_train, batch_size=self.test_batch_size, drop_last=False)

        return eval_train_data_loader

    def __str__(self):
        info = 'Dataset Information\n'
        info += '  Name: {} - {}\n'.format(self.dataset_name, self.sub_name)
        info += '  Classes: {}\n'.format(self.classes)
        info += '  Channels: {}\n'.format(self.num_channels)
        return info


class NormalDataset(BaseDataset):
    def __init__(self, config: Configuration, output_result_path, device, model, dataset_class):
        super().__init__(config, output_result_path, device, model, dataset_class)
        self.equalize_binary_class_data = config.get_param_value('equalize_binary_class_data', False)
        self.sub_name = "NormalDataset"

    def get_train_loader(self, i_class: int = -1, model=None):
        print('  Creating dataset for classifier: {}'.format(i_class))

        dataset_train = self.features_dataset
        if self.network_type == NetworkTypes.BINARY_ENSAMBLE or self.network_type == NetworkTypes.BINARY_ENSAMBLE_M or \
                self.network_type == NetworkTypes.BINARY_ENSAMBLE_B:

            if i_class < 0 or i_class >= self.num_classes:
                raise ValueError('Incorrect class: {}'.format(i_class))

            train_labels = dataset_train.targets
            if i_class == 0:
                idx_no_c = train_labels == 1
            else:
                idx_no_c = train_labels < i_class

            images_no_c = dataset_train.data[idx_no_c]
            n_images_no_c = images_no_c.shape[0]
            labels_no_c_originals = torch.unique(train_labels[idx_no_c]).numpy()
            labels_no_c = torch.zeros(n_images_no_c, dtype=torch.float)

            # -- Class c = 1: Replicate several times the data of the "old_class"
            idx_c = train_labels == i_class

            images = dataset_train.data[idx_c]
            n_images_c = images.shape[0]
            images_c = deepcopy(images)

            if self.equalize_binary_class_data:  # if true in config then data are copies until it matches class 0 size
                n_copies = math.floor(n_images_no_c / n_images_c)

                for i in range(n_copies - 1):
                    images_c = torch.cat((images_c, images))

                n_repeated_images = n_images_no_c - n_copies * n_images_c
                random_idx = torch.randint(0, n_images_c, (n_repeated_images,))
                images_c = torch.cat((images_c, images[random_idx]), 0)

            n_images_c = images_c.shape[0]
            labels_c_originals = torch.unique(train_labels[idx_c]).numpy()
            labels_c = torch.ones(n_images_c, dtype=torch.float)
            print('    Class 0 created from original labels: {}. Number of images: {}'.format(labels_no_c_originals,
                                                                                              n_images_no_c))
            print('    Class 1 created from original labels: {}. Number of images: {}'.format(labels_c_originals,
                                                                                              n_images_c))

            return self.gen_train_val_loader(images_no_c, images_c, labels_no_c, labels_c, train_labels[idx_no_c],
                                             train_labels[idx_c])
        else:
            return self.gen_train_val_loader(torch.tensor([]), dataset_train.data, torch.tensor([]),
                                             dataset_train.targets, torch.tensor([]), dataset_train.targets)


class ResamplingDataset(BaseDataset):
    def __init__(self, config: Configuration, output_result_path, device, model, dataset_class):
        super().__init__(config, output_result_path, device, model, dataset_class)
        self.resampling_samples = config.get_param_value('resampling/samples')
        self.resample_class_c = config.get_param_value('resampling/resample_class_c')
        self.resampling_fixed_r_enabled = config.get_param_value('resampling/fixed_r_enabled', False)
        self.resample_fixed_r = config.get_param_value('resampling/fixed_r', False)
        self.resampling_current_samples = config.get_param_value('resampling/current_samples')
        if self.resampling_samples % 2:
            raise ValueError('Number of resampling samples must be even')

        self.sub_name = "ResamplingDataset"
        labels = torch.unique(torch.tensor(self.targets))

        self.class_indexes = OrderedDict()
        for label in labels:
            label = label.item()
            self.class_indexes[label] = torch.where(torch.tensor(self.targets) == label)[0]

    def get_train_loader(self, i_class: int = -1, model=None, after_last_classifier=False):
        dataset_train = self.features_dataset

        if i_class == 0:
            prev_classes = [1]  # only used for BE training
        else:
            prev_classes = range(i_class)

        imgs_per_prev_class = math.ceil(self.resampling_samples / len(prev_classes))
        total_images_no_c = len(prev_classes) * imgs_per_prev_class
        leftover_imgs = int(
            total_images_no_c - self.resampling_samples)

        self.reduction = imgs_per_prev_class / self.resampling_samples

        train_data = dataset_train.data
        train_labels = dataset_train.targets

        # Create binary mask that specifies in which class to remove an image
        ones_classes = [0] * (len(prev_classes) - leftover_imgs)
        zeros_classes = [1] * leftover_imgs
        mask = torch.tensor(ones_classes + zeros_classes)
        shuffled_mask = mask[torch.randperm(len(mask))]

        images_no_c = []
        labels_no_c_org = []
        for i, prev_class in enumerate(prev_classes):
            current_class_num_images = imgs_per_prev_class - shuffled_mask[i]
            class_idx = self.class_indexes[prev_class]

            full_repeats = math.floor(current_class_num_images / class_idx.shape[0])
            # extra images to be added but less than one full repeat
            leftover_images = current_class_num_images - (full_repeats * class_idx.shape[0])

            shuffled_images_indexes = torch.randperm(class_idx.shape[0])
            leftover_shuffled_images_indexes = shuffled_images_indexes[0:leftover_images]
            class_idx = torch.cat((full_repeats * [class_idx]) + [class_idx[leftover_shuffled_images_indexes]])

            # save used idx in order to use the same images in the next classifier
            self.class_indexes[prev_class] = class_idx

            images_no_c.append(train_data[class_idx])
            labels_no_c_org.append(train_labels[class_idx])
            print('Number of images for class {}: {}, unique labels: {}'.format(prev_class, class_idx.shape[0],
                                                                                np.unique(train_labels[class_idx])))

        images_no_c = torch.vstack(images_no_c)
        labels_no_c_org = torch.cat(labels_no_c_org)
        n_images_no_c = images_no_c.shape[0]
        if not after_last_classifier:
            class_idx_c = self.class_indexes[i_class]
            n_images_c = self.class_indexes[0].shape[0] if self.resample_class_c else self.resampling_current_samples
            full_repeats = math.floor(n_images_c / class_idx_c.shape[0])
            leftover_images = n_images_c - (full_repeats * class_idx_c.shape[0])
            shuffled_images_indexes = torch.randperm(class_idx_c.shape[0])
            leftover_shuffled_images_indexes = shuffled_images_indexes[0:leftover_images]
            class_idx = torch.cat((full_repeats * [class_idx_c]) + [class_idx_c[leftover_shuffled_images_indexes]])
            self.class_indexes[i_class] = class_idx
            images_c = train_data[class_idx]
            labels_c_org = train_labels[class_idx]
            print('Number of images for class {}: {}, unique labels: {}'.format(i_class, class_idx.shape[0],
                                                                                np.unique(train_labels[class_idx])))

            n_images_c = images_c.shape[0]
            n_images_no_c = images_no_c.shape[0]
            labels_c = torch.ones(n_images_c, dtype=torch.float)
            labels_no_c = torch.zeros(n_images_no_c, dtype=torch.float)

            print('Number of images for class {}: {}'.format(0, n_images_no_c))
            print('Number of images for class {}: {}'.format(1, n_images_c))

            return self.gen_train_val_loader(images_no_c, images_c, labels_no_c, labels_c, labels_no_c_org,
                                             labels_c_org)

    def get_eval_loader(self):

        dataset_train = self.dataset_class(root=self.dataset_path, download=self.download, train=True,
                                           transform=self.test_data_transform)

        eval_train_data_loader = DataLoader(dataset_train, batch_size=self.test_batch_size, drop_last=False)

        print('  Eval dataset made of {} images.'.format(torch.tensor(dataset_train.data).shape[0]))

        return eval_train_data_loader


class ResamplingDatasetFixedR(BaseDataset):
    def __init__(self, config: Configuration, output_result_path, device, model, dataset_class):
        super().__init__(config, output_result_path, device, model, dataset_class)
        self.resample_class_c = config.get_param_value('resampling/resample_class_c')
        self.resampling_fixed_r_enabled = config.get_param_value('resampling/fixed_r_enabled')
        self.fixed_r = config.get_param_value('resampling/fixed_r')

        self.sub_name = "ResamplingDataset - Fixed r of {}".format(self.fixed_r)
        labels = torch.unique(torch.tensor(self.targets))

        self.class_indexes = OrderedDict()
        self.reduced_class = {}
        for label in labels:
            label = label.item()
            self.class_indexes[label] = torch.where(torch.tensor(self.targets) == label)[0]
            self.reduced_class[label] = False

    def get_train_loader(self, i_class: int = -1, model=None):
        dataset_train = self.features_dataset

        if i_class < 0 or i_class >= self.num_classes:
            raise ValueError('Incorrect class: {}'.format(i_class))

        if i_class == 0:
            prev_classes = [1]  # only used for BE training
        else:
            prev_classes = range(i_class)

        train_data = dataset_train.data
        train_labels = dataset_train.targets

        images_no_c = []
        labels_no_c_org = []
        for i, prev_class in enumerate(prev_classes):
            class_idx = self.class_indexes[prev_class]

            if not self.reduced_class[prev_class]:
                if i_class != 1:
                    current_class_num_images = int(class_idx.shape[0] * self.fixed_r)
                    self.reduced_class[prev_class] = True
                else:
                    current_class_num_images = class_idx.shape[0]

                full_repeats = math.floor(current_class_num_images / class_idx.shape[0])
                # extra images to be added but less than one full repeat
                leftover_images = current_class_num_images - (full_repeats * class_idx.shape[0])

                shuffled_images_indexes = torch.randperm(class_idx.shape[0])
                leftover_shuffled_images_indexes = shuffled_images_indexes[0:leftover_images]

                class_idx = torch.cat((full_repeats * [class_idx]) + [class_idx[leftover_shuffled_images_indexes]])

                # save used idx in order to use the same images in the next classifier
                self.class_indexes[prev_class] = class_idx

            images_no_c.append(train_data[class_idx])
            labels_no_c_org.append(train_labels[class_idx])
            print('Number of images for class {}: {}, unique labels: {}'.format(prev_class, class_idx.shape[0],
                                                                                np.unique(train_labels[class_idx])))


        images_no_c = torch.vstack(images_no_c)
        labels_no_c_org = torch.cat(labels_no_c_org)

        class_idx_c = self.class_indexes[i_class]
        images_c = train_data[class_idx_c]
        labels_c_org = train_labels[class_idx_c]

        print('Number of images for class {}: {}, unique labels: {}'.format(i_class, class_idx_c.shape[0],
                                                                            np.unique(train_labels[class_idx_c])))

        n_images_c = images_c.shape[0]
        n_images_no_c = images_no_c.shape[0]
        labels_c = torch.ones(n_images_c, dtype=torch.float)
        labels_no_c = torch.zeros(n_images_no_c, dtype=torch.float)

        print('Number of images for class {}: {}'.format(0, n_images_no_c))
        print('Number of images for class {}: {}'.format(1, n_images_c))

        return self.gen_train_val_loader(images_no_c, images_c, labels_no_c, labels_c, labels_no_c_org,
                                         labels_c_org)

    def get_eval_loader(self):

        dataset_train = self.dataset_class(root=self.dataset_path, download=self.download, train=True,
                                           transform=self.test_data_transform)

        last_class_key = list(self.class_indexes.keys())[-1]
        shuffled_last_class = torch.randperm(self.class_indexes[last_class_key].shape[0])
        self.class_indexes[last_class_key] = self.class_indexes[last_class_key][
            shuffled_last_class[0:int(self.class_indexes[last_class_key].shape[0] * self.fixed_r)]]
        self.reduced_class[last_class_key] = True

        stacked_idx = torch.hstack([self.class_indexes[key] for key in list(self.class_indexes.keys())])
        new_labels = torch.tensor(dataset_train.targets)[stacked_idx]
        new_images = torch.tensor(dataset_train.data)[stacked_idx]

        if self.dataset_name.startswith('CIFAR'):
            dataset_train.data = new_images.numpy().astype(np.uint8)
            dataset_train.targets = new_labels.numpy().astype(np.uint8)
        else:
            dataset_train.data = new_images
            dataset_train.targets = new_labels

        eval_train_data_loader = DataLoader(dataset_train, batch_size=self.test_batch_size, drop_last=False)

        print('  Eval dataset made of {} images.'.format(stacked_idx.shape[0]))

        return eval_train_data_loader


class ConvexHullDataset(BaseDataset):
    def __init__(self, config: Configuration, output_result_path, device, model, dataset_class):
        super().__init__(config, output_result_path, device, model, dataset_class)

        self.convex_hull = config.get_param_value('convex_hull/enabled')

        self.fixed_r_enabled = config.get_param_value('convex_hull/fixed_r_enabled')
        self.other_class_equalization = config.get_param_value('convex_hull/other_class_equalization', False)
        self.use_dim_reduction = config.get_param_value('convex_hull/use_dim_reduction', False)
        self.use_dim_reduction_global = config.get_param_value('convex_hull/use_dim_reduction_global', False)

        self.use_near_centroid_data = config.get_param_value('convex_hull/use_near_centroid_data', False)
        if self.other_class_equalization:
            self.other_class_size = config.get_param_value('convex_hull/other_class_size')
            self.sub_name = "ConvexHullDataset - Equalized other class to {}".format(self.other_class_size)
        if self.fixed_r_enabled:
            self.fixed_r = config.get_param_value('convex_hull/fixed_r')
            self.sub_name = "ConvexHullDataset - Fixed R of {}".format(self.fixed_r)

        if self.use_dim_reduction or self.use_dim_reduction_global:
            self.dim_reduction_components = config.get_param_value('convex_hull/dim_reduction_components')

        if not self.fixed_r_enabled and not self.other_class_equalization:
            raise ValueError('Must select an r method.')

        if self.fixed_r_enabled and self.other_class_equalization:
            raise ValueError('Cannot select two different r methods.')

        self.reduced_class_indexes = OrderedDict()

        labels = torch.unique(torch.tensor(self.targets))

        self.class_indexes = OrderedDict()
        for label in labels:
            label = label.item()
            self.class_indexes[label] = torch.where(torch.tensor(self.targets) == label)[0]

        num_elem = len(self.features_dataset)
        if config.get_param_value('convex_hull/use_images_for_dim_reduction', False):
            class_features = torch.tensor(self.images_dataset.data).reshape(self.images_dataset.data.shape[0], -1)
        else:
            class_features = deepcopy(self.features_dataset.features)

        if self.use_dim_reduction_global:
            class_features = torch.tensor(
                sklearn.decomposition.PCA(n_components=self.dim_reduction_components).fit_transform(
                    class_features).astype(np.float32, copy=False))

        self.dists = torch.ones(num_elem) * -1
        for prev_i_class in range(self.num_classes):
            prev_class_idx = self.class_indexes[prev_i_class]
            prev_class_features = class_features[prev_class_idx]
            if self.use_dim_reduction:
                prev_class_features = torch.tensor(
                    sklearn.decomposition.PCA(n_components=self.dim_reduction_components).fit_transform(
                        prev_class_features).astype(np.float32, copy=False))
            mean_features = prev_class_features.mean(dim=0)
            dists = (prev_class_features - mean_features).pow(2).sum(dim=1).sqrt()
            self.dists[prev_class_idx] = dists
            sorted_dists_indexes = torch.flip(torch.argsort(dists), dims=[0])
            self.class_indexes[prev_i_class] = prev_class_idx[sorted_dists_indexes]

            print('    Class {} dists - min: {} max:{} mean:{} std:{}'.format(prev_i_class, dists.min(), dists.max(),
                                                                              dists.mean(), dists.std()))

    def get_train_loader(self, i_class: int = -1, model=None):
        dataset_train = self.features_dataset

        if i_class < 0 or i_class >= self.num_classes:
            raise ValueError('Incorrect class: {}'.format(i_class))

        train_labels = dataset_train.targets

        if i_class == 0:
            idx_no_c = train_labels == 1
        else:
            idx_no_c = train_labels < i_class

        images_no_c = dataset_train.data[idx_no_c]
        labels_no_c_org = dataset_train.targets[idx_no_c]
        labels_no_c_originals = torch.unique(labels_no_c_org).numpy()

        if self.other_class_equalization:
            idx_c = self.class_indexes[i_class][0:self.other_class_size]
            self.reduced_class_indexes[i_class] = idx_c
        else:
            idx_c = self.class_indexes[i_class]

        images_c = dataset_train.data[idx_c]
        labels_c_originals = torch.unique(dataset_train.targets[idx_c]).numpy()

        if i_class > 1:
            if self.other_class_equalization:
                img_per_class = self.other_class_size / i_class
                round_image_per_class = np.floor(img_per_class)
                left_images = self.other_class_size - (round_image_per_class * i_class)
                left_images_sum = np.array(([1] * int(left_images)) + ([0] * (i_class - int(left_images))))
                np.random.shuffle(left_images_sum)
                images_per_class = round_image_per_class + left_images_sum

            images_no_c = []
            labels_no_c = []
            for prev_i_class in range(i_class):
                prev_class_idx = self.class_indexes[prev_i_class]

                if self.other_class_equalization:
                    num_kept_images = int(images_per_class[prev_i_class])
                else:
                    num_kept_images = int(self.fixed_r * prev_class_idx.shape[0])

                if self.use_near_centroid_data:
                    idx = prev_class_idx[-num_kept_images:]
                else:
                    idx = prev_class_idx[0:num_kept_images]

                self.reduced_class_indexes[prev_i_class] = idx
                dists = self.dists[idx]
                images_no_c.append(dataset_train.data[idx])
                labels_no_c.append(dataset_train.targets[idx])

                print(
                    '    Class {} dists - min: {} max:{} mean:{} std:{}'.format(prev_i_class, dists.min(), dists.max(),
                                                                                dists.mean(), dists.std()))


            images_no_c = torch.vstack(images_no_c)
            labels_no_c_org = torch.hstack(labels_no_c)
            labels_no_c_originals = torch.unique(labels_no_c_org).numpy()

        n_images_c = images_c.shape[0]
        n_images_no_c = images_no_c.shape[0]
        labels_no_c = torch.zeros(n_images_no_c, dtype=torch.float)
        labels_c = torch.ones(n_images_c, dtype=torch.float)

        print('    Class 0 created from original labels: {}. Number of images: {}'.format(labels_no_c_originals,
                                                                                          n_images_no_c))
        print('    Class 1 created from original labels: {}. Number of images: {}'.format(labels_c_originals,
                                                                                          n_images_c))

        return self.gen_train_val_loader(images_no_c, images_c, labels_no_c, labels_c, labels_no_c_org,
                                         dataset_train.targets[idx_c])

    def get_eval_loader(self):
        dataset_train = self.dataset_class(root=self.dataset_path, download=self.download, train=True,
                                           transform=self.test_data_transform)

        last_class_key = list(self.class_indexes.keys())[-1]
        self.reduced_class_indexes[last_class_key] = self.class_indexes[last_class_key][
                                                     0:self.reduced_class_indexes[0].shape[0]]

        stacked_idx = torch.hstack([self.reduced_class_indexes[key] for key in list(self.reduced_class_indexes.keys())])
        new_labels = torch.tensor(dataset_train.targets)[stacked_idx]
        new_images = torch.tensor(dataset_train.data)[stacked_idx]

        if self.dataset_name.startswith('CIFAR'):
            dataset_train.data = new_images.numpy().astype(np.uint8)
            dataset_train.targets = new_labels.numpy().astype(np.uint8)
        else:
            dataset_train.data = new_images
            dataset_train.targets = new_labels

        eval_train_data_loader = DataLoader(dataset_train, batch_size=self.test_batch_size, drop_last=False)

        print('  Eval dataset made of {} images.'.format(stacked_idx.shape[0]))

        return eval_train_data_loader
