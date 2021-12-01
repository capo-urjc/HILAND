import math
from abc import ABC
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import sklearn
import torch
from torch.utils.data import DataLoader

from config.parser import Configuration
from data.binary_datasets import BaseDataset
from data.feature_dataset import FeaturesDataset

class MultiClassBaseDataset(BaseDataset, ABC):
    def __init__(self, config: Configuration, output_result_path, device, model, dataset_class):
        super().__init__(config, output_result_path, device, model, dataset_class)
        self.num_classes_per_group = config.get_param_value('num_classes_per_group')

    def get_mc_train_loader(self, classifier_id):
        raise NotImplementedError

    def get_mc_test_loader(self, k_classifier=None):
        dataset_test = self.dataset_class(root=self.dataset_path, download=self.download, train=False,
                                          transform=self.test_data_transform)

        if k_classifier is not None:
            classes = list(range(0, (k_classifier + 1) * self.num_classes_per_group))
            idx = np.isin(dataset_test.targets, classes)
            dataset_test.targets = torch.tensor(dataset_test.targets)[idx]
            dataset_test.data = dataset_test.data[idx]

        test_data_loader = DataLoader(dataset_test, batch_size=self.test_batch_size, drop_last=False)

        return test_data_loader

    def get_mc_eval_loader(self):
        dataset_train = self.dataset_class(root=self.dataset_path, download=self.download, train=True,
                                           transform=self.test_data_transform)
        eval_train_data_loader = DataLoader(dataset_train, batch_size=self.test_batch_size, drop_last=False)

        return eval_train_data_loader

class NormalMCDataset(MultiClassBaseDataset):
    def __init__(self, config: Configuration, output_result_path, device, model, dataset_class):
        super().__init__(config, output_result_path, device, model, dataset_class)
        self.equalize_binary_class_data = config.get_param_value('equalize_binary_class_data', False)
        self.sub_name = "NormalMCDataset"
        self.print_msg = '  Creating dataset for multi-class classifier: {} ' \
                         'with classes: {} for class 0 and classes {} for class 1.'

    def get_mc_train_loader(self, k_classifier):
        c_classes = np.array(list(range(k_classifier * self.num_classes_per_group,
                                        (k_classifier + 1) * self.num_classes_per_group)))
        no_c_classes = np.array(list(range(k_classifier * self.num_classes_per_group)))

        print(self.print_msg.format(k_classifier, no_c_classes, c_classes))

        dataset_train = self.features_dataset
        train_labels = dataset_train.labels
        train_data = dataset_train.features

        idx_no_c = np.isin(train_labels, no_c_classes)
        images_no_c = train_data[idx_no_c]
        n_images_no_c = images_no_c.shape[0]
        labels_no_c = torch.zeros(n_images_no_c, dtype=torch.float)
        labels_no_c_unique = torch.unique(labels_no_c).numpy()
        labels_no_c_originals = torch.unique(train_labels[idx_no_c]).numpy()

        idx_c = np.isin(train_labels, c_classes)
        images_c = train_data[idx_c]
        n_images_c = images_c.shape[0]
        labels_c = train_labels[idx_c] - (k_classifier * self.num_classes_per_group)
        if k_classifier > 0:
            labels_c += 1

        labels_c_unique = torch.unique(labels_c).numpy()
        labels_c_originals = torch.unique(train_labels[idx_c]).numpy()

        print('    Class {} created from original labels: {}. Number of images: {}'.format(labels_no_c_unique,
                                                                                           labels_no_c_originals,
                                                                                           n_images_no_c))
        print('    Class {} created from original labels: {}. Number of images: {}'.format(labels_c_unique,
                                                                                           labels_c_originals,
                                                                                           n_images_c))

        return self.gen_train_val_loader(images_no_c, images_c, labels_no_c, labels_c,
                                           train_labels[idx_no_c], train_labels[idx_c])


class ResamplingMCEDataset(MultiClassBaseDataset):
    def __init__(self, config: Configuration, output_result_path, device, model, dataset_class):
        super().__init__(config, output_result_path, device, model, dataset_class)
        self.resampling_samples = config.get_param_value('resampling/samples')
        self.resampling_current_samples = config.get_param_value('resampling/current_samples', False)
        # self.resample_class_c = config.get_param_value('resampling/resample_class_c')
        # self.resampling_fixed_r_enabled = config.get_param_value('resampling/fixed_r_enabled', False)
        # self.resample_fixed_r = config.get_param_value('resampling/fixed_r', False)
        if self.resampling_samples % 2:
            raise ValueError('Number of resampling samples must be even')

        self.sub_name = "ResamplingMCEDataset"
        labels = torch.unique(torch.tensor(self.sample_dataset.targets))

        self.class_indexes = OrderedDict()
        for label in labels:
            label = label.item()
            self.class_indexes[label] = torch.where(torch.tensor(self.sample_dataset.targets) == label)[0]

        self.print_msg = '  Creating dataset for multi-class classifier: {} ' \
                         'with classes: {} for class 0 and classes {} for class 1.'

    def get_data_subset(self, class_idx, train_data, train_labels, class_zero, k_classifier):
        if len(class_idx) != 0:
            class_samples = self.resampling_samples / 2

            if self.resampling_current_samples is not None and class_zero:
                class_samples = self.resampling_samples
            elif self.resampling_current_samples is not None and not class_zero:
                class_samples = self.resampling_current_samples

            imgs_per_prev_class = math.ceil(class_samples / len(class_idx))
            total_images_no_c = len(class_idx) * imgs_per_prev_class
            leftover_imgs = int(total_images_no_c - class_samples)

            # Create binary mask that specifies in which class to remove an image
            ones_classes = [0] * (len(class_idx) - leftover_imgs)
            zeros_classes = [1] * leftover_imgs
            mask = torch.tensor(ones_classes + zeros_classes)
            shuffled_mask = mask[torch.randperm(len(mask))]

            images_no_c = []
            indexes = []
            for i, prev_class in enumerate(class_idx):
                current_class_num_images = imgs_per_prev_class - shuffled_mask[i]
                class_idx = self.class_indexes[prev_class]

                full_repeats = math.floor(current_class_num_images / class_idx.shape[0])
                # extra images to be added but less than one full repeat
                leftover_images = current_class_num_images - (full_repeats * class_idx.shape[0])

                shuffled_images_indexes = torch.randperm(class_idx.shape[0])
                leftover_shuffled_images_indexes = shuffled_images_indexes[0:leftover_images]
                # repeats + leftovers
                class_idx = torch.cat((full_repeats * [class_idx]) + [class_idx[leftover_shuffled_images_indexes]])

                # save used idx in order to use the same images in the next classifier
                self.class_indexes[prev_class] = class_idx

                images_no_c.append(train_data[class_idx])
                indexes.append(class_idx)
                print('Number of images for class {}: {}, unique labels: {}'.format(prev_class, class_idx.shape[0],
                                                                                    np.unique(train_labels[class_idx])))

            images_no_c = torch.vstack(images_no_c)
            indexes = torch.hstack(indexes)
            n_images_no_c = images_no_c.shape[0]

            if class_zero:
                labels_no_c = torch.zeros(n_images_no_c, dtype=torch.float)
            else:
                labels_no_c = train_labels[indexes] - (k_classifier * self.num_classes_per_group)
                if k_classifier > 0:
                    labels_no_c += 1
        else:
            images_no_c = torch.tensor([])
            labels_no_c = torch.tensor([])
            indexes = torch.tensor([], dtype=torch.long)

        return images_no_c, labels_no_c, indexes

    def get_mc_train_loader(self, k_classifier, after_last_classifier=False):
        c_classes = np.array(list(range(k_classifier * self.num_classes_per_group,
                                        (k_classifier + 1) * self.num_classes_per_group)))
        no_c_classes = np.array(list(range(k_classifier * self.num_classes_per_group)))

        print(self.print_msg.format(k_classifier, no_c_classes, c_classes))

        dataset_train = self.features_dataset
        train_labels = dataset_train.labels
        train_data = dataset_train.features

        images_no_c, labels_no_c, indexes_no_c = self.get_data_subset(no_c_classes, train_data, train_labels, True,
                                                                      k_classifier)
        if not after_last_classifier:
            images_c, labels_c, indexes_c = self.get_data_subset(c_classes, train_data, train_labels, False, k_classifier)

            n_images_no_c = images_no_c.shape[0]
            labels_no_c_unique = torch.unique(labels_no_c).numpy()
            labels_no_c_originals = torch.unique(train_labels[indexes_no_c]).numpy()
            n_images_c = images_c.shape[0]
            labels_c_unique = torch.unique(labels_c).numpy()
            labels_c_originals = torch.unique(train_labels[indexes_c]).numpy()

            print('    Class {} created from original labels: {}. Number of images: {}'.format(labels_no_c_unique,
                                                                                               labels_no_c_originals,
                                                                                               n_images_no_c))
            print('    Class {} created from original labels: {}. Number of images: {}'.format(labels_c_unique,
                                                                                               labels_c_originals,
                                                                                               n_images_c))

            return self.gen_train_val_loader(images_no_c, images_c, labels_no_c, labels_c,
                                               train_labels[indexes_no_c], train_labels[indexes_c])

    def get_eval_loader(self):

        dataset_train = self.dataset_class(root=self.dataset_path, download=self.download, train=True,
                                           transform=self.test_data_transform)

        eval_train_data_loader = DataLoader(dataset_train, batch_size=self.test_batch_size, drop_last=False)

        print('  Eval dataset made of {} images.'.format(torch.tensor(dataset_train.data).shape[0]))

        return eval_train_data_loader


class ResamplingFixedRMCEDataset(ResamplingMCEDataset):
    def __init__(self, config: Configuration, output_result_path, device, model, dataset_class):
        super().__init__(config, output_result_path, device, model, dataset_class)
        self.resample_fixed_r = config.get_param_value('resampling/fixed_r', False)
        self.sub_name = "ResamplingFixedRMCEDataset"

    def get_mc_train_loader(self, k_classifier):
        c_classes = np.array(list(range(k_classifier * self.num_classes_per_group,
                                        (k_classifier + 1) * self.num_classes_per_group)))
        no_c_classes = np.array(list(range(k_classifier * self.num_classes_per_group)))
        data_per_class = len(self.features_dataset.labels) / self.num_classes

        self.resampling_samples = self.resample_fixed_r * data_per_class * len(no_c_classes)
        self.resampling_current_samples = self.resample_fixed_r * data_per_class * len(c_classes)
        return super().get_mc_train_loader(k_classifier)


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

        labels = torch.unique(torch.tensor(self.sample_dataset.targets))

        self.class_indexes = OrderedDict()
        for label in labels:
            label = label.item()
            self.class_indexes[label] = torch.where(torch.tensor(self.sample_dataset.targets) == label)[0]

        num_elem = len(self.features_dataset)
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

        train_labels = dataset_train.labels

        if i_class == 0:
            idx_no_c = train_labels == 1
        else:
            idx_no_c = train_labels < i_class

        images_no_c = dataset_train.features[idx_no_c]
        labels_no_c_originals = torch.unique(dataset_train.labels[idx_no_c]).numpy()

        if self.other_class_equalization:
            idx_c = self.class_indexes[i_class][0:self.other_class_size]
            self.reduced_class_indexes[i_class] = idx_c
        else:
            idx_c = self.class_indexes[i_class]

        images_c = dataset_train.features[idx_c]
        labels_c_originals = torch.unique(dataset_train.labels[idx_c]).numpy()

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
                images_no_c.append(dataset_train.features[idx])
                labels_no_c.append(dataset_train.labels[idx])

                print(
                    '    Class {} dists - min: {} max:{} mean:{} std:{}'.format(prev_i_class, dists.min(), dists.max(),
                                                                                dists.mean(), dists.std()))

            images_no_c = torch.vstack(images_no_c)
            labels_no_c_originals = torch.unique(torch.hstack(labels_no_c)).numpy()

        n_images_c = images_c.shape[0]
        n_images_no_c = images_no_c.shape[0]
        labels_no_c = torch.zeros(n_images_no_c, dtype=torch.float)
        labels_c = torch.ones(n_images_c, dtype=torch.float)

        print('    Class 0 created from original labels: {}. Number of images: {}'.format(labels_no_c_originals,
                                                                                          n_images_no_c))
        print('    Class 1 created from original labels: {}. Number of images: {}'.format(labels_c_originals,
                                                                                          n_images_c))

        new_images = torch.cat((images_no_c, images_c))
        new_labels = torch.cat((labels_no_c, labels_c))

        dataset_train = FeaturesDataset(new_images, new_labels)
        train_data_loader = DataLoader(dataset_train, batch_size=self.train_batch_size, shuffle=True)
        return train_data_loader

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
