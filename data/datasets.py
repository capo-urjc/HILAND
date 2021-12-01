import copy
from enum import Enum
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, FashionMNIST

from config.parser import Configuration
from data.binary_datasets import ResamplingDatasetFixedR, ResamplingDataset, ConvexHullDataset, NormalDataset
from data.multiclass_datasets import NormalMCDataset, ResamplingMCEDataset, ResamplingFixedRMCEDataset
from model.transfer_learning import NetworkTypes


class DataSourceClass(Enum):
    @staticmethod
    def get_data_source_class(dataset_name):
        if dataset_name == 'MNIST':
            return MNIST
        elif dataset_name == 'FashionMNIST':
            return FashionMNIST
        elif dataset_name == 'CIFAR10':
            return CIFAR10
        elif dataset_name == 'CIFAR100':
            return CIFAR100
        else:
            raise ValueError('Dataset {} not recognized'.format(dataset_name))


class DatasetClass(Enum):
    @staticmethod
    def get_dataset_class(config: Configuration):
        resampling = config.get_param_value('resampling/enabled', mandatory=False)
        fixed_r_resampling = config.get_param_value('resampling/fixed_r_enabled', mandatory=False)

        network_type = NetworkTypes.get_network_type(config.get_param_value('network_type'))
        convex_hull = config.get_param_value('convex_hull/enabled', mandatory=False)
        equalize_binary_class_data = config.get_param_value('equalize_binary_class_data', mandatory=False)

        if convex_hull and (resampling or equalize_binary_class_data or fixed_r_resampling):
            raise ValueError('Convex hull algorithm is not ready to be used with resampling nor equalizing methods.')

        if network_type == NetworkTypes.MULTICLASS_ENSAMBLE:
            if fixed_r_resampling:
                return ResamplingFixedRMCEDataset
            elif resampling:
                return ResamplingMCEDataset
            else:
                return NormalMCDataset
        else:
            if resampling and fixed_r_resampling and network_type != NetworkTypes.MULTICLASS:
                return ResamplingDatasetFixedR
            elif resampling and network_type != NetworkTypes.MULTICLASS:
                return ResamplingDataset
            elif convex_hull and network_type != NetworkTypes.MULTICLASS:
                return ConvexHullDataset
            else:
                return NormalDataset




