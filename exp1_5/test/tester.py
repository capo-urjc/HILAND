import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.parser import Configuration
from model.transfer_learning import NetworkTypes, CNN


class TesterClass:
    @staticmethod
    def get_tester_class(network_type, test_fused_enabled: bool):

        if type(network_type) == str:
            network_type = NetworkTypes.get_network_type(network_type)

        if network_type == NetworkTypes.MULTICLASS:
            return MultiClassTester
        elif network_type == NetworkTypes.BINARY_ENSAMBLE:
            return FusedBinaries if test_fused_enabled else BinaryTester
        elif network_type == NetworkTypes.BINARY_ENSAMBLE_B:
            return FusedBinaries if test_fused_enabled else BinaryTesterB
        elif network_type == NetworkTypes.BINARY_ENSAMBLE_M:
            return BinaryTesterM
        elif network_type == NetworkTypes.MULTICLASS_ENSAMBLE:
            return IncrementalMulticlassTester
        else:
            raise ValueError('Network type {} not recognized'.format(network_type))


class MultiClassTester:
    def __init__(self, config: Configuration, device: torch.device, model: CNN, output_result_path: Path,
                 test_loader: DataLoader):
        self.config = config
        self.device = device
        self.model = model
        self.model.to(device=self.device, dtype=torch.float32)
        self.output_result_path = output_result_path
        self.validation_enabled = self.config.get_param_value('validation/enabled', False)

        self.test_loader = test_loader

    def test(self):
        if self.validation_enabled:
            self.model.model = self.model.best_model
        self.model.eval()
        with torch.no_grad():
            values_prob = []
            iterator = tqdm(self.test_loader)
            for images, labels in iterator:
                images = images.to(self.device)
                outputs = self.model(images, True)
                values_prob.append(outputs)

            labels = torch.tensor(self.test_loader.dataset.targets)
            values_prob = torch.vstack(values_prob).permute(1, 0)
            dataset_classes = self.test_loader.dataset.classes

            return dict(predictions=values_prob.tolist(), labels=labels.tolist(), dataset_classes=dataset_classes)


class IncrementalMulticlassTester(MultiClassTester):
    def __init__(self, config: Configuration, device: torch.device, model: CNN, output_result_path: Path,
                 test_loader: DataLoader):
        super().__init__(config, device, model, output_result_path, test_loader)

    def test(self):
        with torch.no_grad():
            self.model.fuse_fc()
        self.model.to(device=self.device, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            values_prob = []
            iterator = tqdm(self.test_loader)
            for images, labels in iterator:
                images = images.to(self.device)
                outputs = self.model(images, True)
                values_prob.append(outputs)

            labels = torch.tensor(self.test_loader.dataset.targets)
            values_prob = torch.vstack(values_prob).permute(1, 0)
            dataset_classes = self.test_loader.dataset.classes

            return dict(predictions=values_prob.tolist(), labels=labels.tolist(), dataset_classes=dataset_classes)


class FusedBinaries:
    def __init__(self, config: Configuration, device: torch.device, model: CNN, output_result_path: Path,
                 test_loader: DataLoader):
        self.config = config
        self.device = device
        self.model = model
        self.output_result_path = output_result_path

        self.test_loader = test_loader

        self.num_classes = len(self.test_loader.dataset.classes)
        self.network_type = NetworkTypes.get_network_type(self.config.get_param_value('network_type'))

    def test(self):
        self.model.fuse_fc()
        self.model.to(device=self.device, dtype=torch.float32)
        self.model.eval()

        with torch.no_grad():
            values_prob = []

            iterator = tqdm(self.test_loader)
            for images, labels in iterator:
                images = images.to(self.device)
                outputs = self.model(images, True)
                values_prob.append(outputs)

            if self.network_type == NetworkTypes.BINARY_ENSAMBLE_B:
                values_prob = torch.vstack(values_prob).permute(1, 0)
                prob_zero = 1 - values_prob[0, :]
                values_prob = torch.vstack((prob_zero, values_prob))
            else:
                values_prob = torch.vstack(values_prob).permute(1, 0)

            labels = torch.tensor(self.test_loader.dataset.targets)
            dataset_classes = self.test_loader.dataset.classes

            return dict(predictions=values_prob.tolist(), labels=labels.tolist(), dataset_classes=dataset_classes)


class BinaryTester:
    def __init__(self, config: Configuration, device: torch.device, model: CNN, output_result_path: Path,
                 test_loader: DataLoader):
        self.device = device
        self.model = model
        self.output_result_path = output_result_path
        self.test_loader = test_loader
        self.num_classes = len(self.test_loader.dataset.classes)

    def test(self):
        with torch.no_grad():
            append_outputs_prob = []
            for i_class in range(self.num_classes):
                self.model.change_fc(i_class)
                self.model.to(device=self.device, dtype=torch.float32)
                self.model.eval()

                outputs_current_class = []

                for images, labels in tqdm(self.test_loader, desc="  Classifier {}".format(i_class), file=sys.stdout):
                    images = images.to(self.device)
                    outputs = self.model(images, True)
                    outputs_current_class.append(outputs)

                outputs_prob = torch.vstack(outputs_current_class)

                append_outputs_prob.append(outputs_prob)

            labels = torch.tensor(self.test_loader.dataset.targets)
            values_prob = torch.stack(append_outputs_prob)
            dataset_classes = self.test_loader.dataset.classes

            return dict(predictions=values_prob.tolist(), labels=labels.tolist(), dataset_classes=dataset_classes)


class BinaryTesterM:
    def __init__(self, config: Configuration, device: torch.device, model: CNN, output_result_path: Path,
                 test_loader: DataLoader):
        self.device = device
        self.model = model
        self.output_result_path = output_result_path
        self.test_loader = test_loader
        self.num_classes = len(self.test_loader.dataset.classes)

    def test(self):
        with torch.no_grad():
            total_items = self.test_loader.dataset.data.shape[0]

            append_outputs_prob = []
            for i_class in range(1, self.num_classes):
                self.model.change_fc(i_class - 1)
                self.model.to(device=self.device, dtype=torch.float32)
                self.model.eval()

                outputs_current_class = []

                for images, labels in tqdm(self.test_loader, desc="  Classifier {}".format(i_class), file=sys.stdout):
                    images = images.to(self.device)
                    outputs = self.model(images, True)
                    outputs_current_class.append(outputs)

                if i_class == 1:
                    outputs_prob = torch.stack(outputs_current_class).view(total_items, 2)
                    outputs_prob = 10 ** outputs_prob
                    append_outputs_prob.append(outputs_prob[:, 0])
                    append_outputs_prob.append(outputs_prob[:, 1])
                else:
                    outputs_prob = torch.stack(outputs_current_class).view(total_items)
                    append_outputs_prob.append(outputs_prob)

            # -- All the outputs_prob in one matrix
            labels = torch.tensor(self.test_loader.dataset.targets)
            values_prob = torch.stack(append_outputs_prob)
            dataset_classes = self.test_loader.dataset.classes

            return dict(predictions=values_prob.tolist(), labels=labels.tolist(), dataset_classes=dataset_classes)


class BinaryTesterB:
    def __init__(self, config: Configuration, device: torch.device, model: CNN, output_result_path: Path,
                 test_loader: DataLoader):
        self.device = device
        self.model = model
        self.output_result_path = output_result_path
        self.test_loader = test_loader
        self.num_classes = len(self.test_loader.dataset.classes)

    def test(self):
        with torch.no_grad():

            append_outputs_prob = []
            for i_class in range(1, self.num_classes):
                self.model.change_fc(i_class - 1)
                self.model.to(device=self.device, dtype=torch.float32)
                self.model.eval()

                outputs_current_class = []

                for images, labels in tqdm(self.test_loader, desc="  Classifier {}".format(i_class), file=sys.stdout):
                    images = images.to(self.device)
                    outputs = self.model(images, True)
                    outputs_current_class.append(outputs)

                if i_class == 1:
                    outputs_prob = torch.vstack(outputs_current_class)
                    append_outputs_prob.append(1 - outputs_prob)
                    append_outputs_prob.append(outputs_prob)
                else:
                    outputs_prob = torch.vstack(outputs_current_class)
                    append_outputs_prob.append(outputs_prob)

            # -- All the outputs_prob in one matrix
            labels = torch.tensor(self.test_loader.dataset.targets)
            values_prob = torch.stack(append_outputs_prob)
            dataset_classes = self.test_loader.dataset.classes

            return dict(predictions=values_prob.tolist(), labels=labels.tolist(), dataset_classes=dataset_classes)
