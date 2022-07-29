import copy
from abc import abstractmethod, ABC
from enum import Enum
import torch
from torch import nn
from torch.nn import ModuleList
from torchvision.models import resnet18, vgg16, resnet50, densenet161, inception_v3, resnet34, googlenet

from config.parser import Configuration


class NetworkTypes(Enum):
    BINARY_ENSAMBLE = 0
    MULTICLASS = 1
    BINARY_ENSAMBLE_M = 2
    BINARY_ENSAMBLE_B = 3
    MULTICLASS_ENSAMBLE = 4

    @staticmethod
    def get_network_type(network_type: str):
        if network_type == 'BE':
            return NetworkTypes.BINARY_ENSAMBLE
        elif network_type == 'MC':
            return NetworkTypes.MULTICLASS
        elif network_type == 'BEM':
            return NetworkTypes.BINARY_ENSAMBLE_M
        elif network_type == 'BEB':
            return NetworkTypes.BINARY_ENSAMBLE_B
        elif network_type == 'MCE':
            return NetworkTypes.MULTICLASS_ENSAMBLE
        else:
            raise ValueError('Network type {} not recognized'.format(network_type))


class BackendClass(Enum):
    RESNET = 0
    VGG = 1
    DENSENET = 2
    INCEPTION = 3
    GOOGLENET = 4

    @staticmethod
    def get_backend_name(backend_name: str):
        if backend_name.startswith('ResNet'):
            return BackendClass.RESNET
        elif backend_name.startswith('VGG'):
            return BackendClass.VGG
        elif backend_name.startswith('Densenet'):
            return BackendClass.DENSENET
        elif backend_name.startswith('Inception'):
            return BackendClass.INCEPTION
        elif backend_name.startswith('GoogleNet'):
            return BackendClass.GOOGLENET
        else:
            raise ValueError('Backend name {} not recognized'.format(backend_name))

    @staticmethod
    def get_backend_class(backend_name: str):
        if backend_name.startswith('ResNet'):
            return ResNet
        elif backend_name.startswith('VGG'):
            return VGG
        elif backend_name.startswith('Densenet'):
            return Densenet
        elif backend_name.startswith('Inception'):
            return Inception
        elif backend_name.startswith('GoogleNet'):
            return GoogleNet
        else:
            raise ValueError('Backend name {} not recognized'.format(backend_name))


class CNN(nn.Module, ABC):
    def __init__(self, in_channels: int, num_classes: int, config: Configuration, device):
        super().__init__()
        self.config = config
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.network_type = NetworkTypes.get_network_type(self.config.get_param_value('network_type'))
        self.dataset_name = config.get_param_value('dataset_name')
        self.unfreeze_first_task = config.get_param_value('fc_architecture/unfreeze_first_task', False) or False
        self.unfreeze_last_layers = config.get_param_value('fc_architecture/unfreeze_last_layers', False) or False

        self.device = device
        self.fused_fc = None
        self.cnn, self.num_features, self.backend_layers_to_unfreeze = self.get_model()

        if not self.unfreeze_first_task:
            for param in self.cnn.parameters():
                param.requires_grad = False
            print('Backend frozen')
        else:
            if len(self.backend_layers_to_unfreeze) > 0 and self.unfreeze_last_layers:
                for param in self.cnn.parameters():
                    param.requires_grad = False
                print('Backend frozen')
                for layer_index in self.backend_layers_to_unfreeze:
                    print('Unfreezing layer {} - {}'.format(layer_index, self.cnn[0][layer_index]))
                    for name, param in self.cnn[0][layer_index].named_parameters():
                        param.requires_grad = True
                        print('\tUnfrozen {}'.format(name))


        if self.network_type == NetworkTypes.BINARY_ENSAMBLE:
            self.fc_list = ModuleList([self.create_fc(NetworkTypes.BINARY_ENSAMBLE) for _ in range(num_classes)])
            self.model = self.fc_list[0]
            self.old_i_class = 0
        elif self.network_type == NetworkTypes.BINARY_ENSAMBLE_B:
            self.fc_list = ModuleList([self.create_fc(NetworkTypes.BINARY_ENSAMBLE) for _ in range(num_classes - 1)])
            self.model = self.fc_list[0]
            self.old_i_class = 0
        elif self.network_type == NetworkTypes.BINARY_ENSAMBLE_M:
            self.fc_list = ModuleList([self.create_fc(NetworkTypes.BINARY_ENSAMBLE) for _ in range(num_classes - 2)])
            initial_fc = self.create_fc(NetworkTypes.MULTICLASS, num_classes=2)
            self.fc_list.insert(0, initial_fc)
            self.model = initial_fc
            self.old_i_class = 0
        elif self.network_type == NetworkTypes.MULTICLASS_ENSAMBLE:
            self.num_classes_per_group = config.get_param_value('num_classes_per_group')
            num_groups = len(self.num_classes_per_group)
            fcs = [self.create_fc(NetworkTypes.MULTICLASS, num_classes=self.num_classes_per_group[0])] + \
                  [self.create_fc(NetworkTypes.MULTICLASS, num_classes=self.num_classes_per_group[i] + 1)
                   for i in range(1, num_groups)]

            self.fc_list = ModuleList(fcs)
            self.model = self.fc_list[0]
            self.old_i_class = 0
        else:
            self.model = self.create_fc(NetworkTypes.MULTICLASS, num_classes=self.num_classes)

        if self.network_type != NetworkTypes.MULTICLASS:
            self.best_fc_list = [None for _ in self.fc_list]
        else:
            self.best_model = None

        self.model.requires_grad_(True)

    def fuse_fc(self, k_classifier=None):
        hidden_layers_size = self.config.get_param_value('fc_architecture/hidden_layers_size')
        validation_enabled = self.config.get_param_value('validation/enabled', False) or False

        if validation_enabled:
            layers_to_fuse = self.best_fc_list
        else:
            layers_to_fuse = self.fc_list

        if len(hidden_layers_size) == 0:
            if self.network_type == NetworkTypes.BINARY_ENSAMBLE or self.network_type == NetworkTypes.BINARY_ENSAMBLE_B:
                if self.network_type == NetworkTypes.BINARY_ENSAMBLE_B:
                    num_classes_fused = self.num_classes - 1
                else:
                    num_classes_fused = self.num_classes

                self.fused_fc = nn.Sequential(
                    nn.Linear(self.num_features, num_classes_fused),
                    nn.Sigmoid()
                )
                for idx, header in enumerate(layers_to_fuse):
                    self.fused_fc[0].weight[idx, :] = header[0].weight
                    self.fused_fc[0].bias[idx] = header[0].bias
                self.old_i_class = -1
                self.model = self.fused_fc
            elif self.network_type == NetworkTypes.MULTICLASS_ENSAMBLE:
                num_classes = self.num_classes
                if k_classifier is not None:
                    num_classes = self.num_classes_per_group[:(k_classifier + 1)]
                    layers_to_fuse = layers_to_fuse[:(k_classifier + 1)]

                self.fused_fc = nn.Sequential(
                    nn.Linear(self.num_features, num_classes),
                    nn.LogSoftmax(dim=1)
                )
                self.fused_fc.to(self.device)
                class_intervals = [0] + self.num_classes_per_group
                for i, header in enumerate(layers_to_fuse):
                    layer_classes = self.num_classes_per_group[i]
                    idx = torch.tensor(list(range(sum(class_intervals[:i + 1]), sum(class_intervals[:i + 2]))),
                                       dtype=torch.long, device=self.device)
                    self.fused_fc[0].weight[idx, :] = torch.tensor(header[0].weight[-layer_classes:],
                                                                   device=self.device)
                    self.fused_fc[0].bias[idx] = header[0].bias[-layer_classes:]
                self.old_i_class = -1
                self.model = self.fused_fc
        else:
            raise NotImplementedError('Network fusion not implemented for FC with hidden layers')

    def save_current_fc(self):
        if self.network_type != NetworkTypes.MULTICLASS:
            self.best_fc_list[self.old_i_class] = copy.deepcopy(self.model)
        else:
            self.best_model = copy.deepcopy(self.model)

    def freeze_backend(self):
        for param in self.cnn.parameters():
            param.requires_grad = False
        print('Backend frozen')

    def change_fc(self, i_class: int):
        # if self.unfreeze_first_task and i_class == 1:
        #     for param in self.cnn.parameters():
        #         param.requires_grad = False
        #     print('Backend frozen')

        if self.network_type == NetworkTypes.BINARY_ENSAMBLE or \
                self.network_type == NetworkTypes.BINARY_ENSAMBLE_B or \
                self.network_type == NetworkTypes.BINARY_ENSAMBLE_M or \
                self.network_type == NetworkTypes.MULTICLASS_ENSAMBLE:
            if self.old_i_class != -1:
                current_fc = self.model
                current_fc.requires_grad_(False)

                self.fc_list[self.old_i_class] = current_fc

            new_fc = self.fc_list[i_class]
            new_fc.requires_grad_(True)

            self.model = new_fc
            self.old_i_class = i_class
        else:
            raise ValueError('Cannot change the classifier in a non ensamble network type')

    def create_fc(self, network_type, num_classes=None):
        use_dropout = self.config.get_param_value('fc_architecture/dropout', False)
        if use_dropout:
            dropout_prob = self.config.get_param_value('fc_architecture/dropout_prob', False)
        hidden_layers_size = self.config.get_param_value('fc_architecture/hidden_layers_size')
        hidden_activation = ("Sigmoid", nn.Sigmoid) if network_type == NetworkTypes.BINARY_ENSAMBLE else (
            "ReLU", nn.ReLU)
        fully_connected = nn.Sequential()
        num_input_features = self.num_features

        for i, size in enumerate(hidden_layers_size):
            fully_connected.add_module("linear_{}".format(i), nn.Linear(num_input_features, size))
            fully_connected.add_module("{}_{}".format(hidden_activation[0], i), hidden_activation[1]())
            if use_dropout:
                fully_connected.add_module("dropout_{}".format(i), nn.Dropout(p=dropout_prob))
            num_input_features = size

        if network_type == NetworkTypes.BINARY_ENSAMBLE:
            fully_connected.add_module("linear_{}".format(len(hidden_layers_size)), nn.Linear(num_input_features, 1))
            if use_dropout:
                fully_connected.add_module("dropout_{}".format(len(hidden_layers_size)), nn.Dropout(p=dropout_prob))
            fully_connected.add_module("Sigmoid_{}".format(len(hidden_layers_size)), nn.Sigmoid())
        else:
            fully_connected.add_module("linear_{}".format(len(hidden_layers_size)),
                                       nn.Linear(num_input_features, num_classes))
            if use_dropout:
                fully_connected.add_module("dropout_{}".format(len(hidden_layers_size)), nn.Dropout(p=dropout_prob))
            fully_connected.add_module("LogSoftmax", nn.LogSoftmax(dim=1))

        return fully_connected

    def forward(self, x, testing=False):
        # if not testing:
        #     out = self.model(x)
        # else:

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        out = self.cnn(x)
        out = torch.flatten(out, 1)
        out = self.model(out)

        return out

    @abstractmethod
    def get_model(self):
        raise NotImplemented

    def get_model_from_dict(self, name):
        model_name = self.config.get_param_value('backend_name')
        if model_name in self.models:
            model = self.models[model_name](pretrained=True)
            return model
        else:
            raise ValueError('{} type: "{}" not found. Use one of: {}'.format(name, model_name, self.models.keys()))

    def __str__(self):
        info = 'Model Information\n'
        info += '  Network type: {}\n'.format(self.config.get_param_value('network_type'))
        info += '  Backend: {}\n'.format(self.config.get_param_value('backend_name'))
        info += '  Backend features: {}\n'.format(self.num_features)

        return info


class ResNet(CNN):
    def __init__(self, in_channels: int, num_classes: int, config: Configuration, device):
        self.models = dict(ResNet18=resnet18, ResNet50=resnet50, ResNet34=resnet34)
        super().__init__(in_channels, num_classes, config, device)

    def get_model(self):
        model = self.get_model_from_dict('ResNet')
        cnn = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3,
                            model.layer4, model.avgpool)
        return cnn, model.fc.in_features, []


class VGG(CNN):
    def __init__(self, in_channels: int, num_classes: int, config: Configuration, device):
        self.models = dict(VGG16=vgg16)
        super().__init__(in_channels, num_classes, config, device)

    def get_model(self):
        model = self.get_model_from_dict('VGG')
        cnn = nn.Sequential(model.features)
        return cnn, model.classifier[0].in_features, [24, 26, 28]


class Densenet(CNN):
    def __init__(self, in_channels: int, num_classes: int, config: Configuration, device):
        self.models = dict(Densenet=densenet161)
        super().__init__(in_channels, num_classes, config, device)

    def get_model(self):
        model = self.get_model_from_dict('Densenet')
        cnn = nn.Sequential(model.features, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)))
        return cnn, model.classifier.in_features, []


class Inception(CNN):
    def __init__(self, in_channels: int, num_classes: int, config: Configuration, device):
        self.models = dict(Inception=inception_v3)
        super().__init__(in_channels, num_classes, config, device)

    def get_model(self):
        model = self.get_model_from_dict('Inception')

        cnn = nn.Sequential(model.Conv2d_1a_3x3,
                            model.Conv2d_2a_3x3,
                            model.Conv2d_2b_3x3,
                            model.maxpool1,
                            model.Conv2d_3b_1x1,
                            model.Conv2d_4a_3x3,
                            model.maxpool2,
                            model.Mixed_5b,
                            model.Mixed_5c,
                            model.Mixed_5d,
                            model.Mixed_6a,
                            model.Mixed_6b,
                            model.Mixed_6c,
                            model.Mixed_6d,
                            model.Mixed_6e,
                            model.Mixed_7a,
                            model.Mixed_7b,
                            model.Mixed_7c,
                            model.avgpool,
                            model.dropout
                            )
        return cnn, model.fc.in_features, []


class GoogleNet(CNN):
    def __init__(self, in_channels: int, num_classes: int, config: Configuration, device):
        self.models = dict(GoogleNet=googlenet)
        super().__init__(in_channels, num_classes, config, device)

    def get_model(self):
        model = self.get_model_from_dict('GoogleNet')
        cnn = nn.Sequential(model.conv1,
                            model.maxpool1,
                            model.conv2,
                            model.conv3,
                            model.maxpool2,
                            model.inception3a,
                            model.inception3b,
                            model.maxpool3,
                            model.inception4a,
                            model.inception4b,
                            model.inception4c,
                            model.inception4d,
                            model.inception4e,
                            model.maxpool4,
                            model.inception5a,
                            model.inception5b,
                            model.avgpool
                            )
        return cnn, model.fc.in_features, [14, 15]
