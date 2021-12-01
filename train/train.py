import gc
import sys
from pathlib import Path
import torch
from torch import nn
from tqdm import tqdm

from config.parser import Configuration
from data.binary_datasets import BaseDataset
from log.tensorboard import Logger, LoggerTypes
from model.transfer_learning import CNN, NetworkTypes


class TrainerClass:
    @staticmethod
    def get_trainer_class(network_type):

        if type(network_type) == str:
            network_type = NetworkTypes.get_network_type(network_type)

        if network_type == NetworkTypes.MULTICLASS:
            return MulticlassTrainer
        elif network_type == NetworkTypes.BINARY_ENSAMBLE:
            return BinaryTrainer
        elif network_type == NetworkTypes.BINARY_ENSAMBLE_M:
            return BinaryTrainerM
        elif network_type == NetworkTypes.BINARY_ENSAMBLE_B:
            return BinaryTrainerB
        elif network_type == NetworkTypes.MULTICLASS_ENSAMBLE:
            return IncrementalMulticlassTrainer
        else:
            raise ValueError('Network type {} not recognized'.format(network_type))


class MulticlassTrainer:
    def __init__(self, config: Configuration, device: torch.device, model, output_result_path: Path,
                 dataset: BaseDataset, logger: Logger):
        self.device = device
        self.model = model
        self.model.to(device=self.device, dtype=torch.float32)
        self.output_result_path = output_result_path

        self.epochs = config.get_param_value('epochs')
        self.learning_rate = config.get_param_value('learning_rate')
        self.momentum = config.get_param_value('momentum')
        self.max_iterations = config.get_param_value('max_iterations')
        self.validate = config.get_param_value('validation/enabled', False) or False
        if self.validate:
            self.validation_iterations = config.get_param_value("validation/iterations")
        decay = config.get_param_value('weight_decay', False)
        self.weight_decay = decay if decay is not None else 0
        self.learning_rate_decay = config.get_param_value('learning_rate_decay', False)
        self.step_enable = config.get_param_value("learning_rate_decay/step_enable", False)
        if self.learning_rate_decay and self.step_enable:
            self.steps = config.get_param_value("learning_rate_decay/steps", False)
            self.factor = config.get_param_value("learning_rate_decay/factor", False)

        self.logger = logger
        self.dataset = dataset
        self.train_loader = None
        self.val_loader = None

        self.criterion = nn.NLLLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum,
                                         weight_decay=self.weight_decay)

        if self.learning_rate_decay and self.step_enable:
            milestones = self.steps
            gamma = self.factor

            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)

    def train(self):
        self.train_loader, self.val_loader = self.dataset.get_train_loader()
        self.model.train()
        self.best_val_acc = 0
        for epoch_num in range(0, self.epochs):
            global_iteration = epoch_num * len(self.train_loader)
            if global_iteration >= self.max_iterations:
                break

            train_loss = self.train_epoch(epoch_num)

        torch.save(self.model.state_dict(), str(self.output_result_path / 'model.pth'))

    def validation(self):
        self.model.eval()
        correct_preds = []
        for i_batch, (images, labels) in enumerate(self.val_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            correct_preds.append(torch.argmax(outputs, 1) == labels)

        correct_preds = torch.cat(correct_preds)
        val_acc = correct_preds.sum().item() / correct_preds.shape[0]
        if val_acc > self.best_val_acc:
            print('Validation improved from {:.04f} to {:.04f}'.format(self.best_val_acc, val_acc))
            self.best_val_acc = val_acc
            self.model.save_current_fc()

        self.model.train()

    def train_epoch(self, epoch):
        """
        Trains the model with all the training batches from the dataset
        :param epoch: the current epoch
        :return: the mean loss over the dataset batches of the trained network
        """

        train_loss = 0.0
        progress_msg = '  Epoch: {} - Iteration: {} - train_loss: {:.05}'

        iterator = tqdm(self.train_loader, file=sys.stdout)
        for i_batch, (images, labels) in enumerate(iterator):
            global_iteration = epoch * len(self.train_loader) + i_batch

            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.learning_rate_decay and self.step_enable:
                self.scheduler.step()

            train_loss += loss.item()

            if i_batch % 50 == 0:
                iterator.set_description(progress_msg.format(epoch + 1, global_iteration, train_loss / (i_batch + 1)))

            self.logger.write_scalar(LoggerTypes.TRAIN, 'Loss', train_loss / (i_batch + 1), global_iteration)
            self.logger.write_scalar(LoggerTypes.TRAIN, 'Accuracy',
                                     (torch.argmax(outputs, 1) == labels).sum().item() / images.shape[0],
                                     global_iteration)

            if self.validate and (global_iteration + 1) % self.validation_iterations == 0:
                self.validation()

            if (global_iteration + 1) >= self.max_iterations:
                if self.validate:
                    self.validation()
                break

        return train_loss

    def __str__(self):
        return 'Multiclass classifier'


class IncrementalMulticlassTrainer(MulticlassTrainer):
    def __init__(self, config: Configuration, device: torch.device, model, output_result_path: Path,
                 dataset: BaseDataset, logger: Logger):
        super().__init__(config, device, model, output_result_path, dataset, logger)
        self.num_classes_per_group = config.get_param_value('num_classes_per_group')
        self.num_groups = int(self.dataset.num_classes / self.num_classes_per_group)
        self.base_iterations = config.get_param_value('base_iterations')
        self.per_class_iterations = config.get_param_value('per_class_iterations')
        self.inc_train_exemplar_idx = []

    def train(self):
        self.model.train()

        for k_classifier in range(self.num_groups):
            self.model.change_fc(k_classifier)
            self.model.train()

            self.train_loader, self.val_loader = self.dataset.get_mc_train_loader(k_classifier)

            temp_dict = dict()
            for key in self.dataset.class_indexes.keys():
                temp_dict[key] = self.dataset.class_indexes[key].tolist()
            self.inc_train_exemplar_idx.append(temp_dict)

            self.max_iterations = self.base_iterations + k_classifier * self.per_class_iterations
            self.best_val_acc = 0

            if self.learning_rate_decay and self.step_enable:
                milestones = (self.max_iterations * torch.tensor(self.steps)).tolist()
                gamma = self.factor

                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones,
                                                                      gamma=gamma)

            for epoch_num in range(0, self.epochs):
                global_iteration = epoch_num * len(self.train_loader)
                if global_iteration >= self.max_iterations:
                    break
                self.train_epoch(epoch_num, k_classifier)

        self.dataset.get_mc_train_loader(k_classifier + 1, True)
        temp_dict = dict()
        for key in self.dataset.class_indexes.keys():
            temp_dict[key] = self.dataset.class_indexes[key].tolist()
        self.inc_train_exemplar_idx.append(temp_dict)

        torch.save(self.model.state_dict(), str(self.output_result_path / 'model.pth'))

    def train_epoch(self, epoch, k_classifier):
        """
        Trains the model with all the training batches from the dataset
        :param epoch: the current epoch
        :return: the mean loss over the dataset batches of the trained network
        """

        train_loss = 0.0
        progress_msg = '  Epoch: {} - Iteration: {} - train_loss: {:.05}'

        iterator = tqdm(self.train_loader, file=sys.stdout)
        for i_batch, (images, labels) in enumerate(iterator):
            global_iteration = epoch * len(self.train_loader) + i_batch

            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.learning_rate_decay and self.step_enable:
                self.scheduler.step()

            train_loss += loss.item()

            if i_batch % 50 == 0:
                iterator.set_description(progress_msg.format(epoch + 1, global_iteration, train_loss / (i_batch + 1)))

            self.logger.write_grouped_scalar(LoggerTypes.TRAIN, 'Loss', 'Classifier_{}'.format(k_classifier),
                                             train_loss / (i_batch + 1),
                                             global_iteration)
            self.logger.write_grouped_scalar(LoggerTypes.TRAIN, 'Accuracy', 'Classifier_{}'.format(k_classifier),
                                             (torch.argmax(outputs, 1) == labels).sum().item() / images.shape[0],
                                             global_iteration)

            if self.validate and (global_iteration + 1) % self.validation_iterations == 0:
                self.validation(global_iteration, k_classifier)

            if (global_iteration + 1) >= self.max_iterations:
                if self.validate:
                    self.validation(global_iteration, k_classifier)
                break

        return train_loss

    def validation(self, global_iteration, k_classifier):
        self.model.eval()
        correct_preds = []
        running_loss = 0
        for i_batch, (images, labels) in enumerate(self.val_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            running_loss += loss.item()
            correct_preds.append(torch.argmax(outputs, 1) == labels)

        correct_preds = torch.cat(correct_preds)
        val_acc = correct_preds.sum().item() / correct_preds.shape[0]
        if val_acc > self.best_val_acc:
            print('Validation improved from {:.04f} to {:.04f}'.format(self.best_val_acc, val_acc))
            self.best_val_acc = val_acc
            self.model.save_current_fc()

        self.logger.write_grouped_scalar(LoggerTypes.VAL, 'Loss', 'Classifier_{}'.format(k_classifier),
                                         running_loss / (i_batch + 1),
                                         global_iteration)
        self.logger.write_grouped_scalar(LoggerTypes.VAL, 'Accuracy', 'Classifier_{}'.format(k_classifier),
                                         val_acc,
                                         global_iteration)

        self.model.train()


class BinaryTrainer:
    def __init__(self, config: Configuration, device: torch.device, model: CNN, output_result_path: Path,
                 dataset: BaseDataset, logger: Logger):
        self.config = config
        self.device = device
        self.model = model
        self.dataset = dataset
        self.logger = logger
        self.output_result_path = output_result_path

        self.epochs = config.get_param_value('epochs')
        self.learning_rate = config.get_param_value('learning_rate')
        self.batch_size = config.get_param_value('train_batch_size')
        self.momentum = config.get_param_value('momentum')
        self.max_iterations = config.get_param_value('max_iterations')
        self.base_iterations = config.get_param_value('base_iterations')
        self.per_class_iterations = config.get_param_value('per_class_iterations')
        decay = config.get_param_value('weight_decay', False)
        self.weight_decay = decay if decay is not None else 0
        self.model.to(device=self.device, dtype=torch.float32)

        self.validate = config.get_param_value('validation/enabled', False) or False
        if self.validate:
            self.validation_iterations = config.get_param_value("validation/iterations")

        loss_reduction = config.get_param_value('binary_loss/reduction', mandatory=False)
        loss_with_logits = config.get_param_value('binary_loss/use_with_logits', mandatory=False)
        use_rescaling = config.get_param_value('binary_loss/use_rescaling', mandatory=False)

        self.loss_reduction = 'mean' if loss_reduction is None else loss_reduction
        self.loss_with_logits = False if loss_with_logits is None else loss_with_logits
        self.use_rescaling = False if use_rescaling is None else use_rescaling

        if self.use_rescaling:
            self.rescaling_beta = config.get_param_value('binary_loss/rescaling_beta', mandatory=False)
            self.rescaling_alpha = None

        if self.loss_with_logits:
            self.criterion = nn.BCEWithLogitsLoss(reduction=self.loss_reduction)
        else:
            self.criterion = nn.BCELoss(reduction=self.loss_reduction)

        self.inc_train_exemplar_idx = []

    def train(self):
        print('Training {} binary classifiers:'.format(self.dataset.num_classes))
        for i_class in range(self.dataset.num_classes):
            self.model.change_fc(i_class)
            self.model.train()

            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum,
                                        weight_decay=self.weight_decay)
            train_loader, self.val_loader = self.dataset.get_train_loader(i_class)
            max_iterations = self.base_iterations + i_class * self.per_class_iterations

            if self.use_rescaling:
                self.rescaling_alpha = 1 + self.rescaling_beta / self.dataset.reduction  # self.rescaling_alpha_list[i_class]
                print('  Using rescaling alpha = {}'.format(self.rescaling_alpha))

            print('  Training classifier: {}'.format(i_class))
            for epoch_num in range(0, self.epochs):
                global_iteration = epoch_num * len(train_loader)
                if global_iteration >= max_iterations:
                    break

                train_loss = self.train_epoch(epoch_num, i_class, optimizer, train_loader, max_iterations)

        torch.save(self.model.state_dict(), str(self.output_result_path / 'model.pth'))

    def validation(self):
        self.model.eval()
        with torch.no_grad():
            correct_preds = []
            for i_batch, (images, labels) in enumerate(self.val_loader):
                images = images.to(self.device)
                labels = labels.unsqueeze(1).float().to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                correct_preds.append((outputs > 0.5) == labels)

            correct_preds = torch.cat(correct_preds)
            val_acc = correct_preds.sum().item() / correct_preds.shape[0]
            if val_acc > self.best_val_acc:
                print('Validation improved from {:.04f} to {:.04f}'.format(self.best_val_acc, val_acc))
                self.best_val_acc = val_acc
                self.model.save_current_fc()

        self.model.train()

    def train_epoch(self, epoch, i_class, optimizer, train_loader, max_iterations):
        """
        Trains the model with all the training batches from the dataset
        :param max_iterations: number of train iterations to be made
        :param train_loader: the data loader to be used
        :param epoch: the current epoch
        :param optimizer: the network optimizer to use for backpropagation
        :return: a tuple containing the mean loss over the dataset batches of the trained network and its accuracy
        """
        train_loss = torch.tensor(0.0, device=self.device)

        progress_msg = '    Epoch: {} - Iteration: {} - train_loss: {}'
        iterator = tqdm(train_loader, file=sys.stdout)
        for i_batch, (images, labels) in enumerate(iterator):
            global_iteration = epoch * len(train_loader) + i_batch

            images = images.to(self.device)
            labels = labels.unsqueeze(1).float().to(self.device)

            if self.use_rescaling:
                self.criterion.weight = torch.ones((self.batch_size, 1), device=self.device) * self.rescaling_alpha
                self.criterion.weight[labels == 1] = 1

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            self.model.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss

            if i_batch % 10 == 0:
                iterator.set_description(progress_msg.format(epoch + 1,
                                                             global_iteration,
                                                             train_loss.item() / (i_batch + 1)))

            self.logger.write_grouped_scalar(LoggerTypes.TRAIN, 'Loss', 'Classifier_{}'.format(i_class),
                                             train_loss.item() / (i_batch + 1),
                                             global_iteration)
            self.logger.write_grouped_scalar(LoggerTypes.TRAIN, 'Accuracy', 'Classifier_{}'.format(i_class),
                                             (outputs.round() == labels).sum().item() / images.shape[0],
                                             global_iteration)

            if self.validate and (global_iteration + 1) % self.validation_iterations == 0:
                self.validation()

            if (global_iteration + 1) >= self.max_iterations:
                if self.validate:
                    self.validation()
                break

        return train_loss.item() / (i_batch + 1)

    def __str__(self):
        return 'Binary trainer'


class BinaryTrainerM(BinaryTrainer):
    def train(self):
        print('Training {} binary classifiers:'.format(self.dataset.num_classes - 1))
        for i_class in range(1, self.dataset.num_classes):
            self.model.change_fc(i_class - 1)
            self.model.train()

            if i_class == 1:
                criterion = nn.NLLLoss()
            else:
                if self.loss_with_logits:
                    criterion = nn.BCEWithLogitsLoss(reduction=self.loss_reduction)
                else:
                    criterion = nn.BCELoss(reduction=self.loss_reduction)

            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum,
                                        weight_decay=self.weight_decay)
            train_loader, self.val_loader = self.dataset.get_train_loader(i_class)
            max_iterations = self.base_iterations + i_class * self.per_class_iterations

            if self.use_rescaling:
                self.rescaling_alpha = 1 + self.rescaling_beta / self.dataset.reduction
                print('  Using rescaling alpha = {}'.format(self.rescaling_alpha))

            print('  Training classifier: {}'.format(i_class))
            for epoch_num in range(0, self.epochs):
                global_iteration = epoch_num * len(train_loader)
                if global_iteration >= max_iterations:
                    break

                train_loss = self.train_epoch(epoch_num, i_class, criterion, optimizer, train_loader, max_iterations)

        torch.save(self.model.state_dict(), str(self.output_result_path / 'model.pth'))

    def train_epoch(self, epoch, i_class, criterion, optimizer, train_loader, max_iterations):
        """
        Trains the model with all the training batches from the dataset
        :param max_iterations: number of train iterations to be made
        :param train_loader: the data loader to be used
        :param epoch: the current epoch
        :param optimizer: the network optimizer to use for backpropagation
        :return: a tuple containing the mean loss over the dataset batches of the trained network and its accuracy
        """
        train_loss = torch.tensor(0.0, device=self.device)

        progress_msg = '    Epoch: {} - Iteration: {} - train_loss: {}'
        iterator = tqdm(train_loader, file=sys.stdout)
        for i_batch, (images, labels) in enumerate(iterator):
            global_iteration = epoch * len(train_loader) + i_batch

            images = images.to(self.device)
            if i_class == 1:
                labels = labels.long().to(self.device)
            else:
                labels = labels.unsqueeze(1).float().to(self.device)
                if self.use_rescaling:
                    self.criterion.weight = torch.ones((self.batch_size, 1), device=self.device) * self.rescaling_alpha
                    self.criterion.weight[labels == 1] = 1

            outputs = self.model(images)
            loss = criterion(outputs, labels)

            self.model.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss

            if i_batch % 10 == 0:
                iterator.set_description(progress_msg.format(epoch + 1,
                                                             global_iteration,
                                                             train_loss.item() / (i_batch + 1)))

            self.logger.write_grouped_scalar(LoggerTypes.TRAIN, 'Loss', 'Classifier_{}'.format(i_class),
                                             train_loss.item() / (i_batch + 1),
                                             global_iteration)
            if i_class == 1:
                self.logger.write_grouped_scalar(LoggerTypes.TRAIN, 'Accuracy', 'Classifier_{}'.format(i_class),
                                                 (torch.argmax(outputs) == labels).sum().item() / images.shape[0],
                                                 global_iteration)
            else:
                self.logger.write_grouped_scalar(LoggerTypes.TRAIN, 'Accuracy', 'Classifier_{}'.format(i_class),
                                                 (outputs.round() == labels).sum().item() / images.shape[0],
                                                 global_iteration)

            if (global_iteration + 1) >= max_iterations:
                break

        return train_loss.item() / (i_batch + 1)


class BinaryTrainerB(BinaryTrainer):
    def train(self):
        print('Training {} binary classifiers:'.format(self.dataset.num_classes - 1))
        for i_class in range(1, self.dataset.num_classes):
            gc.collect()
            self.model.change_fc(i_class - 1)  #
            self.model.train()
            self.best_val_acc = 0
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum,
                                        weight_decay=self.weight_decay)
            train_loader, self.val_loader = self.dataset.get_train_loader(i_class)

            if hasattr(self.dataset, 'class_indexes'):
                temp_dict = dict()
                for key in self.dataset.class_indexes.keys():
                    temp_dict[key] = self.dataset.class_indexes[key].tolist()
                self.inc_train_exemplar_idx.append(temp_dict)

            max_iterations = self.base_iterations + i_class * self.per_class_iterations

            if self.use_rescaling:
                self.rescaling_alpha = 1 + (
                        self.rescaling_beta / self.dataset.reduction)  # self.rescaling_alpha_list[i_class - 1]
                print('  Using rescaling alpha = {}'.format(self.rescaling_alpha))

            print('  Training classifier: {}'.format(i_class))
            for epoch_num in range(0, self.epochs):
                global_iteration = epoch_num * len(train_loader)
                if global_iteration >= max_iterations:
                    break

                train_loss = self.train_epoch(epoch_num, i_class, optimizer, train_loader, max_iterations)
        if hasattr(self.dataset, 'class_indexes'):
            self.dataset.get_train_loader(i_class + 1, None, True)
            temp_dict = dict()
            for key in self.dataset.class_indexes.keys():
                temp_dict[key] = self.dataset.class_indexes[key].tolist()
            self.inc_train_exemplar_idx.append(temp_dict)

        torch.save(self.model.state_dict(), str(self.output_result_path / 'model.pth'))

    def __str__(self):
        return 'Binary trainer that skips the classifier 0'
