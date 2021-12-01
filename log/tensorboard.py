from enum import Enum
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class LoggerTypes(Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2

    @staticmethod
    def get_name(logger_type):
        if logger_type is LoggerTypes.TRAIN:
            return 'Train'
        elif logger_type is LoggerTypes.VAL:
            return 'Val'
        elif logger_type is LoggerTypes.TEST:
            return 'Test'
        else:
            raise ValueError('Logger type no defined: {}'.format(logger_type))


class Logger:
    def __init__(self, output_dir, params_dict):
        output_dir_log = output_dir / 'logs'
        output_dir_train_writer = output_dir_log / 'train'
        output_dir_val_writer = output_dir_log / 'validation'
        output_dir_test_writer = output_dir_log / 'test'

        output_dir_train_writer.mkdir(parents=True, exist_ok=True)
        output_dir_val_writer.mkdir(parents=True, exist_ok=True)
        output_dir_test_writer.mkdir(parents=True, exist_ok=True)

        self.train_writer = SummaryWriter(output_dir_train_writer)
        self.val_writer = SummaryWriter(output_dir_val_writer)
        self.test_writer = SummaryWriter(output_dir_test_writer)

        # Init tensorboard writer and write train params as markdown table
        formatted_params = self.__table_from_dict__(params_dict)  # table format for tensorboard
        self.train_writer.add_text('Params', """Param | Value\n-------|-------\n{}""".format(formatted_params))

        self.scalar_loss_values = {}
        for log_type in list(LoggerTypes):
            self.scalar_loss_values[log_type] = {}

        self.scalar_values = {}
        for log_type in list(LoggerTypes):
            self.scalar_values[log_type] = {}

    def __table_from_dict__(self, params_dict):
        table = []
        for key in list(params_dict.keys()):
            if type(params_dict[key]) is dict:
                table.append(self.__table_from_dict__(params_dict[key]))
            else:
                table.append('{} | {}'.format(key, params_dict[key]))

        table.sort()
        return '\n'.join(table)

    def __get_writer__(self, logger_type):
        if logger_type is LoggerTypes.TRAIN:
            return self.train_writer
        elif logger_type is LoggerTypes.VAL:
            return self.val_writer
        elif logger_type is LoggerTypes.TEST:
            return self.test_writer
        else:
            raise ValueError('Logger writer type "{}" is not defined'.format(type))

    def write_scalar(self, logger_type, name, value, epoch):
        writer = self.__get_writer__(logger_type)
        writer.add_scalar(LoggerTypes.get_name(logger_type) + name, value, epoch)
        self.scalar_values[logger_type][name] = value

    def write_grouped_scalar(self, logger_type, group, name, value, epoch):
        writer = self.__get_writer__(logger_type)
        writer.add_scalar("{}/{}".format(group, name), value, epoch)
        self.scalar_values[logger_type][name] = value

    def write_scalar_dict(self, logger_type, logger_dict, epoch):
        writer = self.__get_writer__(logger_type)
        for log_name in logger_dict.keys():
            for metric_name in logger_dict[log_name].keys():
                name = '{}/{}'.format(log_name, metric_name)
                value = logger_dict[log_name][metric_name]
                writer.add_scalar(LoggerTypes.get_name(logger_type) + name, value, epoch)
                if name not in self.scalar_loss_values[logger_type]:
                    self.scalar_loss_values[logger_type][name] = []
                self.scalar_loss_values[logger_type][name].append(value)

    def write_epoch_metrics(self, epoch):
        for log_type in self.scalar_loss_values.keys():
            writer = self.__get_writer__(log_type)
            for name in self.scalar_loss_values[log_type].keys():
                metric_name = 'Epoch_{}'.format(name)
                writer.add_scalar(metric_name, np.mean(self.scalar_loss_values[log_type][name]), epoch)

        for log_type in self.scalar_values.keys():
            writer = self.__get_writer__(log_type)
            for name in self.scalar_values[log_type].keys():
                metric_name = 'Epoch_{}'.format(name)
                writer.add_scalar(metric_name, self.scalar_values[log_type][name], epoch)

    def end_logging(self):
        self.train_writer.close()
        self.val_writer.close()
        self.test_writer.close()
