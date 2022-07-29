import gc
import json
import os
import random
import sys
import time
import uuid
from pathlib import Path
import numpy as np
import torch
from config.parser import Configuration
from data.datasets import DatasetClass, DataSourceClass
from log.logger import ConsoleLogger
from model.transfer_learning import BackendClass
from test.tester import TesterClass
from train.train import TrainerClass
from log.tensorboard import Logger
from data.external.cub200 import Cub2011


def init_seeds(seed: int, dataset_name: str):
    """
    Establishes a common seed for all the random number generators used in the code to enable reproducibility
    :param seed: the random seed value
    """
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_deterministic(True)

    if dataset_name == 'CUB200':
        Cub2011.random_order()


def main(config: Configuration, task_name: str):
    start_time = time.time()

    dataset_name = config.get_param_value('dataset_name')
    seed = config.get_param_value('seed')

    init_seeds(seed, dataset_name)
    torch.set_num_threads(1)

    output_folder_path = config.get_param_value('output_folder_path')
    output_folder_name = config.get_param_value('output_folder_name', False)

    if output_folder_name is None:
        output_folder_name = str(uuid.uuid1())

    output_result_path = Path(output_folder_path) / output_folder_name
    output_result_path.mkdir(parents=True, exist_ok=True)
    output_json_results_path = output_result_path / 'results_test.json'

    # Create a copy of the configuration used in the output directory
    config.save_config(output_result_path)

    logger = Logger(output_result_path, config.get_config())
    sys.stdout = ConsoleLogger(output_result_path, sys.stdout)

    gpu_id = config.get_param_value('gpu_id', False)
    device_name = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() and gpu_id is not None else "cpu")


    source_dataset_class = DataSourceClass.get_data_source_class(dataset_name)
    if dataset_name == 'CUB200':
        num_classes = 200
    else:
        raise NotImplementedError
    num_channels = 3

    network_class = BackendClass.get_backend_class(config.get_param_value('backend_name'))
    model = network_class(in_channels=num_channels, num_classes=num_classes, config=config, device=device_name)
    model = model.to(device_name)

    dataset_class = DatasetClass.get_dataset_class(config)
    dataset = dataset_class(config, output_result_path, device_name, model, source_dataset_class)

    print(dataset)
    print(dataset.features_dataset.class_mapping)
    print(model)

    if output_json_results_path.exists():
        with open(str(output_json_results_path), 'r') as fp:
            output_dict = json.load(fp)
    else:
        output_dict = {
            "time": {}
        }

    if task_name == 'train' or task_name == 'train_test':
        trainer_class = TrainerClass.get_trainer_class(config.get_param_value('network_type'))
        trainer = trainer_class(config=config, device=device_name, model=model, output_result_path=output_result_path,
                                dataset=dataset, logger=logger)

        print('Training with {}'.format(trainer))
        start_train_time = time.time()
        trainer.train()
        end_train_time = time.time()

        output_dict['time']['train'] = round(end_train_time - start_train_time, 4)
        gc.collect()
        tester_class = TesterClass.get_tester_class(config.get_param_value('network_type'),
                                                    config.get_param_value('fc_architecture/test_fusion_enabled',
                                                                           mandatory=True))
        eval_loader = dataset.get_eval_loader()
        tester = tester_class(config=config, device=device_name, model=model, output_result_path=output_result_path,
                              test_loader=eval_loader)
        output_result_dict = tester.test()
        output_dict['train_metrics'] = output_result_dict
        output_dict['train_metrics']['class_mapping'] = eval_loader.dataset.class_mapping.tolist()
        if hasattr(trainer, "inc_train_exemplar_idx"):
            output_dict['inc_train_exemplar_idx'] = trainer.inc_train_exemplar_idx

        if 'test_metrics' in output_dict:
            del output_dict['test_metrics']

    if task_name == 'test':
        model_path = str(output_result_path / 'model.pth')
        print('Loading test model form {}'.format(model_path))
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    if task_name == 'test' or task_name == 'train_test':
        print('Testing')
        tester_class = TesterClass.get_tester_class(config.get_param_value('network_type'),
                                                    config.get_param_value('fc_architecture/test_fusion_enabled',
                                                                           mandatory=True))
        test_loader = dataset.get_test_loader()
        tester = tester_class(config=config, device=device_name, model=model, output_result_path=output_result_path,
                              test_loader=test_loader)

        print(tester)

        start_test_time = time.time()
        output_result_dict = tester.test()
        output_dict['test_metrics'] = output_result_dict
        output_dict['test_metrics']['class_mapping'] = test_loader.dataset.class_mapping.tolist()

        end_test_time = time.time()
        output_dict['time']['test'] = round(end_test_time - start_test_time, 4)

    print('Execution time: {}s'.format(round(time.time() - start_time, 4)))
    end_time = time.time()

    output_dict['time']['total'] = round(end_time - start_time, 4)
    logger.end_logging()

    with open(str(output_result_path / 'results_test.json'), 'w') as fp:
        json.dump(output_dict, fp)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train arguments')
    parser.add_argument('task', choices=('train', 'test', 'train_test', 'stats', 'alpha', 'eval'))
    parser.add_argument('-c', '--config', type=str, required=True, help='path to the config file to be used')
    parser_args = vars(parser.parse_args())

    configuration = Configuration(parser_args['config'])
    task = parser_args['task']

    seed_value = configuration.get_param_value('seed')
    dataset_name = configuration.get_param_value('dataset_name')

    if isinstance(seed_value, list):
        for i, seed in enumerate(seed_value):
            configuration.config['seed'] = seed
            configuration.config['output_folder_name'] = str(i)
            main(configuration, task)
    else:
        main(configuration, task)
