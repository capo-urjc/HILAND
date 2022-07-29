import json
from pathlib import Path


class Configuration:
    """JSON configuration file parser"""

    def __init__(self, config_path, config_dict=None):
        if config_path is not None:
            f = open(config_path, 'r')
            self.config = json.load(f)
            f.close()
        else:
            self.config = config_dict

    def get_config(self):
        """
        Provides access to the configuration
        :return: the configuration dict
        """
        return self.config

    def get_param_value(self, param: str, mandatory=True):
        sub_params = param.split('/')
        config = self.config
        for sub_param in sub_params:
            config = self.__get_param_value(config, sub_param, mandatory)

        return config

    def __get_param_value(self, config: dict, param: str, mandatory=True):
        """
        Access to a configuration property
        :param param: name of the property
        :param mandatory: rise error if param value is not found
        :return: value of the property
        """
        if config is not None and param in config:
            return config[param]
        elif mandatory:
            raise ValueError('Parameter "{}" is mandatory'.format(param))
        else:
            return None

    def save_config(self, output_dir: Path):
        """
        Stores a copy of the configuration
        :param output_dir: path to the output directory
        """
        with open(str(output_dir / 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=4)
