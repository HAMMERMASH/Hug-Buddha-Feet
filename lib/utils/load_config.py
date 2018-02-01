import yaml
from easydict import EasyDict as edict

def load_config(filename):
  with open(filename, 'r') as f:
    config_dict = yaml.load(f)
    config = edict(config_dict)
    return config
