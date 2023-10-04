# config_utils.py
import argparse
import yaml
import os

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)

def parse_args():
    parser = argparse.ArgumentParser(description='Your script description here')
    parser.add_argument('--config', type=str, default='miccai.yaml',
                        help='Path to the configuration YAML file')
    parser.add_argument('--lambda_L2', type=float, default=None)
    # Add more arguments as needed for other hyperparameters
    return parser.parse_args()

def convert_numbers_to_floats(config):
    if isinstance(config, dict):
        for key, value in config.items():
            config[key] = convert_numbers_to_floats(value)
    elif isinstance(config, list):
        for i, value in enumerate(config):
            config[i] = convert_numbers_to_floats(value)
    elif isinstance(config, str):
        # Check if the string can be converted to a float (including scientific notation)
        try:
            config = float(config)
        except ValueError:
            pass  # Keep it as a string if it cannot be converted
    return config

def update_config_with_args(args):
    config_file_path = os.path.join('config',args.config)
    config = load_config(config_file_path)
    if args.lambda_L2 is not None:
        config['training']['lambda_L2'] = args.lambda_L2
    # Add similar checks for other hyperparameters
    config = convert_numbers_to_floats(config)
    return config