import argparse

from model.training.train import run_test_iteration
from utils.config import Config

parser = argparse.ArgumentParser(description='Train your own basketball players detector.')

parser.add_argument('--config', help="Config file path", type=str)
args = parser.parse_args()


if __name__ == "__main__":
    config = Config.from_configfile(args.config)
    run_test_iteration(config)
