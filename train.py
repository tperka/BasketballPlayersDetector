import argparse

from model.training.train import begin_training, train_faster_rcnn
from utils.config import Config

parser = argparse.ArgumentParser(description='Train your own basketball players detector.')

parser.add_argument('--config', help="Config file path", type=str)
parser.add_argument('--rcnn', help="Train baseline model in Faster R-CNN architecture", action='store_true')
args = parser.parse_args()


if __name__ == "__main__":
    config = Config.from_configfile(args.config)
    if args.rcnn:
        train_faster_rcnn(config)
    begin_training(config)
