# Basketball Players Detector

### Setup

It is highly recommended to use Python virtual environment.

Please first run:

`pip install -r requirements.txt`

To install all the dependencies.

### Downloading dataset

To download dataset described in the paper please run:

`python3 download_dataset.py <output_dir>`

The dataset will be downloaded to temp .zip file and then extracted to `output_dir`.

### Training your own model

First, you have to define feature extractor exact architecture in `model/basenet_configs.py`. There are three
architectures included:

1. VGG
2. ResNet
3. EfficientNet

The key in map should correspond to model name. Then you should create config file. You can find example with
decsription in
`configs/example.config`. After it's created you should run:

`python3 train.py --config <path_to_config> [--rcnn]`

Specify `--rcnn` flag to train baseline FasterRCNN model.

### Testing model

To run your trained model on test dataset just run:

`python3 test.py --config <path_to_config>`

It will run iteration on test dataset and print all the stats.

### Running on video

To run trained model on given video please use:

`python3 run_on_video.py --config <path_to_config> <path_to_video> <output_filepath>`

Where `output_path` is exact filepath of output video with bounding boxes.

## Included example

The best model from paper is saved in `saved_models/bpd.pth`. Its exact config is available in `configs/example.config`.

