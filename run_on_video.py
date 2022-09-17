import argparse
from time import perf_counter

import cv2
import torchvision
import os
import torch

from dataset.transforms import TestAugmentation, NORMALIZATION_MEAN, NORMALIZATION_STD
from model import builder, utils
from model.utils import get_all_saved_models, get_filename_from_config
from utils.config import Config

parser = argparse.ArgumentParser(description='Run trained basketball player detector on video')

parser.add_argument('--config', help="Config file path. Only 'Model' section is required", type=str)
parser.add_argument('video_filepath', help="File path of input video stream", type=str)
parser.add_argument('output', help="Output file path for result video with bounding boxes.", type=str)

args = parser.parse_args()

@torch.no_grad()
def run_on_video(config, input_path, output_path):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model_filepath = config.path_to_model
    model = builder.build_basketball_detector(config.model_name, scale=config.scale,
                                              max_detections=config.max_detections,
                                              nms_threshold=config.nms_threshold,
                                              c_r_depth=config.classifier_regressor_depth,
                                              fpn_depth=config.fpn_lateral_depth,
                                              c_r_kernel_size=config.classifier_regressor_kernel_size,
                                              channel_attention=config.channel_attention_module,
                                              delta=config.delta, player_threshold=config.player_threshold)

    assert os.path.exists(model_filepath), "Specified model filepath: {} must exist".format(
        model_filepath)
    model_state_dict = torch.load(model_filepath, map_location=device)["model"]
    model.load_state_dict(model_state_dict)
    model.eval()
    try:
        model.summary(False)
    except Exception:
        pass

    if torch.cuda.is_available():
        print("Using cuda")
        print(torch.cuda.get_device_name())
    else:
        print("Using cpu")
    model = model.to(device)

    test_vid = cv2.VideoCapture(input_path)
    fps = int(test_vid.get(cv2.CAP_PROP_FPS))
    width = int(test_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(test_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_list = []
    frames_cnt = 0
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD)])
    start = perf_counter()
    while True:
        ret, frame = test_vid.read()
        if not ret:
            break

        frame_tens = transforms(frame)
        frame_tens.unsqueeze_(0)
        frame_tens = frame_tens.to(device)
        result = model(frame_tens)

        bboxes = result[0]["boxes"]

        # drawing = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        color = (0, 255, 0)
        for bbox in bboxes:
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)

        frames_list.append(frame)
        frames_cnt += 1

    end = perf_counter()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps,
                             (width, height))
    for frame in frames_list:
        writer.write(frame)
    test_vid.release()
    writer.release()
    processing_time = (end - start) * 1000
    processing_time_per_frame = processing_time / frames_cnt
    fps = 1000 / processing_time_per_frame
    print(f"Processing {frames_cnt} frames took {processing_time} ms [ {fps} FPS ]")
    return fps


if __name__ == "__main__":
    config = Config.from_configfile(args.config, training=False)
    run_on_video(config, args.video_filepath, args.output)
