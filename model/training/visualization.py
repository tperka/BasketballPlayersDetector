import os
import random
from time import perf_counter
from timeit import default_timer as timer

import numpy as np
import torch
from cv2 import cv2
from matplotlib import pyplot as plt, patches as patches

import dataset
import model
from dataset.NCAADataset import NCAADataset
from dataset.transforms import TestAugmentation, tensor2image, TrainAugmentation, denormalize_trans
from model import builder
from model.training.train import get_groundtruth_maps
from model.training.utils import unpack_coco_batch
from model.utils import load_checkpoint, get_filename_from_config


def show_on_image(img, target, idx, save_path=None, without_axis=True):
    boxes = target["boxes"]
    if "scores" in target:
        scores = target["scores"]
    else:
        scores = None

    boxes = boxes.detach()
    if isinstance(img, torch.Tensor):
        img = tensor2image(img)
        width, height = img.shape[1], img.shape[0]
    else:
        width, height = img.width, img.shape
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), tight_layout={"pad": 0.00})
    if not without_axis:
        ax.set_title(str(idx))
    show_bbs(img, boxes, ax, scores, save_path=save_path, without_axis=without_axis)



@torch.no_grad()
def visualize_n_predictions(config, n=10):
    model = builder.build_basketball_detector(config["model_name"], nms_threshold=0.45, player_threshold=0.3,
                                              c_r_kernel_size=config["kernel_size"], delta=config["delta"],
                                              c_r_depth=config["classifier_regressor_depth"],
                                              fpn_depth=config["fpn_lateral_depth"],
                                              channel_attention=config["channel_attention"], max_detections=500)
    checkpoint = load_checkpoint(config)
    model.load_state_dict(checkpoint["model"])
    #model.summary(True)
    model.eval()
    dataset = NCAADataset(os.path.join(config["dataset_root_dir"], config["quality"], "val"), transform=TestAugmentation((490, 360)))

    for _ in range(n):
        idx = random.randrange(len(dataset))
        image, _ ,_ = dataset[idx]
        image = torch.unsqueeze(image, 0)
        #gt_map = get_groundtruth_maps(batch, model, torch.device("cpu"))
        start = perf_counter()
        boxes_with_scores = model(image)
        #img = compose_confidence_maps(gt_map[1], boxes_with_scores[1], model.player_downsampling)
        end = perf_counter()
        print(f"Prediction took: {(end - start) * 1000} ms")
        #img = denormalize_trans(image[0])
        for target in boxes_with_scores:
            show_on_image(image[0], target, idx)


def show_bbs(image, bbs, ax, scores=None, save_path=None, without_axis=False):
    ax.imshow(image)
    if scores is None:
        scores = [None] * len(bbs)
    for row, score in zip(bbs, scores):
        rect = patches.Polygon(np.array([(row[0], row[1]), (row[2], row[1]), (row[2], row[3]), (row[0], row[3])]),
                                          linewidth=3,
                                 edgecolor="r",
                                 facecolor="none")
        ax.add_patch(rect)
        if score:
            ax.text(row[0], row[1], "{:.2f}".format(score.item()), color="c", alpha=1.0, fontweight="demibold")

    if without_axis:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    if save_path:
        plt.savefig(save_path, format='png', transparent=True)
    plt.show()


def heatmap2image(tensor, channel=1):
    # Convert 1-channel (or more) heatmap/confidence map to numpy image
    # tensor: (h, w) or (h, w, n_channels) tensor
    # channel: channel to show/convert to image (used if there's more than one channel)

    assert tensor.dim() == 2 or tensor.dim() == 3

    if tensor.dim() == 3:
        tensor = tensor[:, :, channel]

    image = tensor.cpu().numpy().astype(np.uint8)*255
    heatmap = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    return heatmap


def compose_confidence_maps(target_map, predicted_map, upscale_factor):
    # Visualize target and predicted confidence map side-by-side
    # target_map: Target (ground truth) confidence map
    # predicted_map: Predicted confidence map
    target_img = model.training.visualization.heatmap2image(target_map)
    predicted_image = model.training.visualization.heatmap2image(predicted_map)
    h, w = target_img.shape[0], target_img.shape[1]

    out_img = np.zeros((h, w * 2, 3), dtype=target_img.dtype)

    # Show ground truth confidence map on the left and predicted confidence map on the right
    out_img[:, :w] = target_img
    out_img[:, w:] = predicted_image
    out_img = cv2.resize(out_img, (w * 2 * upscale_factor, h * upscale_factor), cv2.INTER_NEAREST)
    cv2.line(out_img, (w * upscale_factor, 0), (w * upscale_factor, h * upscale_factor), (0, 255, 255),
             thickness=1)
    return out_img


def show_confidence_map(conf_map: torch.Tensor, loc_map: torch.Tensor, image: torch.Tensor, ax):
    scale = image.shape[0] / loc_map.shape[0]
    h, w = loc_map.shape[0], loc_map.shape[1]
    ax.imshow(image)
    #conf_map.flatten()
    if len(loc_map.shape) == 3:
        loc_map = loc_map.unsqueeze(2)
    for i in range(loc_map.shape[0]):
        for j in range(loc_map.shape[1]):
            # if conf_map is not None and conf_map[i, j].float().sum() != 0:
            #     xy = (j * scale, i * scale)
            #     rect = patches.Rectangle(xy, scale, scale, alpha=0.4, color="yellow")
            #     ax.add_patch(rect)
            # else:
            #     mesh = patches.Rectangle((j * scale, i * scale), scale, scale, alpha=1.0, edgecolor="black", linewidth=1, facecolor=None, fill=False)
            #     ax.add_patch(mesh)
            for locc in loc_map[i, j]:
                if locc.float().sum() != 0:
                    x = locc[0] * w * scale
                    y = locc[1] * h * scale
                    width = locc[2] * w * scale
                    height = locc[3] * h * scale
                    xc = j * scale + (scale - 1) / 2
                    yc = i * scale + (scale - 1) / 2
                    bx = xc + x
                    by = yc + y

                    x1 = bx - 0.5 * width
                    x2 = bx + 0.5 * width
                    y1 = by - 0.5 * height
                    y2 = by + 0.5 * height
                    if len(locc) == 5 and locc[4] == dataset.labels.PLAYER_LABEL:
                        xy = (j * scale, i * scale)
                        rect = patches.Rectangle(xy, scale, scale, alpha=0.4, color="yellow")
                        ax.add_patch(rect)
                    rect = patches.Rectangle((x1, y1), (x2 - x1), (y2 - y1), edgecolor="red", facecolor="none")
                    ax.add_patch(rect)
    plt.savefig("lol.png", format='png', transparent=True)
    plt.show()


def visualize_conf_map(config, n=5, groundtruth=True):
    model = builder.build_basketball_detector(config["model_name"])
    model_state_dict = torch.load(config["model_filepath"], map_location="cpu")["model"]
    model.load_state_dict(model_state_dict)
    model.train()
    train, test = dataset.dataloaders.get_dataloaders(config, False)
    i = 0
    for batch in test:
        normal_batch = unpack_coco_batch(batch) if len(batch) == 2 else batch
        torch.set_printoptions(profile="full")
        imagos = batch[0]
        imagos = imagos.squeeze()
        img = tensor2image(imagos)
        fig, ax = plt.subplots(figsize=(img.shape[1] / 100, img.shape[0] / 100), tight_layout={"pad": 0.00})
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # if "ybd" not in model.name:
        if groundtruth:
            player_loc_t, player_conf_t = get_groundtruth_maps(normal_batch, model, torch.device("cpu"))
        else:
            result = model(normal_batch)
        player_conf_t = player_conf_t.squeeze()
        player_loc_t = player_loc_t.squeeze()
        show_confidence_map(player_conf_t, player_loc_t, img, ax)
        # else:
            # player_loc_t = get_groundtruth_maps(normal_batch, model, torch.device("cpu"))
            # player_loc_t = player_loc_t.squeeze()
            # show_confidence_map(None, player_loc_t, imagos, ax)
        i += 1
        if i >= n:
            break


def show_on_images(dataset, n):
    # Use a breakpoint in the code line below to debug your script.
    for i in range(n):
        idx = random.randrange(len(dataset))
        sample = dataset[idx]
        image = tensor2image(sample[0])
        target = sample[1]
        bbs = target["boxes"]
        fig, ax = plt.subplots()
        show_bbs(image, bbs, ax)



if __name__ == "__main__":
    dataset_path = f"{os.path.expanduser('~')}/ncaa"
    model_name = "eca_8"
    config = {
        "dataset_root_dir": "/home/tperka/ncaa",
        "quality": "LQ",
        "train_batch_size": 10,
        "test_batch_size": 4,
        "num_workers": 6,
        "model_name": model_name,
        "output_dir": "saved_models",
        "scheduler_milestones": [30, 50],
        "classifier_regressor_depth": 128,
        "fpn_lateral_depth": 32,
        "channel_attention": False,
        "lr": 1e-3,
        "max_epochs": 60,
        "delta": 3,
        "kernel_size": 3,
    }
    config["model_filepath"] = os.path.join("saved_models", model_name, f"{get_filename_from_config(config)}.pth")
    #visualize_conf_map(config, 1, True)

    ds = NCAADataset(f"{dataset_path}/LQ/test", transform=TrainAugmentation(size=(490,360)), coco_api=True)
    for i in range(1):
        idx = 1701#random.randrange(0, len(ds))
        print(idx)
        img, target = ds[idx]
        show_on_image(img, target, i, save_path="temp.png")

