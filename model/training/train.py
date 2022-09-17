import csv
import datetime
import math
import os
import sys
import time

import torch
import torchvision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import optim
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import pycocoutils.engine
from dataset.dataloaders import get_dataloaders, get_test_dataloader
from model import utils
from model.training import const
from model.training.ssd_loss import SSDLoss
from model.training.utils import PredictionsToDict, unpack_coco_batch, build_model_from_config
from model.utils import save_model, get_filename_from_config, load_checkpoint


def train_one_epoch(model, dataloader, config, optimizer, loss_func, device, epoch, alpha_class, alpha_local, target_dir, print_info_freq=50):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: {epoch}"
    model_filename = get_filename_from_config(config)
    filename = f"{target_dir}/{model_filename}_train_losses.csv"
    with open(filename, "a", newline="") as f:
        loss_logger = csv.writer(f)
        if epoch == 1:
            loss_logger.writerow(["epoch", "total_loss", "classification_loss", "localization_loss"])
        for batch in metric_logger.log_every(dataloader, print_info_freq, header):

            gt_maps = get_groundtruth_maps(batch, model, device)
            images = batch[0]
            images = images.to(device)
            with torch.set_grad_enabled(True):
                predictions = model(images)

                optimizer.zero_grad()
                loss_localization, loss_classification = loss_func(predictions, gt_maps)
                total_loss = alpha_local * loss_localization + alpha_class * loss_classification

                loss_logger.writerow([epoch, total_loss.item(), loss_classification.item(), loss_localization.item()])
                if not math.isfinite(total_loss.item()):
                    print(f"Loss is {total_loss}, stopping training. Did it diverge?")
                    sys.exit(1)

                total_loss.backward()
                optimizer.step()

                metric_logger.update(loss=total_loss, classification_loss=loss_classification, localization_loss=loss_localization)
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])


@torch.no_grad()
def evaluate(model, dataloader, config, device, loss_func, alpha_local, alpha_class, epoch, coco, target_dir):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    filename = get_filename_from_config(config)
    loss_filename = f"{target_dir}/{filename}_test_losses.csv"
    stats_filename = f"{target_dir}/{filename}_stats.csv"

    predictions_to_dict_converter = PredictionsToDict()

    coco_dets = []
    with open(loss_filename, "a", newline="") as f:
        model.eval()
        loss_logger = csv.writer(f)
        if epoch == 1:
            loss_logger.writerow(["epoch", "total_loss", "classification_loss", "localization_loss"])
        for coco_batch in metric_logger.log_every(dataloader, 100, header):
            batch = unpack_coco_batch(coco_batch)
            images, targets = coco_batch
            images = images.to(device)
            gt_maps = get_groundtruth_maps(batch, model, device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_time = time.time()
            predictions = model(images)

            model.train()
            train_predictions = model(images)
            model.eval()
            for prediction, target in zip(predictions, targets):
                dets = predictions_to_dict_converter(prediction, target)
                if dets is not None:
                    coco_dets += dets

            model_time = time.time() - model_time

            loss_localization, loss_classification = loss_func(train_predictions, gt_maps)
            total_loss = alpha_local * loss_localization + alpha_class * loss_classification
            loss_logger.writerow([epoch, total_loss.item(), loss_classification.item(), loss_localization.item()])
            metric_logger.update(model_time=model_time, total_loss=total_loss, class_loss=loss_classification, local_loss=loss_localization)

    coco_result = coco.loadRes(coco_dets)
    coco_eval = COCOeval(coco, coco_result, iouType="bbox")
    #coco_eval.accumulate()
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # for metric, score in coco_eval.eval:
    #     print(f'{metric}: {score:.3f}')
    #metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    with open(stats_filename, "a", newline="") as f:
        stats_logger = csv.writer(f)
        if epoch == 1:
            stats_logger.writerow(const.COCO_STATS_CSV_HEADER)
        stats_logger.writerow(coco_eval.stats[const.COCO_STATS_INDEXES].tolist())
    return coco_eval.stats[const.COCO_STATS_INDEXES].tolist()


def get_groundtruth_maps(batch, model, device):
    images, boxes, labels = batch
    images = images.to(device)
    h, w = images.shape[-2], images.shape[-1]
    gt_maps = model.create_groundtruth_maps(boxes, labels, (w, h))
    gt_maps = [element.to(device) for element in gt_maps]
    return gt_maps


def train_model(config, model, dataloaders, optimizer, scheduler, device=torch.device("cpu"), begin_epoch=0):

    # values used in SSD original implementation, for this specific task optimal values might be different
    alpha_localization = 0.3
    alpha_classification = 1

    loss_func = SSDLoss(neg_pos_ratio=3)

    train_dataloader, val_dataloader = dataloaders
    coco_gt_filepath = os.path.join(config.dataset_root_dir, config.dataset_quality, "val", "annotations", "coco.json")

    coco = COCO(coco_gt_filepath)
    print("Starting training...")
    start_time = time.time()
    for epoch in range(begin_epoch + 1, config.epochs + 1):
        train_one_epoch(model, train_dataloader, config, optimizer=optimizer, loss_func=loss_func, device=device, epoch=epoch, alpha_local=alpha_localization, alpha_class=alpha_classification, target_dir=config.output_dir)
        scheduler.step()
        evaluate(model, val_dataloader, config, device=device, loss_func=loss_func, alpha_class=alpha_classification, alpha_local=alpha_localization, epoch=epoch, coco=coco, target_dir=config.output_dir)
        save_model(config, model, optimizer, scheduler, epoch)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def begin_training(config):
    model = build_model_from_config(config)

    if not os.path.exists(config.output_dir):
        os.mkdir(config.output_dir)

    assert os.path.exists(config.output_dir), "Cannot create folder to save trained model: {}".format(config.output_dir)
    if config.model_name == "rcnn":
        train_faster_rcnn(config)
        return

    dataloaders = get_dataloaders(config.dataset_root_dir, config.dataset_quality, config.train_batch, config.test_batch, config.dataloader_workers)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")
    device = torch.device(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler_milestones = config.lr_scheduler_milestones
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, scheduler_milestones)
    epoch = 0
    if config.path_to_model:
        print(f"Path to model is specified, loading training params from {config.path_to_model}...")
        assert os.path.exists(config.path_to_model), f"Specified model filepath: {config.path_to_model} must exist"
        saved_dict = torch.load(config.path_to_model, map_location=device)
        model_state_dict = saved_dict["model"]
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(saved_dict["optimizer"])
        scheduler.load_state_dict(saved_dict["lr_scheduler"])
        epoch = saved_dict["epoch"]

    model.summary()
    model = model.to(device)

    train_model(config, model, dataloaders, optimizer, scheduler, device, begin_epoch=epoch)


def train_faster_rcnn(config):
    loss_filename = f"{config.output_dir}/faster_rcnn_train_losses.csv"
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    train, test = get_dataloaders(config.dataset_root_dir, config.dataset_quality, config.train_batch, config.test_batch, config.dataloader_workers, train_coco=True)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        print("Using cuda")
        print(torch.cuda.get_device_name())
    else:
        print("Using cpu")

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=config.learning_rate)
    scheduler_milestones = [10, 25]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, scheduler_milestones)
    for epoch in range(1, config.epochs + 1):
        metric_logger = pycocoutils.engine.train_one_epoch(model, optimizer, train, device, epoch, print_freq=100)
        with open(loss_filename, 'a', newline="") as f:
            loss_logger = csv.writer(f)
            if epoch == 1:
                loss_logger.writerow(["epoch", "loss"])
            loss_logger.writerow([epoch, metric_logger.loss])
        lr_scheduler.step()
        pycocoutils.engine.evaluate(model, test, device=device)
        save_model(config, model, optimizer, lr_scheduler, epoch)


@torch.no_grad()
def run_test_iteration(config):
    coco_gt_filepath = os.path.join(config.dataset_root_dir, config.dataset_quality, "test", "annotations", "coco.json")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    coco = COCO(coco_gt_filepath)
    model = build_model_from_config(config)

    checkpoint = load_checkpoint(config)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    #model.summary(True)
    model.eval()

    predictions_to_dict_converter = PredictionsToDict()
    coco_dets = []
    test_dataloader = get_test_dataloader(config)
    for coco_batch in test_dataloader:
        images, targets = coco_batch
        images = images.to(device)
        predictions = model(images)
        for prediction, target in zip(predictions, targets):
            dets = predictions_to_dict_converter(prediction, target)
            if dets is not None:
                coco_dets += dets

    coco_result = coco.loadRes(coco_dets)
    coco_eval = COCOeval(coco, coco_result, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    coco_eval_list = coco_eval.stats[const.COCO_STATS_INDEXES].tolist()


