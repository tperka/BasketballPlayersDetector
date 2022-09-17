import math

import torchvision.ops
from torch import nn
import torch

import dataset.labels
from model import utils

class BasketballDetector(nn.Module):
    def __init__(self, name, base_net: nn.Module, bbox_regressor: nn.Module, classifier: nn.Module, player_downsampling, player_delta=3, max_player_detections=100, player_threshold=0.0, nms_threshold=0.7, padded_feature_map=False):
        super(BasketballDetector, self).__init__()

        assert isinstance(name, str), "Name of model should be a string"

        self.name = name
        self.base_net = base_net
        self.bbox_regressor = bbox_regressor
        self.classifier = classifier
        self.max_player_detections = max_player_detections
        self.player_threshold = player_threshold

        self.downsampling = player_downsampling
        self.delta = player_delta

        self.softmax = nn.Softmax(dim=1)
        self.nms_iou_threshold = nms_threshold
        self.nms = torchvision.ops.nms
        self.padded_feature_map = padded_feature_map


    def detect_from_map(self, confidence_map, loc_map):
        confidence_map = confidence_map[:, 1]
        batch_size, h, w = confidence_map.shape[0], confidence_map.shape[1], confidence_map.shape[2]
        confidence_map = confidence_map.view(batch_size, -1)

        values, indices = torch.sort(confidence_map, dim=-1, descending=True)
        if self.max_player_detections < indices.shape[1]:
            indices = indices[:, :self.max_player_detections]

        xc = indices % w
        yc = indices // w

        xc = xc.float() * self.downsampling + (self.downsampling - 1.) / 2
        yc = yc.float() * self.downsampling + (self.downsampling - 1.) / 2

        loc_map = loc_map.view(batch_size, 4, -1)
        loc_map[:, 0] *= w * self.downsampling
        loc_map[:, 2] *= w * self.downsampling
        loc_map[:, 1] *= h * self.downsampling
        loc_map[:, 3] *= h * self.downsampling

        detections = torch.zeros((batch_size, self.max_player_detections, 5), dtype=float).to(confidence_map.device)

        for n in range(batch_size):
            temp = loc_map[n, :, indices[n]]

            bx = xc[n] + temp[0]
            by = yc[n] + temp[1]

            detections[n, :, 0] = bx - 0.5 * temp[2]  # x1
            detections[n, :, 2] = bx + 0.5 * temp[2]  # x2
            detections[n, :, 1] = by - 0.5 * temp[3]  # y1
            detections[n, :, 3] = by + 0.5 * temp[3]  # y2
            detections[n, :, 4] = values[n, :self.max_player_detections]

        return detections

    def detect(self, feature_map, bbox):
        detections = self.detect_from_map(feature_map, bbox)

        result = []
        for detection in detections:
            detection = detection[detection[..., 4] >= self.player_threshold]

            boxes = detection[..., 0:4]
            scores = detection[..., 4]

            labels = torch.tensor([dataset.labels.PLAYER_LABEL] * len(detection), dtype=torch.int64)
            indices = self.nms(boxes, scores, iou_threshold=self.nms_iou_threshold)
            boxes_with_scores = {"boxes": boxes[indices], "labels": labels[indices], "scores": scores[indices]}
            result.append(boxes_with_scores)

        return result

    def create_groundtruth_maps(self, boxes, labels, image_shape):
        minibatch_size = len(boxes)

        w, h = image_shape
        confidence_map_height = h // self.downsampling
        confidence_map_width = w // self.downsampling
        if self.padded_feature_map:
            width = w / self.downsampling
            height = h / self.downsampling
            confidence_map_width = math.ceil(width)
            confidence_map_height = math.ceil(height)

        target_loc_map = torch.zeros([minibatch_size, confidence_map_height, confidence_map_width, 4], dtype=torch.float)
        target_conf_map = torch.zeros([minibatch_size, confidence_map_height, confidence_map_width], dtype=torch.long)

        for idx, (boxes, labels) in enumerate(zip(boxes, labels)):
            for box, label in zip(boxes, labels):
                bbox_center_x, bbox_center_y, bbox_width, bbox_height = utils.get_box_metrics(box)

                x1, y1, x2, y2 = self.get_positive_cells_for_bbox(bbox_center_x, bbox_center_y, confidence_map_width, confidence_map_height)

                target_conf_map[idx, y1:y2 + 1, x1:x2 + 1] = label
                pixel_coords_x = torch.tensor(range(x1, x2 + 1)).float() * self.downsampling + (self.downsampling - 1) / 2
                pixel_coords_y = torch.tensor(range(y1, y2 + 1)).float() * self.downsampling + (
                        self.downsampling - 1) / 2

                pixel_coords_x = (bbox_center_x - pixel_coords_x) / w
                pixel_coords_y = (bbox_center_y - pixel_coords_y) / h

                target_loc_map[idx, y1:y2 + 1, x1:x2 + 1, 0] = pixel_coords_x.unsqueeze(0)
                target_loc_map[idx, y1:y2 + 1, x1:x2 + 1, 1] = pixel_coords_y.unsqueeze(1)

                target_loc_map[idx, y1:y2 + 1, x1:x2 + 1, 2] = bbox_width / w
                target_loc_map[idx, y1:y2 + 1, x1:x2 + 1, 3] = bbox_height / h

        return target_loc_map, target_conf_map

    def get_positive_cells_for_bbox(self, bbox_center_x, bbox_center_y, conf_width, conf_height):
        cell_x = int(bbox_center_x / self.downsampling)
        cell_y = int(bbox_center_y / self.downsampling)
        if isinstance(self.delta, tuple):
            x_delta, y_delta = self.delta
        else:
            x_delta = y_delta = self.delta

        x1 = max(cell_x - x_delta // 2, 0)
        x2 = min(cell_x + x_delta // 2, conf_width - 1)
        y1 = max(cell_y - y_delta // 2, 0)
        y2 = min(cell_y + y_delta // 2, conf_height - 1)
        return x1, y1, x2, y2

    def forward(self, x):
        height, width = x.shape[2], x.shape[3]

        x = self.base_net(x)

        assert x.shape[2] - height // self.downsampling <= 1
        assert x.shape[3] - width // self.downsampling <= 1

        player_feature_map = self.classifier(x)
        player_bboxes = self.bbox_regressor(x)

        if self.training:
            player_feature_map = player_feature_map.permute(0, 2, 3, 1).contiguous()
            player_bboxes = player_bboxes.permute(0, 2, 3, 1).contiguous()

            output = (player_bboxes, player_feature_map)
        else:
            player_feature_map = self.softmax(player_feature_map)
            output = self.detect(player_feature_map, player_bboxes)

        return output

    def summary(self, with_arch = False):
        print(f"Model name: {self.name}")
        if with_arch:
            print("Base network:")
            print(self.base_net)
            print("")
            print("Classifier:")
            print(self.classifier)
            print("")
            print("Bounding box regressor:")
            print(self.bbox_regressor)

        print("***************** Number of parameters *****************")
        params_all, params_trainable = utils.count_model_parameters(self.base_net)
        print(f"Base network (all/trainable): {params_all} / {params_trainable}")

        params_all, params_trainable = utils.count_model_parameters(self.classifier)
        print(f"Classifier (all/trainable): {params_all} / {params_trainable}")

        params_all, params_trainable = utils.count_model_parameters(self.bbox_regressor)
        print(f"Bounding box regressor (all/trainable): {params_all} / {params_trainable}")

        params_all, params_trainable = utils.count_model_parameters(self)
        print(f"Total (all/trainable): {params_all} / {params_trainable}")

