from model import builder


class PredictionsToDict:
    def __init__(self):
        self.id = 0

    def __call__(self, predictions, targets):
        scores = predictions["scores"].tolist()
        labels = predictions["labels"].tolist()
        boxes = predictions["boxes"].tolist()
        detections = []
        for score, label, box in zip(scores, labels, boxes):
            if box[2] <= box[0] or box[3] <= box[1]:
                continue

            detection = {"id": self.id, "segmentation": [], "area": (box[2] - box[0]) * (box[3] - box[1]),
                         "iscrowd": 0, "ignore": 0, "image_id": targets["image_id"].item(), "bbox":
                             [box[0], box[1], box[2] - box[0], box[3] - box[1]], "category_id": label, "score": score}
            detections.append(detection)
            self.id += 1
        return detections

def unpack_coco_batch(coco_batch):
    images, targets = coco_batch
    boxes = [target["boxes"] for target in targets]
    labels = [target["labels"] for target in targets]
    return images, boxes, labels


def build_model_from_config(config):
    return builder.build_basketball_detector(config.model_name, scale=config.scale,
                                              max_detections=config.max_detections,
                                              nms_threshold=config.nms_threshold,
                                              c_r_depth=config.classifier_regressor_depth,
                                              fpn_depth=config.fpn_lateral_depth,
                                              c_r_kernel_size=config.classifier_regressor_kernel_size,
                                              channel_attention=config.channel_attention_module,
                                              delta=config.delta, player_threshold=config.player_threshold)
