import json
import os

import pandas as pd
import torch.utils.data
from PIL import Image

from dataset.labels import PLAYER_LABEL


class NCAADataset(torch.utils.data.Dataset):
    IMAGE_SUFFIX = ".png"
    CSV_SUFFIX = "_info.csv"
    CSV_COLUMNS = ["x", "y", "w", "h"]

    def __init__(self, root_dir, transform=None, coco_api=False):
        assert os.path.exists(root_dir), f"Specified root path does not exist: {root_dir}"
        self.root_dir = root_dir
        self.transform = transform
        _, _, self.filenames = next(os.walk(root_dir))
        self.dataset = []
        self.coco_api = coco_api
        self.coco = {"images": [], "annotations": [],
                     "categories": [{"id": PLAYER_LABEL, "name": "player", "supercategory": "none"}]}
        self.annotation_id = 0

    def load_to_memory(self):
        for idx in range(len(self)):
            self.dataset.append(self.get_item_from_file(idx))

    def __len__(self):
        return len([name for name in self.filenames if name.endswith(self.IMAGE_SUFFIX)])

    def __getitem__(self, idx):
        return self.dataset[idx] if self.dataset else self.get_item_from_file(idx)

    def get_item_from_file(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        basename = f"{idx:04d}"
        img_name = os.path.join(self.root_dir, basename + self.IMAGE_SUFFIX)
        info_csv_name = os.path.join(self.root_dir, basename + self.CSV_SUFFIX)

        img = Image.open(img_name).convert("RGB")

        info_csv = pd.read_csv(info_csv_name, usecols=self.CSV_COLUMNS)
        imago = {"id": idx, "file_name": f"{basename}.png", "width": img.width, "height": img.height}
        self.coco["images"].append(imago)
        # remove invalid boxes
        info_csv = info_csv.loc[
                   (info_csv["w"] > 0) & (info_csv["h"] > 0) & (0 <= info_csv["x"]) & (info_csv["x"] < img.width) &
                   (0 <= info_csv["y"]) & (info_csv["y"] < img.height) & (0 < info_csv["x"] + info_csv["w"]) & (
                               info_csv["x"] + info_csv["w"] <= img.width) & (0 < info_csv["y"] + info_csv["h"]) & (
                               info_csv["y"] + info_csv["h"] <= img.height), :]

        for index, row in info_csv.iterrows():
            annotations = {"id": self.annotation_id, "segmentation": [], "area": row["h"] * row["w"], "iscrowd": 0,
                           "ignore": 0,
                           "image_id": idx, "bbox": row.tolist(), "category_id": PLAYER_LABEL}
            self.annotation_id += 1
            self.coco["annotations"].append(annotations)

        # change format of those bbs from x, y, w, h to x1, y1, x2, y2
        info_csv['w'] = info_csv['x'] + info_csv['w']
        info_csv['h'] = info_csv['y'] + info_csv['h']

        bbs = info_csv.values

        # we predict only one class - player
        labels = torch.ones((len(bbs),), dtype=torch.int64)

        boxes = torch.as_tensor(bbs, dtype=torch.float32)

        if self.transform:
            img, boxes, labels = self.transform((img, boxes, labels))

        if self.coco_api:
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = torch.tensor([idx])
            # dummy variables needed for training FasterRCNN
            target["area"] = torch.zeros((len(bbs),), dtype=torch.int64)
            target["iscrowd"] = torch.zeros((len(bbs),), dtype=torch.int64)
            return img, target

        return img, boxes, labels


if __name__ == "__main__":
    filepath = "/home/tperka/ncaa/LQ/train"
    ds = NCAADataset(filepath)
    for idx in range(len(ds)):
        abc = ds[idx]
    with open(f"{filepath}/annotations/coco.json", "w") as f:
        json.dump(ds.coco, f)
