import os.path

import torch

from dataset.NCAADataset import NCAADataset
from torch.utils.data import DataLoader

from dataset.transforms import TrainAugmentation, TestAugmentation

resolution_map = {
    "LQ": (490, 360),
    "HQ": (1280, 720)
}


def get_test_dataloader(config):
    test_dataset = NCAADataset(os.path.join(config.dataset_root_dir, config.dataset_quality, "test"),
                               transform=TestAugmentation(resolution_map[config.dataset_quality]), coco_api=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.test_batch, num_workers=config.dataloader_workers,
                                 pin_memory=True, collate_fn=collate_test)

    return test_dataloader


def get_dataloaders(dataset_root_dir, quality, train_batch_size, val_batch_size, num_workers,shuffle=True, train_coco=False):
    train_dataset = NCAADataset(os.path.join(dataset_root_dir, quality, "train"), transform=TrainAugmentation(resolution_map[quality]), coco_api=train_coco)
    train_dataloader = DataLoader(train_dataset, shuffle=shuffle, batch_size=train_batch_size,
                                  num_workers=num_workers, pin_memory=True, collate_fn=collate_train if not train_coco else collate_test)

    val_dataset = NCAADataset(os.path.join(dataset_root_dir, quality, "val"), transform=TestAugmentation(resolution_map[quality]), coco_api=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=num_workers, pin_memory=True, collate_fn=collate_test)

    return train_dataloader, val_dataloader


def collate_train(batch):
    images = torch.stack([e[0] for e in batch], dim=0)
    boxes = [e[1] for e in batch]
    labels = [e[2] for e in batch]
    return images, boxes, labels


def collate_test(batch):
    images = torch.stack([e[0] for e in batch], dim=0)
    targets = [e[1] for e in batch]
    return images, targets
