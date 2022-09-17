from pprint import pprint

from matplotlib import pyplot as plt

from dataset.NCAADataset import NCAADataset
from model.training.visualization import show_bbs
import pandas as pd


def check_bboxes(dataset):
    for idx in range(len(dataset)):
        img, target = dataset[idx]
        bbs = target["boxes"].detach().numpy()

        for row in bbs:
            if row[2] < row[0] or row[3] < row[1] or row[0] > img.width or row[1] > img.height:
                print(f"Error with bbs at index {idx}")
                print(bbs)
                fig, ax = plt.subplots()
                show_bbs(img, bbs, ax)


def check_resolutions_distribution(dataset):
    resolutions_dict = {}
    for idx in range(len(dataset)):
        img, target = dataset[idx]
        resolution = (img.width, img.height)
        if resolution in resolutions_dict.keys():
            resolutions_dict[resolution] += 1
        else:
            resolutions_dict[resolution] = 1


    print("Resolutions distribution in dataset:")
    pprint(resolutions_dict)


def check_bboxes_metrics(dataset):
    bboxes = []
    for idx in range(len(dataset)):
        img, boxes, labels = dataset[idx]
        bboxes += [x + [img.width, img.height] for x in boxes.tolist()]

    df = pd.DataFrame(bboxes, columns=['x1', 'y1', 'w', 'h', 'img_width', 'img_height'], copy=False)
    df['w'] -= df['x1']
    df['h'] -= df['y1']
    df['bbox_width_ratio'] = df['w'] / df['img_width']
    df['bbox_height_ratio'] = df['h'] / df['img_height']
    print(df.describe().to_string())

if __name__ == "__main__":
    dataset = NCAADataset("/home/tperka/Desktop/ncaa/LQ/train")
    check_bboxes_metrics(dataset)
