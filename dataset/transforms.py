import numpy as np
import torch
from PIL import Image

from torchvision import transforms
import torchvision.transforms.functional as TF

# Calculated using: https://towardsdatascience.com/how-to-calculate-the-mean-and-standard-deviation-normalizing-datasets-in-pytorch-704bd7d05f4c
NORMALIZATION_MEAN = [0.5178, 0.4368, 0.3387]
NORMALIZATION_STD = [0.2886, 0.2323, 0.1933]


denormalize_trans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.], std=[1.0 / e for e in NORMALIZATION_STD]),
                                        transforms.Normalize(mean=[-e for e in NORMALIZATION_MEAN], std=[1., 1., 1.])])

normalize_trans = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD)])


def tensor2image(image):
    img = denormalize_trans(image)
    temp = img.permute(1, 2, 0).cpu().numpy()
    return temp


def apply_transform_and_clip(boxes, labels, M, shape):
    """
    :param M: affine transformation matrix
    :param shape: (width, height) tuple
    :return:
    """
    assert len(boxes) == len(labels)
    if len(boxes) == 0:
        return boxes, labels

    ones = np.ones((len(boxes), 1))
    ext_pts1 = np.append(boxes[:, :2], ones, 1).transpose()     # Upper right corner
    ext_pts2 = np.append(boxes[:, 2:4], ones, 1).transpose()    # Lower left corner

    transformed_pts1 = np.dot(M[:2], ext_pts1).transpose()
    transformed_pts2 = np.dot(M[:2], ext_pts2).transpose()
    # We need to find out which corner is top right and which is bottom left, after the transform
    transformed_boxes = np.zeros_like(boxes)
    transformed_boxes[:, 0] = np.minimum(transformed_pts1[:, 0], transformed_pts2[:, 0])
    transformed_boxes[:, 1] = np.minimum(transformed_pts1[:, 1], transformed_pts2[:, 1])
    transformed_boxes[:, 2] = np.maximum(transformed_pts1[:, 0], transformed_pts2[:, 0])
    transformed_boxes[:, 3] = np.maximum(transformed_pts1[:, 1], transformed_pts2[:, 1])

    assert boxes.shape == transformed_boxes.shape
    clipped_boxes, cliped_labels = clip(transformed_boxes, labels, shape)

    return torch.as_tensor(clipped_boxes, dtype=torch.float32), cliped_labels


def clip(boxes, labels, shape):
    """

    :param boxes: list of (x1, y1, x2, y2) coordinates
    :param shape: (width, height) tuple
    :return:
    """
    box_contained = lambda e: 0 <= e[0] < shape[0] and 0 <= e[1] < shape[1] and 0 <= e[2] < shape[0] and 0 <= e[3] < shape[1]
    mask = [bool(box_contained(box)) for box in boxes]
    return boxes[mask], labels[mask]


class RandomAffineTransforms:
    def __init__(self, rotation_degrees=0, scale=(1,1), translation=(0, 0), hflip_prob=0.5):
        if isinstance(rotation_degrees, int):
            if rotation_degrees < 0:
                raise ValueError("If rotation degrees are specified as single int it should be non-negative")
            self.rotation_degrees = (-rotation_degrees, rotation_degrees)
        else:
            assert isinstance(rotation_degrees, (tuple, list) and len(rotation_degrees) == 2), "rotation degrees " \
                                                                                               "should be a list or " \
                                                                                               "tuple with lenght of 2"
            self.rotation_degrees = rotation_degrees

        if translation is not None:
            assert isinstance(translation, (tuple, list)) and len(translation) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translation:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translation = translation

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        self.hflip_prob = hflip_prob

    def draw_params(self, height, width):

        rotation_angle = np.random.uniform(self.rotation_degrees[0], self.rotation_degrees[1])
        if self.translation:
            max_dx = self.translation[0] * width
            max_dy = self.translation[0] * height
            translations = (np.round(np.random.uniform(-max_dx, max_dx)),
                            np.round(np.random.uniform(-max_dy, max_dy)))

            scale = np.random.uniform(self.scale[0], self.scale[1])

            return rotation_angle, translations, scale

    def __call__(self, sample):
        img, boxes, labels = sample
        height, width = img.height, img.width

        rotation_angle, translations, scale = self.draw_params(height, width)
        img_center = (width // 2 + 0.5, height // 2 + 0.5)
        coefficients = TF._get_inverse_affine_matrix(img_center, rotation_angle, translations, scale, (0.0, 0.0))
        inverse_affine_matrix = np.eye(3)
        inverse_affine_matrix[:2] = np.array(coefficients).reshape(2, 3)

        if np.random.rand() < self.hflip_prob:
            flip_matrix = np.eye(3)
            flip_matrix[0, 0] = -1
            flip_matrix[0, 2] = width - 1
            inverse_affine_matrix = flip_matrix @ inverse_affine_matrix

        img = img.transform((width, height), Image.AFFINE, inverse_affine_matrix[:2].reshape(6), Image.BILINEAR)

        affine_matrix = np.linalg.pinv(inverse_affine_matrix)
        boxes, labels = apply_transform_and_clip(boxes, labels, affine_matrix, (width, height))

        return img, boxes, labels


class ToTensorAndNormalize(object):
    # Convert image to tensors and normalize the image, ground truth is not changed
    def __init__(self):
        self.image_transforms = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD)])

    def __call__(self, sample):
        # numpy image: H x W x C
        # torch image: C X H X W
        image, boxes, labels = sample
        return self.image_transforms(image), boxes, labels

class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img, boxes, labels = sample
        w, h = img.width, img.height
        out_w, out_h = self.size
        #img.thumbnail((out_h, out_w))
        img = img.resize((out_w, out_h))
        scale_h = out_h / h
        scale_w = out_w / w

        output_boxes = torch.Tensor([[box[0] * scale_w, box[1] * scale_h, box[2] * scale_w, box[3] * scale_h] for box
                                     in boxes])

        return img, output_boxes, labels

class ColorJitter(object):
    def __init__(self, brightness=0., contrast=0., saturation=0., hue=0.):
        self.image_transform = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, sample):
        image, boxes, labels = sample
        return self.image_transform(image), boxes, labels


class TrainAugmentation(object):
    def __init__(self, size):
        self.size = size
        self.augment = transforms.Compose([
            Resize(size),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
            RandomAffineTransforms(rotation_degrees=8, scale=(0.8, 1.2), hflip_prob=0.5),
            ToTensorAndNormalize()
        ])

    def __call__(self, sample):
        return self.augment(sample)


class TestAugmentation:
    def __init__(self, size):
        self.transforms = transforms.Compose([
            Resize(size),
            ToTensorAndNormalize()
            ])

    def __call__(self, sample):
        return self.transforms(sample)


