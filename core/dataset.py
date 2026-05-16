import os
import math
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def augment_hsv(image, h_gain=0.015, s_gain=0.7, v_gain=0.4):
    random_gains = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    image_dtype = image.dtype

    pixel_range = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((pixel_range * random_gains[0]) % 180).astype(image_dtype)
    lut_sat = np.clip(pixel_range * random_gains[1], 0, 255).astype(image_dtype)
    lut_val = np.clip(pixel_range * random_gains[2], 0, 255).astype(image_dtype)

    image_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(image_dtype)
    cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR, dst=image)


def random_perspective(combination, translate=0.1, shear=0.0, degrees=10, scale=0.25, perspective=0.0, border=(0, 0)):
    image, drivable_mask, lane_mask = combination
    height = image.shape[0] + border[0] * 2
    width = image.shape[1] + border[1] * 2

    translation_matrix = np.eye(3)
    translation_matrix[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    translation_matrix[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height

    shear_matrix = np.eye(3)
    shear_matrix[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    shear_matrix[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

    rotation_matrix = np.eye(3)
    angle = random.uniform(-degrees, degrees)
    scale_factor = random.uniform(1 - scale, 1 + scale)
    rotation_matrix[:2] = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale_factor)

    perspective_matrix = np.eye(3)
    perspective_matrix[2, 0] = random.uniform(-perspective, perspective)
    perspective_matrix[2, 1] = random.uniform(-perspective, perspective)

    center_matrix = np.eye(3)
    center_matrix[0, 2] = -image.shape[1] / 2
    center_matrix[1, 2] = -image.shape[0] / 2

    transform_matrix = translation_matrix @ shear_matrix @ rotation_matrix @ perspective_matrix @ center_matrix

    if (border[0] != 0) or (border[1] != 0) or (transform_matrix != np.eye(3)).any():
        if perspective:
            image = cv2.warpPerspective(image, transform_matrix, dsize=(width, height), borderValue=(114, 114, 114))
            drivable_mask = cv2.warpPerspective(drivable_mask, transform_matrix, dsize=(width, height), borderValue=0)
            lane_mask = cv2.warpPerspective(lane_mask, transform_matrix, dsize=(width, height), borderValue=0)
        else:
            image = cv2.warpAffine(image, transform_matrix[:2], dsize=(width, height), borderValue=(114, 114, 114))
            drivable_mask = cv2.warpAffine(drivable_mask, transform_matrix[:2], dsize=(width, height), borderValue=0)
            lane_mask = cv2.warpAffine(lane_mask, transform_matrix[:2], dsize=(width, height), borderValue=0)

    return (image, drivable_mask, lane_mask)


def random_crop(combination, crop_size=(540, 960)):
    image, drivable_mask, lane_mask = combination
    height, width, _ = image.shape
    crop_height, crop_width = crop_size

    if height > crop_height and width > crop_width:
        top = random.randint(0, height - crop_height)
        left = random.randint(0, width - crop_width)
        image = image[top : top + crop_height, left : left + crop_width]
        drivable_mask = drivable_mask[top : top + crop_height, left : left + crop_width]
        lane_mask = lane_mask[top : top + crop_height, left : left + crop_width]

    return (image, drivable_mask, lane_mask)


class BDD100KDataset(Dataset):
    def __init__(self, data_root, is_train=True, img_size=(360, 640)):
        self.is_train = is_train
        self.dataset_split = "train" if is_train else "val"
        self.image_dir = os.path.join(data_root, "images", self.dataset_split)
        self.drivable_dir = os.path.join(data_root, "segments", self.dataset_split)
        self.lane_dir = os.path.join(data_root, "lane", self.dataset_split)
        self.image_names = [name for name in os.listdir(self.image_dir) if name.endswith(".jpg")]
        self.target_height, self.target_width = img_size

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_path = os.path.join(self.image_dir, image_name)

        annotation_name = image_name.replace(".jpg", ".png")
        drivable_path = os.path.join(self.drivable_dir, annotation_name)
        lane_path = os.path.join(self.lane_dir, annotation_name)

        image = cv2.imread(image_path)
        drivable_mask = cv2.imread(drivable_path, cv2.IMREAD_GRAYSCALE)
        lane_mask = cv2.imread(lane_path, cv2.IMREAD_GRAYSCALE)

        if self.is_train:
            if random.random() < 0.5:
                combination = (image, drivable_mask, lane_mask)
                image, drivable_mask, lane_mask = random_perspective(combination)

            if random.random() < 0.5:
                augment_hsv(image)

            if random.random() < 0.1:
                combination = (image, drivable_mask, lane_mask)
                image, drivable_mask, lane_mask = random_crop(combination, crop_size=(540, 960))

            if random.random() < 0.5:
                image = cv2.flip(image, 1)
                drivable_mask = cv2.flip(drivable_mask, 1)
                lane_mask = cv2.flip(lane_mask, 1)

            if random.random() < 0.1:
                image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

            if random.random() < 0.1:
                image = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=0)

        image = cv2.resize(image, (self.target_width, self.target_height))
        drivable_mask = cv2.resize(drivable_mask, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)
        lane_mask = cv2.resize(lane_mask, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)

        drivable_mask = (drivable_mask > 0).astype(np.int64)
        lane_mask = (lane_mask > 0).astype(np.int64)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = image.transpose((2, 0, 1))

        image_tensor = torch.from_numpy(np.ascontiguousarray(image))
        drivable_tensor = torch.from_numpy(np.ascontiguousarray(drivable_mask))
        lane_tensor = torch.from_numpy(np.ascontiguousarray(lane_mask))

        return image_tensor, drivable_tensor, lane_tensor
