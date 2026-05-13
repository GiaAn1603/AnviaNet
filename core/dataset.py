import math
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def random_perspective(image, drivable_area_mask, lane_line_mask, translate=0.1, shear=0.0, degrees=10, scale=0.25, perspective=0.0, border=(0, 0)):
    height = image.shape[0] + border[0] * 2
    width = image.shape[1] + border[1] * 2

    translation_matrix = np.eye(3, dtype=np.float32)
    translation_matrix[0, 2] = random.uniform(a=0.5 - translate, b=0.5 + translate) * width
    translation_matrix[1, 2] = random.uniform(a=0.5 - translate, b=0.5 + translate) * height

    shear_matrix = np.eye(3, dtype=np.float32)
    shear_matrix[0, 1] = math.tan(random.uniform(a=-shear, b=shear) * math.pi / 180)
    shear_matrix[1, 0] = math.tan(random.uniform(a=-shear, b=shear) * math.pi / 180)

    rotation_matrix = np.eye(3, dtype=np.float32)
    angle = random.uniform(a=-degrees, b=degrees)
    scale_factor = random.uniform(a=1 - scale, b=1 + scale)
    rotation_matrix[:2] = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale_factor)

    perspective_matrix = np.eye(3, dtype=np.float32)
    perspective_matrix[2, 0] = random.uniform(a=-perspective, b=perspective)
    perspective_matrix[2, 1] = random.uniform(a=-perspective, b=perspective)

    center_matrix = np.eye(3, dtype=np.float32)
    center_matrix[0, 2] = -image.shape[1] / 2
    center_matrix[1, 2] = -image.shape[0] / 2

    transform_matrix = translation_matrix @ shear_matrix @ rotation_matrix @ perspective_matrix @ center_matrix

    if (border[0] != 0) or (border[1] != 0) or (transform_matrix != np.eye(3, dtype=np.float32)).any():
        if perspective:
            image = cv2.warpPerspective(image, transform_matrix, dsize=(width, height), borderValue=(114, 114, 114))
            drivable_area_mask = cv2.warpPerspective(drivable_area_mask, transform_matrix, dsize=(width, height), borderValue=0)
            lane_line_mask = cv2.warpPerspective(lane_line_mask, transform_matrix, dsize=(width, height), borderValue=0)
        else:
            image = cv2.warpAffine(image, transform_matrix[:2], dsize=(width, height), borderValue=(114, 114, 114))
            drivable_area_mask = cv2.warpAffine(drivable_area_mask, transform_matrix[:2], dsize=(width, height), borderValue=0)
            lane_line_mask = cv2.warpAffine(lane_line_mask, transform_matrix[:2], dsize=(width, height), borderValue=0)

    return image, drivable_area_mask, lane_line_mask


def augment_hsv(image, hue_gain=0.015, saturation_gain=0.7, value_gain=0.4):
    random_factors = np.random.uniform(low=-1.0, high=1.0, size=3) * [hue_gain, saturation_gain, value_gain] + 1
    hue_factor, saturation_factor, value_factor = random_factors

    hsv_image = cv2.cvtColor(image, code=cv2.COLOR_BGR2HSV)
    hue_channel, saturation_channel, value_channel = cv2.split(hsv_image)
    image_dtype = image.dtype

    pixel_values = np.arange(start=0, stop=256, dtype=np.int16)
    hue_lut = ((pixel_values * hue_factor) % 180).astype(dtype=image_dtype)
    saturation_lut = np.clip(pixel_values * saturation_factor, a_min=0, a_max=255).astype(dtype=image_dtype)
    value_lut = np.clip(pixel_values * value_factor, a_min=0, a_max=255).astype(dtype=image_dtype)

    augmented_hue = cv2.LUT(hue_channel, hue_lut)
    augmented_saturation = cv2.LUT(saturation_channel, saturation_lut)
    augmented_value = cv2.LUT(value_channel, value_lut)

    augmented_hsv = cv2.merge((augmented_hue, augmented_saturation, augmented_value))
    augmented_image = cv2.cvtColor(augmented_hsv, code=cv2.COLOR_HSV2BGR)

    return augmented_image


class BDD100KDataset(Dataset):
    def __init__(self, data_root_path, is_train, image_size, perspective_probability=0.5, hsv_probability=0.5, flip_probability=0.5):
        self.is_train = is_train
        self.dataset_split = "train" if is_train else "val"
        self.image_directory = os.path.join(data_root_path, "images", self.dataset_split)
        self.drivable_area_directory = os.path.join(data_root_path, "segments", self.dataset_split)
        self.lane_line_directory = os.path.join(data_root_path, "lane", self.dataset_split)
        self.image_names = [name for name in os.listdir(self.image_directory) if name.endswith(".jpg")]
        self.target_height, self.target_width = image_size

        self.perspective_probability = perspective_probability
        self.hsv_probability = hsv_probability
        self.flip_probability = flip_probability

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_path = os.path.join(self.image_directory, image_name)

        annotation_name = image_name.replace(".jpg", ".png")
        drivable_area_path = os.path.join(self.drivable_area_directory, annotation_name)
        lane_line_path = os.path.join(self.lane_line_directory, annotation_name)

        image = cv2.imread(image_path)
        drivable_area_mask = cv2.imread(drivable_area_path, flags=cv2.IMREAD_GRAYSCALE)
        lane_line_mask = cv2.imread(lane_line_path, flags=cv2.IMREAD_GRAYSCALE)

        if self.is_train:
            if random.random() < self.perspective_probability:
                image, drivable_area_mask, lane_line_mask = random_perspective(image, drivable_area_mask, lane_line_mask)

            if random.random() < self.hsv_probability:
                image = augment_hsv(image)

            if random.random() < self.flip_probability:
                image = cv2.flip(image, flipCode=1)
                drivable_area_mask = cv2.flip(drivable_area_mask, flipCode=1)
                lane_line_mask = cv2.flip(lane_line_mask, flipCode=1)

        image = cv2.resize(image, dsize=(self.target_width, self.target_height))
        drivable_area_mask = cv2.resize(drivable_area_mask, dsize=(self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)
        lane_line_mask = cv2.resize(lane_line_mask, dsize=(self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)

        drivable_area_mask = (drivable_area_mask > 0).astype(dtype=np.int64)
        lane_line_mask = (lane_line_mask > 0).astype(dtype=np.int64)

        image = cv2.cvtColor(image, code=cv2.COLOR_BGR2RGB)
        image = image.astype(dtype=np.float32) / 255.0
        image = image.transpose(2, 0, 1)

        image_tensor = torch.from_numpy(np.ascontiguousarray(image))
        drivable_area_tensor = torch.from_numpy(np.ascontiguousarray(drivable_area_mask))
        lane_line_tensor = torch.from_numpy(np.ascontiguousarray(lane_line_mask))

        return image_tensor, drivable_area_tensor, lane_line_tensor
