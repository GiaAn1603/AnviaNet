from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class TaskDecoderConfig:
    in_channels: int
    stage1_channels: int
    stage2_channels: int
    skip_channels: int
    attention_kernel_size: int


@dataclass
class LossConfig:
    drivable_area_class_id: int = 1
    drivable_area_tversky_alpha: float = 0.7
    drivable_area_tversky_gamma: float = 1.33333333333
    drivable_area_ohem_ratio: float = 0.7
    drivable_area_lovasz_weight: float = 0.1

    lane_line_class_id: int = 1
    lane_line_tversky_alpha: float = 0.9
    lane_line_tversky_gamma: float = 1.33333333333
    lane_line_ohem_ratio: float = 0.3
    lane_line_cldice_weight: float = 0.1
    lane_line_cldice_iterations: int = 5

    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    auxiliary_weight: float = 0.4

    warmup_epochs: float = 5.0


@dataclass
class AnviaNetConfig:
    image_height: int = 360
    image_width: int = 640
    class_count: int = 2

    encoder_stage2_channels: int = 116
    encoder_stage3_channels: int = 232
    encoder_stage3_block_count: int = 2
    encoder_out_channels: int = 128
    encoder_epm_split_groups: int = 4
    encoder_fsa_sge_groups: int = 8
    encoder_fsa_ema_split_factor: int = 8
    encoder_half_skip_channels: int = 40
    encoder_quarter_skip_channels: int = 40

    caam_in_channels: int = 128
    caam_activation_channels: int = 32
    caam_bin_size: Tuple[int, int] = (3, 4)

    bottleneck_in_channels: int = 128
    bottleneck_out_channels: int = 384
    bottleneck_kernel_size: int = 1

    drivable_area_decoder: TaskDecoderConfig = field(default_factory=lambda: TaskDecoderConfig(in_channels=64, stage1_channels=32, stage2_channels=8, skip_channels=16, attention_kernel_size=7))
    lane_line_decoder: TaskDecoderConfig = field(default_factory=lambda: TaskDecoderConfig(in_channels=320, stage1_channels=24, stage2_channels=8, skip_channels=24, attention_kernel_size=7))

    loss: LossConfig = field(default_factory=LossConfig)

    def __post_init__(self):
        downsample_factor = 8
        feature_height = self.image_height // downsample_factor
        feature_width = self.image_width // downsample_factor

        if feature_height % self.caam_bin_size[0] != 0:
            raise ValueError(f"Feature map height ({feature_height}) % CAAM bin height ({self.caam_bin_size[0]}) != 0. Adjust image_height or caam_bin_size.")

        if feature_width % self.caam_bin_size[1] != 0:
            raise ValueError(f"Feature map width ({feature_width}) % CAAM bin width ({self.caam_bin_size[1]}) != 0. Adjust image_width or caam_bin_size.")
