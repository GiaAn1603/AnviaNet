from dataclasses import dataclass, field
from typing import Tuple


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

    warmup_epochs: float = 5.0


@dataclass
class AnviaNetConfig:
    image_height: int = 360
    image_width: int = 640
    class_count: int = 2

    encoder_in_channels: int = 116
    encoder_out_channels: int = 128
    encoder_half_skip_channels: int = 12
    encoder_quarter_skip_channels: int = 12

    caam_in_channels: int = 128
    caam_activation_channels: int = 128
    caam_bin_size: Tuple[int, int] = (3, 4)

    bottleneck_in_channels: int = 128
    bottleneck_out_channels: int = 64
    bottleneck_kernel_size: int = 3

    decoder_in_channels: int = 64
    decoder_stage1_channels: int = 32
    decoder_stage2_channels: int = 8
    decoder_skip_channels: int = 12
    decoder_attention_kernel_size: int = 7

    loss: LossConfig = field(default_factory=LossConfig)

    def __post_init__(self):
        downsample_factor = 8
        feature_height = self.image_height // downsample_factor
        feature_width = self.image_width // downsample_factor

        if feature_height % self.caam_bin_size[0] != 0:
            raise ValueError(f"Feature map height ({feature_height}) % CAAM bin height ({self.caam_bin_size[0]}) != 0. Adjust image_height or caam_bin_size.")

        if feature_width % self.caam_bin_size[1] != 0:
            raise ValueError(f"Feature map width ({feature_width}) % CAAM bin width ({self.caam_bin_size[1]}) != 0. Adjust image_width or caam_bin_size.")
