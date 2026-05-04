from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class LossConfig:
    tversky_da_alpha: float = 0.5
    tversky_da_gamma: float = 1.0
    tversky_ll_alpha: float = 0.5
    tversky_ll_gamma: float = 1.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    ohem_ratio_da: float = 0.7
    ohem_ratio_ll: float = 0.3


@dataclass
class AnviaNetConfig:
    img_height: int = 360
    img_width: int = 640
    num_classes: int = 2
    lane_class_id: int = 1

    encoder_in_channels: int = 116
    encoder_out_channels: int = 128

    caam_in_channels: int = 128
    caam_num_classes: int = 128
    caam_bin_size: Tuple[int, int] = (3, 4)

    bottleneck_in_channels: int = 128
    bottleneck_out_channels: int = 64

    decoder_in_channels: int = 64
    decoder_skip_channels: int = 12

    loss: LossConfig = field(default_factory=LossConfig)

    def __post_init__(self):
        downsample_factor = 8
        feat_height = self.img_height // downsample_factor
        feat_width = self.img_width // downsample_factor

        if feat_height % self.caam_bin_size[0] != 0:
            raise ValueError(
                f"Feature map height ({feat_height}) must be divisible by CAAM bin height ({self.caam_bin_size[0]}). "
                f"Please adjust img_height (currently {self.img_height}) or caam_bin_size."
            )

        if feat_width % self.caam_bin_size[1] != 0:
            raise ValueError(
                f"Feature map width ({feat_width}) must be divisible by CAAM bin width ({self.caam_bin_size[1]}). "
                f"Please adjust img_width (currently {self.img_width}) or caam_bin_size."
            )
