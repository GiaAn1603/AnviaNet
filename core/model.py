import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights

from core.modules import (
    ConvBatchNormPReLU,
    GraphConvolutionNetwork,
    StripPooling,
    SpatialAttention,
    StageFusionModule,
    EfficientPyramidModule,
    FullScaleAttentionModule,
    UpConvBlock,
    DualBranchUpsamplingBlock,
)


class ShuffleNetEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        backbone = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.DEFAULT)
        self.stem_conv = nn.Sequential(nn.Conv2d(5, 24, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(24), nn.ReLU(inplace=True))
        self._initialize_weights(backbone)

        self.maxpool = backbone.maxpool
        self.stage2 = backbone.stage2
        self.stage3 = backbone.stage3[: config.encoder_stage3_block_count]

        self.stage_fusion_module = StageFusionModule(config.encoder_stage2_channels, config.encoder_stage3_channels, config.encoder_out_channels)
        self.efficient_pyramid_module = EfficientPyramidModule(config.encoder_out_channels, config.encoder_out_channels, split_groups=config.encoder_epm_split_groups, dilations=config.encoder_epm_dilations)
        self.full_scale_attention_module = FullScaleAttentionModule(config.encoder_out_channels, config.encoder_out_channels, spatial_enhance_groups=config.encoder_fsa_sge_groups, multi_scale_attention_split_factor=config.encoder_fsa_ema_split_factor)

        self.half_skip_compressor = nn.Sequential(nn.Conv2d(24, config.encoder_half_skip_channels, kernel_size=1, bias=False), nn.BatchNorm2d(config.encoder_half_skip_channels), nn.PReLU(config.encoder_half_skip_channels))
        self.quarter_skip_compressor = nn.Sequential(nn.Conv2d(24, config.encoder_quarter_skip_channels, kernel_size=1, bias=False), nn.BatchNorm2d(config.encoder_quarter_skip_channels), nn.PReLU(config.encoder_quarter_skip_channels))

    @torch.no_grad()
    def _initialize_weights(self, backbone):
        self.stem_conv[0].weight[:, :3, :, :] = backbone.conv1[0].weight
        self.stem_conv[0].weight[:, 3:, :, :].fill_(0.0)
        self.stem_conv[1].load_state_dict(backbone.conv1[1].state_dict())

    def _add_coordinates(self, images):
        batch_size, _, height, width = images.size()

        x_range = torch.arange(height, dtype=torch.float32, device=images.device).view(1, 1, height, 1)
        x_channel = x_range.expand(batch_size, 1, height, width)
        x_channel = x_channel / (height - 1) * 2 - 1

        y_range = torch.arange(width, dtype=torch.float32, device=images.device).view(1, 1, 1, width)
        y_channel = y_range.expand(batch_size, 1, height, width)
        y_channel = y_channel / (width - 1) * 2 - 1

        x_channel = x_channel.to(dtype=images.dtype)
        y_channel = y_channel.to(dtype=images.dtype)

        augmented_images = torch.cat([images, x_channel, y_channel], dim=1)

        return augmented_images

    def forward(self, images):
        augmented_images = self._add_coordinates(images)
        half_features = self.stem_conv(augmented_images)

        quarter_features = self.maxpool(half_features)
        stage2_features = self.stage2(quarter_features)
        stage3_features = self.stage3(stage2_features)

        fused_stage_features = self.stage_fusion_module(stage2_features, stage3_features)
        encoder_features = self.efficient_pyramid_module(fused_stage_features)
        encoder_features = self.full_scale_attention_module(encoder_features)

        half_skip = self.half_skip_compressor(half_features)
        quarter_skip = self.quarter_skip_compressor(quarter_features)

        return encoder_features, half_skip, quarter_skip


class ContextAwareAttentionModule(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.bin_height = config.caam_bin_height
        self.bin_width = config.caam_bin_width

        self.class_activation_conv = nn.Conv2d(config.caam_in_channels, config.caam_activation_channels, kernel_size=1)
        self.class_activation_pool = nn.AdaptiveAvgPool2d((config.caam_bin_height, config.caam_bin_width))
        self.sigmoid = nn.Sigmoid()

        total_bins = self.bin_height * self.bin_width
        self.graph_convolution_network = GraphConvolutionNetwork(total_bins, config.caam_in_channels)
        self.local_to_global_conv = nn.Conv2d(total_bins, 1, kernel_size=1)
        self.activation = nn.PReLU(1)

        inner_channels = config.caam_in_channels // 4
        self.projection_query = nn.Linear(config.caam_in_channels, inner_channels)
        self.projection_key = nn.Linear(config.caam_in_channels, inner_channels)
        self.projection_value = nn.Linear(config.caam_in_channels, inner_channels)
        self.output_projection = nn.Sequential(nn.Conv2d(inner_channels, config.caam_in_channels, kernel_size=1, bias=False), nn.BatchNorm2d(config.caam_in_channels), nn.PReLU(config.caam_in_channels))

    def _patch_split(self, feature_map, bin_height, bin_width):
        batch_size, channels, height, width = feature_map.size()

        patch_height = height // bin_height
        patch_width = width // bin_width

        patched_tensor = feature_map.view(batch_size, channels, bin_height, patch_height, bin_width, patch_width)
        patched_tensor = patched_tensor.permute(0, 2, 4, 3, 5, 1).contiguous()
        patched_tensor = patched_tensor.view(batch_size, -1, patch_height, patch_width, channels)

        return patched_tensor

    def _patch_recover(self, patched_tensor, bin_height, bin_width):
        batch_size, _, patch_height, patch_width, channels = patched_tensor.size()

        height = patch_height * bin_height
        width = patch_width * bin_width

        feature_map = patched_tensor.view(batch_size, bin_height, bin_width, patch_height, patch_width, channels)
        feature_map = feature_map.permute(0, 5, 1, 3, 2, 4).contiguous()
        feature_map = feature_map.view(batch_size, channels, height, width)

        return feature_map

    def forward(self, input_tensor):
        residual = input_tensor

        activation_map = self.class_activation_conv(input_tensor)
        class_score = self.sigmoid(self.class_activation_pool(activation_map))

        patched_class_activation_map = self._patch_split(activation_map, bin_height=self.bin_height, bin_width=self.bin_width)
        patched_features = self._patch_split(input_tensor, bin_height=self.bin_height, bin_width=self.bin_width)

        batch_size = patched_class_activation_map.shape[0]
        patch_height, patch_width = patched_class_activation_map.shape[2], patched_class_activation_map.shape[3]
        activation_channels, feature_channels = patched_class_activation_map.shape[-1], patched_features.shape[-1]

        pixels_per_patch = patch_height * patch_width
        patched_class_activation_map = patched_class_activation_map.view(batch_size, -1, pixels_per_patch, activation_channels)
        patched_features = patched_features.view(batch_size, -1, pixels_per_patch, feature_channels)

        bin_confidence = class_score.view(batch_size, activation_channels, -1).transpose(dim0=1, dim1=2)
        bin_confidence = bin_confidence.unsqueeze(dim=3) if bin_confidence.dim() == 3 else bin_confidence
        pixel_confidence = F.softmax(patched_class_activation_map, dim=2)

        local_features = torch.matmul(pixel_confidence.transpose(dim0=2, dim1=3), patched_features) * bin_confidence
        local_features = self.graph_convolution_network(local_features)

        global_features = self.local_to_global_conv(local_features)
        global_features = self.activation(global_features).repeat(1, patched_features.shape[1], 1, 1)

        query = self.projection_query(patched_features)
        key = self.projection_key(local_features)
        value = self.projection_value(global_features)

        affinity_matrix = torch.matmul(query, key.transpose(dim0=2, dim1=3))
        affinity_matrix = F.softmax(affinity_matrix, dim=-1)

        output_tensor = torch.matmul(affinity_matrix, value)
        output_tensor = output_tensor.view(batch_size, -1, patch_height, patch_width, value.shape[-1])
        output_tensor = self._patch_recover(output_tensor, bin_height=self.bin_height, bin_width=self.bin_width)

        projected_tensor = self.output_projection(output_tensor)
        output_tensor = residual + projected_tensor

        return output_tensor


class TaskDecoder(nn.Module):
    def __init__(self, config, decoder_config, use_dual_branch_upsampling):
        super().__init__()
        block = DualBranchUpsamplingBlock if use_dual_branch_upsampling else UpConvBlock
        self.stage1 = block(decoder_config.in_channels, decoder_config.stage1_channels, skip_connection_channels=decoder_config.skip_channels)
        self.stage2 = block(decoder_config.stage1_channels, decoder_config.stage2_channels, skip_connection_channels=decoder_config.skip_channels)
        self.attention = nn.Sequential(StripPooling(decoder_config.stage2_channels), SpatialAttention(decoder_config.attention_kernel_size))
        self.output_head = block(decoder_config.stage2_channels, config.class_count, is_last_layer=True)

    def forward(self, latent_features, half_skip, quarter_skip):
        output_tensor = self.stage1(latent_features, quarter_skip)
        output_tensor = self.stage2(output_tensor, half_skip)
        output_tensor = self.attention(output_tensor)
        output_tensor = self.output_head(output_tensor)

        return output_tensor


class AnviaNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.shufflenet_encoder = ShuffleNetEncoder(config)
        self.context_aware_attention_module = ContextAwareAttentionModule(config)
        self.bottleneck = ConvBatchNormPReLU(config.bottleneck_in_channels, config.bottleneck_out_channels, kernel_size=config.bottleneck_kernel_size)
        self.drivable_area_decoder = TaskDecoder(config, config.drivable_area_decoder, use_dual_branch_upsampling=False)
        self.lane_line_decoder = TaskDecoder(config, config.lane_line_decoder, use_dual_branch_upsampling=True)

        auxiliary_channels = config.encoder_out_channels // 2
        self.drivable_area_auxiliary_head = nn.Sequential(
            nn.Conv2d(config.encoder_out_channels, auxiliary_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(auxiliary_channels),
            nn.PReLU(auxiliary_channels),
            nn.Conv2d(auxiliary_channels, config.class_count, kernel_size=1),
        )
        self.lane_line_auxiliary_head = nn.Sequential(
            nn.Conv2d(config.encoder_out_channels, auxiliary_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(auxiliary_channels),
            nn.PReLU(auxiliary_channels),
            nn.Conv2d(auxiliary_channels, config.class_count, kernel_size=1),
        )

    def forward(self, images):
        encoder_features, half_skip, quarter_skip = self.shufflenet_encoder(images)

        caam_features = self.context_aware_attention_module(encoder_features)
        latent_features = self.bottleneck(caam_features)

        drivable_area_features = latent_features[:, : self.config.drivable_area_decoder.in_channels, :, :]
        lane_line_features = latent_features[:, -self.config.lane_line_decoder.in_channels :, :, :]

        drivable_area_half_skip_features = half_skip[:, : self.config.drivable_area_decoder.skip_channels, :, :]
        lane_line_half_skip_features = half_skip[:, -self.config.lane_line_decoder.skip_channels :, :, :]

        drivable_area_quarter_skip_features = quarter_skip[:, : self.config.drivable_area_decoder.skip_channels, :, :]
        lane_line_quarter_skip_features = quarter_skip[:, -self.config.lane_line_decoder.skip_channels :, :, :]

        drivable_area_predictions = self.drivable_area_decoder(drivable_area_features, drivable_area_half_skip_features, drivable_area_quarter_skip_features)
        lane_line_predictions = self.lane_line_decoder(lane_line_features, lane_line_half_skip_features, lane_line_quarter_skip_features)

        if self.training:
            drivable_area_auxiliary_predictions = self.drivable_area_auxiliary_head(encoder_features)
            lane_line_auxiliary_predictions = self.lane_line_auxiliary_head(encoder_features)

            return drivable_area_predictions, lane_line_predictions, drivable_area_auxiliary_predictions, lane_line_auxiliary_predictions

        return drivable_area_predictions, lane_line_predictions
