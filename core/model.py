import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights


class ConvBatchNormPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.PReLU(out_channels)

    def forward(self, input_tensor):
        output_tensor = self.conv(input_tensor)
        output_tensor = self.batch_norm(output_tensor)
        output_tensor = self.activation(output_tensor)

        return output_tensor


class GraphConvolutionNetwork(nn.Module):
    def __init__(self, node_count, channel_count):
        super().__init__()
        self.node_interaction = nn.Conv2d(node_count, node_count, 1, bias=False)
        self.activation = nn.PReLU(node_count)
        self.channel_interaction = nn.Linear(channel_count, channel_count, bias=False)

    def forward(self, input_tensor):
        output_tensor = self.node_interaction(input_tensor)
        output_tensor = self.activation(output_tensor + input_tensor)
        output_tensor = self.channel_interaction(output_tensor)

        return output_tensor


class StripPooling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.horizontal_pool = nn.AdaptiveAvgPool2d((1, None))
        self.vertical_pool = nn.AdaptiveAvgPool2d((None, 1))

        intermediate_channels = in_channels // 2 if in_channels >= 16 else in_channels
        self.horizontal_conv = nn.Sequential(nn.Conv2d(in_channels, intermediate_channels, 1, bias=False), nn.BatchNorm2d(intermediate_channels), nn.PReLU(intermediate_channels))
        self.vertical_conv = nn.Sequential(nn.Conv2d(in_channels, intermediate_channels, 1, bias=False), nn.BatchNorm2d(intermediate_channels), nn.PReLU(intermediate_channels))

        self.output_conv = nn.Sequential(nn.Conv2d(intermediate_channels, in_channels, 1, bias=False), nn.BatchNorm2d(in_channels))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        _, _, height, width = input_tensor.size()

        horizontal_pooled_tensor = self.horizontal_conv(self.horizontal_pool(input_tensor))
        horizontal_pooled_tensor = F.interpolate(horizontal_pooled_tensor, size=(height, width), mode="bilinear", align_corners=True)

        vertical_pooled_tensor = self.vertical_conv(self.vertical_pool(input_tensor))
        vertical_pooled_tensor = F.interpolate(vertical_pooled_tensor, size=(height, width), mode="bilinear", align_corners=True)

        output_tensor = self.output_conv(horizontal_pooled_tensor + vertical_pooled_tensor)
        output_tensor = input_tensor * self.sigmoid(output_tensor)

        return output_tensor


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        average_output = torch.mean(input_tensor, dim=1, keepdim=True)
        maximum_output, _ = torch.max(input_tensor, dim=1, keepdim=True)

        spatial_descriptors = torch.cat([average_output, maximum_output], dim=1)
        attention_map = self.conv(spatial_descriptors)

        output_tensor = input_tensor * self.sigmoid(attention_map)

        return output_tensor


class EfficientPyramidModule(nn.Module):
    def __init__(self, in_channels, out_channels, split_groups):
        super().__init__()

        self.split_groups = split_groups
        self.group_channels = out_channels // split_groups

        self.compression_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.PReLU(out_channels))
        self.pyramid_convs = nn.ModuleList()

        for group_index in range(self.split_groups):
            dilation = group_index + 1
            self.pyramid_convs.append(
                nn.Sequential(
                    nn.Conv2d(self.group_channels, self.group_channels, 3, stride=1, padding=dilation, dilation=dilation, groups=self.group_channels, bias=False),
                    nn.BatchNorm2d(self.group_channels),
                    nn.PReLU(self.group_channels),
                ),
            )

        self.fusion_conv = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.PReLU(out_channels))

    def _channel_shuffle(self, input_tensor, groups):
        batch_size, channel_count, height, width = input_tensor.size()
        channels_per_group = channel_count // groups

        input_tensor = input_tensor.view(batch_size, groups, channels_per_group, height, width)
        input_tensor = torch.transpose(input_tensor, dim0=1, dim1=2).contiguous()

        output_tensor = input_tensor.view(batch_size, -1, height, width)

        return output_tensor

    def forward(self, input_tensor):
        compressed_tensor = self.compression_conv(input_tensor)
        splits = torch.chunk(compressed_tensor, chunks=self.split_groups, dim=1)
        output_branches = [self.pyramid_convs[group_index](branch) for group_index, branch in enumerate(splits)]

        output_tensor = torch.cat(output_branches, dim=1)
        output_tensor = self._channel_shuffle(output_tensor, groups=self.split_groups)
        output_tensor = self.fusion_conv(output_tensor)
        output_tensor += compressed_tensor

        return output_tensor


class UpSimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.transposed_conv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-03)
        self.activation = nn.PReLU(out_channels)

    def forward(self, input_tensor):
        output_tensor = self.transposed_conv(input_tensor)
        output_tensor = self.batch_norm(output_tensor)
        output_tensor = self.activation(output_tensor)

        return output_tensor


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_last_layer=False, skip_connection_channels=0):
        super().__init__()

        self.is_last_layer = is_last_layer
        self.upsample_layer = UpSimpleBlock(in_channels, out_channels)

        if not is_last_layer:
            fusion_in_channels = out_channels + skip_connection_channels
            self.fusion_layer = ConvBatchNormPReLU(fusion_in_channels, out_channels, 3)

        self.output_layer = ConvBatchNormPReLU(out_channels, out_channels, 3)

    def forward(self, input_tensor, skip_features=None):
        upsampled_features = self.upsample_layer(input_tensor)

        if not self.is_last_layer and skip_features is not None:
            upsampled_features = torch.cat([upsampled_features, skip_features], dim=1)
            upsampled_features = self.fusion_layer(upsampled_features)

        output_tensor = self.output_layer(upsampled_features)

        return output_tensor


class DualBranchUpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_last_layer=False, skip_connection_channels=0):
        super().__init__()

        self.is_last_layer = is_last_layer
        self.fine_branch = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2, bias=False), nn.BatchNorm2d(out_channels, eps=1e-03), nn.PReLU(out_channels))
        self.coarse_branch = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels, eps=1e-03), nn.PReLU(out_channels))

        if not is_last_layer:
            fusion_in_channels = out_channels + skip_connection_channels
            self.fusion_layer = ConvBatchNormPReLU(fusion_in_channels, out_channels, 3)

        self.output_layer = ConvBatchNormPReLU(out_channels, out_channels, 3)

    def forward(self, input_tensor, skip_features=None):
        upsampled_features = self.fine_branch(input_tensor) + self.coarse_branch(input_tensor)

        if not self.is_last_layer and skip_features is not None:
            upsampled_features = self.fusion_layer(torch.cat([upsampled_features, skip_features], dim=1))

        output_tensor = self.output_layer(upsampled_features)

        return output_tensor


class ShuffleNetEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        backbone = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.DEFAULT)
        self.stem_conv = nn.Sequential(nn.Conv2d(5, 24, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(24), nn.ReLU(inplace=True))
        self._initialize_weights(backbone)

        self.maxpool = backbone.maxpool
        self.stage2 = backbone.stage2

        self.efficient_pyramid_module = EfficientPyramidModule(config.encoder_in_channels, config.encoder_out_channels, split_groups=config.encoder_epm_split_groups)
        self.half_skip_compressor = nn.Sequential(nn.Conv2d(24, config.encoder_half_skip_channels, 1, bias=False), nn.BatchNorm2d(config.encoder_half_skip_channels), nn.PReLU(config.encoder_half_skip_channels))
        self.quarter_skip_compressor = nn.Sequential(nn.Conv2d(24, config.encoder_quarter_skip_channels, 1, bias=False), nn.BatchNorm2d(config.encoder_quarter_skip_channels), nn.PReLU(config.encoder_quarter_skip_channels))

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

        encoder_features = self.efficient_pyramid_module(stage2_features)
        half_skip = self.half_skip_compressor(half_features)
        quarter_skip = self.quarter_skip_compressor(quarter_features)

        return encoder_features, half_skip, quarter_skip


class ContextAwareAttentionModule(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.bin_size = config.caam_bin_size
        self.class_activation_conv = nn.Conv2d(config.caam_in_channels, config.caam_activation_channels, 1)
        self.class_activation_pool = nn.AdaptiveAvgPool2d(config.caam_bin_size)
        self.sigmoid = nn.Sigmoid()

        bins_height, bins_width = config.caam_bin_size
        total_bins = bins_height * bins_width
        self.graph_convolution_network = GraphConvolutionNetwork(total_bins, config.caam_in_channels)
        self.local_to_global_conv = nn.Conv2d(total_bins, 1, 1)
        self.activation = nn.PReLU(1)

        inner_channels = config.caam_in_channels // 2
        self.projection_query = nn.Linear(config.caam_in_channels, inner_channels)
        self.projection_key = nn.Linear(config.caam_in_channels, inner_channels)
        self.projection_value = nn.Linear(config.caam_in_channels, inner_channels)
        self.output_projection = nn.Sequential(nn.Conv2d(inner_channels, config.caam_in_channels, 1, bias=False), nn.BatchNorm2d(config.caam_in_channels), nn.PReLU(config.caam_in_channels))

    def _patch_split(self, feature_map, bin_size):
        batch_size, channels, height, width = feature_map.size()
        bins_height, bins_width = bin_size

        patch_height = height // bins_height
        patch_width = width // bins_width

        patched_tensor = feature_map.view(batch_size, channels, bins_height, patch_height, bins_width, patch_width)
        patched_tensor = patched_tensor.permute(0, 2, 4, 3, 5, 1).contiguous()
        patched_tensor = patched_tensor.view(batch_size, -1, patch_height, patch_width, channels)

        return patched_tensor

    def _patch_recover(self, patched_tensor, bin_size):
        batch_size, _, patch_height, patch_width, channels = patched_tensor.size()
        bins_height, bins_width = bin_size

        height = patch_height * bins_height
        width = patch_width * bins_width

        feature_map = patched_tensor.view(batch_size, bins_height, bins_width, patch_height, patch_width, channels)
        feature_map = feature_map.permute(0, 5, 1, 3, 2, 4).contiguous()
        feature_map = feature_map.view(batch_size, channels, height, width)

        return feature_map

    def forward(self, input_tensor):
        residual = input_tensor

        activation_map = self.class_activation_conv(input_tensor)
        class_score = self.sigmoid(self.class_activation_pool(activation_map))

        patched_class_activation_map = self._patch_split(activation_map, self.bin_size)
        patched_features = self._patch_split(input_tensor, self.bin_size)

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
        output_tensor = self._patch_recover(output_tensor, self.bin_size)

        projected_tensor = self.output_projection(output_tensor)
        output_tensor = residual + projected_tensor

        return output_tensor


class TaskDecoder(nn.Module):
    def __init__(self, config, use_dual_branch_upsampling):
        super().__init__()
        block = DualBranchUpsamplingBlock if use_dual_branch_upsampling else UpConvBlock
        self.stage1 = block(config.decoder_in_channels, config.decoder_stage1_channels, skip_connection_channels=config.decoder_skip_channels)
        self.stage2 = block(config.decoder_stage1_channels, config.decoder_stage2_channels, skip_connection_channels=config.decoder_skip_channels)
        self.attention = nn.Sequential(StripPooling(config.decoder_stage2_channels), SpatialAttention(config.decoder_attention_kernel_size))
        self.output_head = block(config.decoder_stage2_channels, config.class_count, is_last_layer=True)

    def forward(self, latent_features, half_skip, quarter_skip):
        output_tensor = self.stage1(latent_features, quarter_skip)
        output_tensor = self.stage2(output_tensor, half_skip)
        output_tensor = self.attention(output_tensor)
        output_tensor = self.output_head(output_tensor)

        return output_tensor


class AnviaNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.shufflenet_encoder = ShuffleNetEncoder(config)
        self.context_aware_attention_module = ContextAwareAttentionModule(config)
        self.bottleneck = ConvBatchNormPReLU(config.bottleneck_in_channels, config.bottleneck_out_channels, config.bottleneck_kernel_size)
        self.drivable_area_decoder = TaskDecoder(config, use_dual_branch_upsampling=False)
        self.lane_line_decoder = TaskDecoder(config, use_dual_branch_upsampling=True)

    def forward(self, images):
        encoder_features, half_skip, quarter_skip = self.shufflenet_encoder(images)

        caam_features = self.context_aware_attention_module(encoder_features)
        latent_features = self.bottleneck(caam_features)

        drivable_area_predictions = self.drivable_area_decoder(latent_features, half_skip, quarter_skip)
        lane_line_predictions = self.lane_line_decoder(latent_features, half_skip, quarter_skip)

        return drivable_area_predictions, lane_line_predictions
