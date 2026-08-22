import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_shuffle(input_tensor, groups):
    batch_size, channel_count, height, width = input_tensor.size()
    channels_per_group = channel_count // groups

    input_tensor = input_tensor.view(batch_size, groups, channels_per_group, height, width)
    input_tensor = torch.transpose(input_tensor, dim0=1, dim1=2).contiguous()

    output_tensor = input_tensor.view(batch_size, -1, height, width)

    return output_tensor


class ConvBatchNormPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
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
        self.node_interaction = nn.Conv2d(node_count, node_count, kernel_size=1, bias=False)
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
        self.horizontal_conv = nn.Sequential(nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, bias=False), nn.BatchNorm2d(intermediate_channels), nn.PReLU(intermediate_channels))
        self.vertical_conv = nn.Sequential(nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, bias=False), nn.BatchNorm2d(intermediate_channels), nn.PReLU(intermediate_channels))

        self.output_conv = nn.Sequential(nn.Conv2d(intermediate_channels, in_channels, kernel_size=1, bias=False), nn.BatchNorm2d(in_channels))
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
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        average_output = torch.mean(input_tensor, dim=1, keepdim=True)
        maximum_output, _ = torch.max(input_tensor, dim=1, keepdim=True)

        spatial_descriptors = torch.cat([average_output, maximum_output], dim=1)
        attention_map = self.conv(spatial_descriptors)

        output_tensor = input_tensor * self.sigmoid(attention_map)

        return output_tensor


class StageFusionModule(nn.Module):
    def __init__(self, fine_channels, coarse_channels, out_channels):
        super().__init__()

        self.fine_projection = ConvBatchNormPReLU(fine_channels, out_channels, kernel_size=1)
        self.coarse_projection = ConvBatchNormPReLU(coarse_channels, out_channels, kernel_size=1)

        fused_channels = out_channels * 2
        self.fusion_conv = ConvBatchNormPReLU(fused_channels, out_channels, kernel_size=1)

    def forward(self, fine_features, coarse_features):
        _, _, height, width = fine_features.size()

        upsampled_coarse_features = F.interpolate(coarse_features, size=(height, width), mode="bilinear", align_corners=False)

        projected_fine_features = self.fine_projection(fine_features)
        projected_coarse_features = self.coarse_projection(upsampled_coarse_features)

        fused_features = torch.cat([projected_fine_features, projected_coarse_features], dim=1)
        output_tensor = self.fusion_conv(fused_features)

        return output_tensor


class EfficientPyramidModule(nn.Module):
    def __init__(self, in_channels, out_channels, split_groups, dilations):
        super().__init__()

        self.split_groups = split_groups
        self.group_channels = out_channels // split_groups

        self.compression_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels), nn.PReLU(out_channels))
        self.pyramid_convs = nn.ModuleList()

        for dilation in dilations:
            self.pyramid_convs.append(
                nn.Sequential(
                    nn.Conv2d(self.group_channels, self.group_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, groups=self.group_channels, bias=False),
                    nn.BatchNorm2d(self.group_channels),
                    nn.PReLU(self.group_channels),
                ),
            )

        self.fusion_conv = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels), nn.PReLU(out_channels))

    def forward(self, input_tensor):
        compressed_tensor = self.compression_conv(input_tensor)
        splits = torch.chunk(compressed_tensor, chunks=self.split_groups, dim=1)
        output_branches = [self.pyramid_convs[group_index](branch) for group_index, branch in enumerate(splits)]

        output_tensor = torch.cat(output_branches, dim=1)
        output_tensor = channel_shuffle(output_tensor, groups=self.split_groups)
        output_tensor = self.fusion_conv(output_tensor)
        output_tensor += compressed_tensor

        return output_tensor


class SpatialGroupEnhance(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups
        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.ones(1, groups, 1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        batch_size, channels, height, width = input_tensor.size()
        channels_per_group = channels // self.groups
        batch_groups = batch_size * self.groups
        grouped_tensor = input_tensor.view(batch_groups, channels_per_group, height, width)

        global_features = self.average_pool(grouped_tensor)
        attention_map = (grouped_tensor * global_features).sum(dim=1, keepdim=True)
        attention_map = attention_map.view(batch_groups, -1)

        normalized_attention = attention_map - attention_map.mean(dim=1, keepdim=True)
        normalized_attention = normalized_attention / (attention_map.std(dim=1, keepdim=True) + 1e-5)

        scaled_attention = normalized_attention.view(batch_size, self.groups, height, width)
        scaled_attention = scaled_attention * self.weight + self.bias
        scaled_attention = scaled_attention.view(batch_groups, 1, height, width)

        output_tensor = grouped_tensor * self.sigmoid(scaled_attention)
        output_tensor = output_tensor.view(batch_size, channels, height, width)

        return output_tensor


class EfficientMultiScaleAttention(nn.Module):
    def __init__(self, in_channels, split_factor):
        super().__init__()

        self.split_factor = split_factor
        group_channels = in_channels // self.split_factor

        self.global_average_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.horizontal_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.vertical_pool = nn.AdaptiveAvgPool2d((1, None))

        self.group_norm = nn.GroupNorm(num_groups=group_channels, num_channels=group_channels)
        self.interaction_conv = nn.Conv2d(group_channels, group_channels, kernel_size=1)
        self.spatial_conv = nn.Conv2d(group_channels, group_channels, kernel_size=3, stride=1, padding=1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_tensor):
        batch_size, channels, height, width = input_tensor.size()
        channels_per_group = channels // self.split_factor
        batch_splits = batch_size * self.split_factor
        grouped_tensor = input_tensor.view(batch_splits, channels_per_group, height, width)

        horizontal_features = self.horizontal_pool(grouped_tensor)
        vertical_features = self.vertical_pool(grouped_tensor).transpose(dim0=2, dim1=3)

        concatenated_features = torch.cat([horizontal_features, vertical_features], dim=2)
        interacted_features = self.interaction_conv(concatenated_features)
        horizontal_attention, vertical_attention = torch.split(interacted_features, [height, width], dim=2)

        cross_spatial_features = grouped_tensor * horizontal_attention.sigmoid() * vertical_attention.transpose(dim0=2, dim1=3).sigmoid()
        cross_spatial_features = self.group_norm(cross_spatial_features)
        local_spatial_features = self.spatial_conv(grouped_tensor)

        cross_global_features = self.global_average_pool(cross_spatial_features).view(batch_splits, channels_per_group, 1).transpose(dim0=1, dim1=2)
        cross_weights = self.softmax(cross_global_features)
        reshaped_local_features = local_spatial_features.view(batch_splits, channels_per_group, -1)

        local_global_features = self.global_average_pool(local_spatial_features).view(batch_splits, channels_per_group, 1).transpose(dim0=1, dim1=2)
        local_weights = self.softmax(local_global_features)
        reshaped_cross_features = cross_spatial_features.view(batch_splits, channels_per_group, -1)

        fused_attention = torch.matmul(cross_weights, reshaped_local_features) + torch.matmul(local_weights, reshaped_cross_features)
        fused_attention = fused_attention.view(batch_splits, 1, height, width)

        output_tensor = (grouped_tensor * fused_attention.sigmoid()).view(batch_size, channels, height, width)

        return output_tensor


class FullScaleAttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_enhance_groups, multi_scale_attention_split_factor):
        super().__init__()

        self.action_channels = max(8, math.ceil(in_channels / 4 / 8) * 8)
        self.idle_channels = in_channels - self.action_channels

        self.partial_conv = nn.Conv2d(self.action_channels, self.action_channels, kernel_size=3, padding=1, groups=self.action_channels)
        self.spatial_group_enhance = SpatialGroupEnhance(groups=spatial_enhance_groups)

        self.mixed_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=4, bias=False)
        self.efficient_multi_scale_attention = EfficientMultiScaleAttention(out_channels, split_factor=multi_scale_attention_split_factor)

        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.Hardswish(inplace=True)

    def forward(self, input_tensor):
        action_features, idle_features = torch.split(input_tensor, [self.action_channels, self.idle_channels], dim=1)
        action_features = self.partial_conv(action_features)
        action_features = self.spatial_group_enhance(action_features)

        fused_features = torch.cat([action_features, idle_features], dim=1)
        mixed_features = self.mixed_conv(fused_features)
        shuffled_features = channel_shuffle(mixed_features, groups=4)

        attention_features = self.efficient_multi_scale_attention(shuffled_features)

        output_tensor = self.batch_norm(attention_features)
        output_tensor = self.activation(output_tensor)

        return output_tensor


class UpSimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.transposed_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
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
            self.fusion_layer = ConvBatchNormPReLU(fusion_in_channels, out_channels, kernel_size=3)

        self.output_layer = ConvBatchNormPReLU(out_channels, out_channels, kernel_size=3)

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
        self.fine_branch = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False), nn.BatchNorm2d(out_channels, eps=1e-03), nn.PReLU(out_channels))
        self.coarse_branch = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels, eps=1e-03), nn.PReLU(out_channels))

        if not is_last_layer:
            fusion_in_channels = out_channels + skip_connection_channels
            self.fusion_layer = ConvBatchNormPReLU(fusion_in_channels, out_channels, kernel_size=3)

        self.output_layer = ConvBatchNormPReLU(out_channels, out_channels, kernel_size=3)

    def forward(self, input_tensor, skip_features=None):
        upsampled_features = self.fine_branch(input_tensor) + self.coarse_branch(input_tensor)

        if not self.is_last_layer and skip_features is not None:
            upsampled_features = self.fusion_layer(torch.cat([upsampled_features, skip_features], dim=1))

        output_tensor = self.output_layer(upsampled_features)

        return output_tensor
