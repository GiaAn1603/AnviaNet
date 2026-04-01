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


class ShuffleNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, half_skip_channels, quarter_skip_channels):
        super().__init__()
        backbone = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.DEFAULT)
        self.stem_conv = backbone.conv1
        self.maxpool = backbone.maxpool
        self.stage2 = backbone.stage2
        self.bottleneck = ConvBatchNormPReLU(in_channels, out_channels, 1)
        self.half_skip_compressor = nn.Sequential(nn.Conv2d(24, half_skip_channels, 1, bias=False), nn.BatchNorm2d(half_skip_channels), nn.PReLU(half_skip_channels))
        self.quarter_skip_compressor = nn.Sequential(nn.Conv2d(24, quarter_skip_channels, 1, bias=False), nn.BatchNorm2d(quarter_skip_channels), nn.PReLU(quarter_skip_channels))

    def forward(self, images):
        half_features = self.stem_conv(images)
        quarter_features = self.maxpool(half_features)
        stage2_features = self.stage2(quarter_features)

        encoder_features = self.bottleneck(stage2_features)
        half_skip = self.half_skip_compressor(half_features)
        quarter_skip = self.quarter_skip_compressor(quarter_features)

        return encoder_features, half_skip, quarter_skip


class ContextAwareAttentionModule(nn.Module):
    def __init__(self, in_channels, activation_channels, bin_size):
        super().__init__()

        self.bin_size = bin_size
        self.class_activation_conv = nn.Conv2d(in_channels, activation_channels, 1)
        self.class_activation_pool = nn.AdaptiveAvgPool2d(bin_size)
        self.sigmoid = nn.Sigmoid()

        bins_height, bins_width = bin_size
        total_bins = bins_height * bins_width
        self.graph_convolution_network = GraphConvolutionNetwork(total_bins, in_channels)
        self.local_to_global_conv = nn.Conv2d(total_bins, 1, 1)
        self.activation = nn.PReLU(1)

        inner_channels = in_channels // 2
        self.projection_query = nn.Linear(in_channels, inner_channels)
        self.projection_key = nn.Linear(in_channels, inner_channels)
        self.projection_value = nn.Linear(in_channels, inner_channels)
        self.output_projection = nn.Sequential(nn.Conv2d(inner_channels, in_channels, 1, bias=False), nn.BatchNorm2d(in_channels), nn.PReLU(in_channels))

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
    def __init__(self, in_channels, class_count, stage1_channels, stage2_channels, skip_connection_channels):
        super().__init__()
        self.stage1 = UpConvBlock(in_channels, stage1_channels, skip_connection_channels=skip_connection_channels)
        self.stage2 = UpConvBlock(stage1_channels, stage2_channels, skip_connection_channels=skip_connection_channels)
        self.output_head = UpConvBlock(stage2_channels, class_count, is_last_layer=True)

    def forward(self, latent_features, half_skip, quarter_skip):
        output_tensor = self.stage1(latent_features, quarter_skip)
        output_tensor = self.stage2(output_tensor, half_skip)
        output_tensor = self.output_head(output_tensor)

        return output_tensor


class AnviaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.shufflenet_encoder = ShuffleNetEncoder(116, 128, half_skip_channels=12, quarter_skip_channels=12)
        self.context_aware_attention_module = ContextAwareAttentionModule(128, 128, bin_size=(3, 4))
        self.bottleneck = ConvBatchNormPReLU(128, 64, 3)
        self.drivable_area_decoder = TaskDecoder(64, 2, stage1_channels=32, stage2_channels=8, skip_connection_channels=12)
        self.lane_line_decoder = TaskDecoder(64, 2, stage1_channels=32, stage2_channels=8, skip_connection_channels=12)

    def forward(self, images):
        encoder_features, half_skip, quarter_skip = self.shufflenet_encoder(images)

        caam_features = self.context_aware_attention_module(encoder_features)
        latent_features = self.bottleneck(caam_features)

        drivable_area_predictions = self.drivable_area_decoder(latent_features, half_skip, quarter_skip)
        lane_line_predictions = self.lane_line_decoder(latent_features, half_skip, quarter_skip)

        return drivable_area_predictions, lane_line_predictions
