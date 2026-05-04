import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
from core.config import AnviaNetConfig


def patch_split(feature_map, bin_size):
    batch_size, channels, height, width = feature_map.size()
    bins_height, bins_width = bin_size

    patch_height = height // bins_height
    patch_width = width // bins_width

    patched_tensor = feature_map.view(batch_size, channels, bins_height, patch_height, bins_width, patch_width)
    patched_tensor = patched_tensor.permute(0, 2, 4, 3, 5, 1).contiguous()
    patched_tensor = patched_tensor.view(batch_size, -1, patch_height, patch_width, channels)

    return patched_tensor


def patch_recover(patched_tensor, bin_size):
    batch_size, _, patch_height, patch_width, channels = patched_tensor.size()
    bins_height, bins_width = bin_size

    height = patch_height * bins_height
    width = patch_width * bins_width

    feature_map = patched_tensor.view(batch_size, bins_height, bins_width, patch_height, patch_width, channels)
    feature_map = feature_map.permute(0, 5, 1, 3, 2, 4).contiguous()
    feature_map = feature_map.view(batch_size, channels, height, width)

    return feature_map


class GraphConvolutionNetwork(nn.Module):
    def __init__(self, num_nodes, num_channels):
        super().__init__()
        self.node_interaction = nn.Conv2d(num_nodes, num_nodes, kernel_size=1, bias=False)
        self.activation = nn.PReLU(num_nodes)
        self.channel_interaction = nn.Linear(num_channels, num_channels, bias=False)

    def forward(self, features):
        out = self.node_interaction(features)
        out = self.activation(out + features)
        out = self.channel_interaction(out)

        return out


class ConvBatchNormPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, dropout_rate=0.0):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False, groups=groups)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.PReLU(out_channels)
        self.dropout_layer = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None

    def forward(self, feature_map):
        out = self.convolution(feature_map)
        out = self.batch_norm(out)
        out = self.activation(out)

        if self.dropout_layer:
            out = self.dropout_layer(out)

        return out


class UpSimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.transposed_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-03)
        self.activation = nn.PReLU(out_channels)

    def forward(self, feature_map):
        out = self.transposed_conv(feature_map)
        out = self.batch_norm(out)
        out = self.activation(out)

        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_map):
        avg_out = torch.mean(feature_map, dim=1, keepdim=True)
        max_out, _ = torch.max(feature_map, dim=1, keepdim=True)

        spatial_descriptors = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.conv(spatial_descriptors)

        out = feature_map * self.sigmoid(attention_map)

        return out


class ShuffleNetEncoder(nn.Module):
    def __init__(self, config: AnviaNetConfig):
        super().__init__()
        backbone = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.DEFAULT)
        self.conv1 = nn.Sequential(nn.Conv2d(5, 24, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(24), nn.ReLU(inplace=True))

        with torch.no_grad():
            self.conv1[0].weight[:, :3, :, :] = backbone.conv1[0].weight
            self.conv1[0].weight[:, 3:, :, :].fill_(0.0)
            self.conv1[1].load_state_dict(backbone.conv1[1].state_dict())

        self.maxpool = backbone.maxpool
        self.stage2 = backbone.stage2
        self.bottleneck_conv = ConvBatchNormPReLU(in_channels=config.encoder_in_channels, out_channels=config.encoder_out_channels, kernel_size=1)
        self.compress_skip1 = nn.Sequential(nn.Conv2d(24, 12, kernel_size=1, bias=False), nn.BatchNorm2d(12), nn.PReLU(12))
        self.compress_skip2 = nn.Sequential(nn.Conv2d(24, 12, kernel_size=1, bias=False), nn.BatchNorm2d(12), nn.PReLU(12))

    def _add_coords(self, x):
        batch_size, _, h, w = x.size()

        xx_range = torch.arange(h, dtype=torch.float32, device=x.device).view([1, 1, h, 1])
        xx_channel = xx_range.expand(batch_size, 1, h, w)
        xx_channel = xx_channel / (h - 1) * 2 - 1

        yy_range = torch.arange(w, dtype=torch.float32, device=x.device).view([1, 1, 1, w])
        yy_channel = yy_range.expand(batch_size, 1, h, w)
        yy_channel = yy_channel / (w - 1) * 2 - 1

        xx_channel = xx_channel.to(x.dtype)
        yy_channel = yy_channel.to(x.dtype)

        return torch.cat([x, xx_channel, yy_channel], dim=1)

    def forward(self, image):
        image_coords = self._add_coords(image)
        feat_half = self.conv1(image_coords)
        feat_quarter = self.maxpool(feat_half)
        stage2_features = self.stage2(feat_quarter)

        encoder_features = self.bottleneck_conv(stage2_features)

        skip1 = self.compress_skip1(feat_half)
        skip2 = self.compress_skip2(feat_quarter)

        return encoder_features, skip1, skip2


class ContextAwareAttentionModule(nn.Module):
    def __init__(self, in_channels, num_classes, bin_size, norm_layer=nn.BatchNorm2d):
        super().__init__()
        inner_channels = in_channels // 2
        self.bin_size = bin_size
        self.extract_cam = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.pool_cam = nn.AdaptiveAvgPool2d(bin_size)
        self.sigmoid = nn.Sigmoid()

        bins_height, bins_width = bin_size
        total_bins = bins_height * bins_width
        self.gcn = GraphConvolutionNetwork(num_nodes=total_bins, num_channels=in_channels)
        self.fuse_local_to_global = nn.Conv2d(total_bins, 1, kernel_size=1)

        self.proj_query = nn.Linear(in_channels, inner_channels)
        self.proj_key = nn.Linear(in_channels, inner_channels)
        self.proj_value = nn.Linear(in_channels, inner_channels)

        self.output_projection = nn.Sequential(
            nn.Conv2d(inner_channels, in_channels, kernel_size=1, bias=False),
            norm_layer(in_channels),
            nn.PReLU(in_channels),
        )
        self.activation = nn.PReLU(1)

    def forward(self, features):
        residual = features

        activation_map = self.extract_cam(features)
        cls_score = self.sigmoid(self.pool_cam(activation_map))

        patched_cam = patch_split(activation_map, self.bin_size)
        patched_features = patch_split(features, self.bin_size)

        batch_size = patched_cam.shape[0]
        patch_height = patched_cam.shape[2]
        patch_width = patched_cam.shape[3]
        num_classes = patched_cam.shape[-1]
        feature_channels = patched_features.shape[-1]

        patched_cam = patched_cam.view(batch_size, -1, patch_height * patch_width, num_classes)
        patched_features = patched_features.view(batch_size, -1, patch_height * patch_width, feature_channels)

        bin_confidence = cls_score.view(batch_size, num_classes, -1).transpose(1, 2)

        if bin_confidence.dim() == 3:
            bin_confidence = bin_confidence.unsqueeze(3)

        pixel_confidence = F.softmax(patched_cam, dim=2)

        local_features = torch.matmul(pixel_confidence.transpose(2, 3), patched_features) * bin_confidence
        local_features = self.gcn(local_features)

        global_features = self.fuse_local_to_global(local_features)
        global_features = self.activation(global_features).repeat(1, patched_features.shape[1], 1, 1)

        query = self.proj_query(patched_features)
        key = self.proj_key(local_features)
        value = self.proj_value(global_features)

        affinity_matrix = torch.matmul(query, key.transpose(2, 3))
        affinity_matrix = F.softmax(affinity_matrix, dim=-1)

        out = torch.matmul(affinity_matrix, value)
        out = out.view(batch_size, -1, patch_height, patch_width, value.shape[-1])
        out = patch_recover(out, self.bin_size)

        out_conv = self.output_projection(out)
        out = residual + out_conv

        return out


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_connection_channels=3, is_last_layer=False, kernel_size=3):
        super().__init__()
        self.is_last_layer = is_last_layer
        self.upsample_layer = UpSimpleBlock(in_channels=in_channels, out_channels=out_channels)

        if not is_last_layer:
            self.fusion_layer = ConvBatchNormPReLU(
                in_channels=out_channels + skip_connection_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
            )

        self.output_layer = ConvBatchNormPReLU(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )

    def forward(self, feature_map, skip_features=None):
        upsampled_features = self.upsample_layer(feature_map)

        if not self.is_last_layer and skip_features is not None:
            upsampled_features = torch.cat([upsampled_features, skip_features], dim=1)
            upsampled_features = self.fusion_layer(upsampled_features)

        out = self.output_layer(upsampled_features)

        return out


class TaskDecoder(nn.Module):
    def __init__(self, in_channels, skip_channels=12, use_attention=False):
        super().__init__()
        self.stage1 = UpConvBlock(in_channels=in_channels, out_channels=32, skip_connection_channels=skip_channels)
        self.stage2 = UpConvBlock(in_channels=32, out_channels=8, skip_connection_channels=skip_channels)
        self.attention = SpatialAttention() if use_attention else nn.Identity()
        self.output_head = UpConvBlock(in_channels=8, out_channels=2, is_last_layer=True)

    def forward(self, latent_features, skip_half, skip_quarter):
        out = self.stage1(latent_features, skip_quarter)
        out = self.stage2(out, skip_half)
        out = self.attention(out)
        out = self.output_head(out)

        return out


class AnviaNet(nn.Module):
    def __init__(self, config: AnviaNetConfig = None):
        super().__init__()

        if config is None:
            config = AnviaNetConfig()

        self.encoder = ShuffleNetEncoder(config=config)

        self.caam = ContextAwareAttentionModule(
            in_channels=config.caam_in_channels,
            num_classes=config.caam_num_classes,
            bin_size=config.caam_bin_size,
            norm_layer=nn.BatchNorm2d,
        )

        self.bottleneck = ConvBatchNormPReLU(in_channels=config.bottleneck_in_channels, out_channels=config.bottleneck_out_channels)

        self.decoder_da = TaskDecoder(in_channels=config.decoder_in_channels, skip_channels=config.decoder_skip_channels, use_attention=False)
        self.decoder_ll = TaskDecoder(in_channels=config.decoder_in_channels, skip_channels=config.decoder_skip_channels, use_attention=True)

    def forward(self, image):
        encoder_features, skip_half, skip_quarter = self.encoder(image)

        caam_features = self.caam(encoder_features)
        latent_features = self.bottleneck(caam_features)

        out_da = self.decoder_da(latent_features, skip_half, skip_quarter)
        out_ll = self.decoder_ll(latent_features, skip_half, skip_quarter)

        return out_da, out_ll
