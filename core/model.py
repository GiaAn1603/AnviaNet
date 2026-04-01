import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights


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


class AvgDownsampler(nn.Module):
    def __init__(self, num_downsamples):
        super().__init__()
        self.pool_layers = nn.ModuleList()

        for _ in range(num_downsamples):
            self.pool_layers.append(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))

    def forward(self, image):
        for pool in self.pool_layers:
            image = pool(image)

        return image


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


class ShuffleNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.DEFAULT)
        self.conv1 = backbone.conv1
        self.maxpool = backbone.maxpool
        self.stage2 = backbone.stage2
        self.bottleneck_conv = ConvBatchNormPReLU(in_channels=116, out_channels=128, kernel_size=1)
        self.downsample_1x = AvgDownsampler(num_downsamples=1)
        self.downsample_2x = AvgDownsampler(num_downsamples=2)

    def forward(self, image):
        downsampled_1x = self.downsample_1x(image)
        downsampled_2x = self.downsample_2x(image)

        features = self.conv1(image)
        features = self.maxpool(features)
        stage2_features = self.stage2(features)

        encoder_features = self.bottleneck_conv(stage2_features)

        return encoder_features, downsampled_1x, downsampled_2x


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


class AnviaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ShuffleNetEncoder()
        self.caam = ContextAwareAttentionModule(in_channels=128, num_classes=128, bin_size=(3, 4), norm_layer=nn.BatchNorm2d)
        self.conv_caam = ConvBatchNormPReLU(in_channels=128, out_channels=64)
        self.decoder_da_stage1 = UpConvBlock(in_channels=64, out_channels=32)
        self.decoder_da_stage2 = UpConvBlock(in_channels=32, out_channels=8)
        self.decoder_da_output = UpConvBlock(in_channels=8, out_channels=2, is_last_layer=True)
        self.decoder_ll_stage1 = UpConvBlock(in_channels=64, out_channels=32)
        self.decoder_ll_stage2 = UpConvBlock(in_channels=32, out_channels=8)
        self.decoder_ll_output = UpConvBlock(in_channels=8, out_channels=2, is_last_layer=True)

    def forward(self, image):
        encoder_features, downsampled_1x, downsampled_2x = self.encoder(image)

        caam_features = self.caam(encoder_features)
        caam_features = self.conv_caam(caam_features)

        out_da = self.decoder_da_stage1(feature_map=caam_features, skip_features=downsampled_2x)
        out_da = self.decoder_da_stage2(feature_map=out_da, skip_features=downsampled_1x)
        out_da = self.decoder_da_output(feature_map=out_da)

        out_ll = self.decoder_ll_stage1(feature_map=caam_features, skip_features=downsampled_2x)
        out_ll = self.decoder_ll_stage2(feature_map=out_ll, skip_features=downsampled_1x)
        out_ll = self.decoder_ll_output(feature_map=out_ll)

        return out_da, out_ll
