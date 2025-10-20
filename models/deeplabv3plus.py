import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels=256, dilations=[6, 12, 18]):
        super().__init__()
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.aspp_branches = nn.ModuleList()
        for dilation in dilations:
            self.aspp_branches.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilations) + 2), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        size = x.shape[-2:]

        conv_1x1 = self.conv_1x1(x)
        aspp_outputs = [conv_1x1]

        for branch in self.aspp_branches:
            aspp_outputs.append(branch(x))

        image_pool = self.image_pool(x)
        image_pool = F.interpolate(image_pool, size=size, mode='bilinear', align_corners=False)
        aspp_outputs.append(image_pool)

        concatenated = torch.cat(aspp_outputs, dim=1)
        output = self.project(concatenated)

        return output


class DeepLabV3PlusDecoder(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.low_level_projection = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, high_level_features, low_level_features):
        size = low_level_features.shape[-2:]

        low_level_features = self.low_level_projection(low_level_features)

        high_level_features = F.interpolate(high_level_features, size=size, mode='bilinear', align_corners=False)

        concatenated = torch.cat([high_level_features, low_level_features], dim=1)
        output = self.decoder(concatenated)

        return output


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=4, output_stride=16):
        super().__init__()
        self.num_classes = num_classes
        self.output_stride = output_stride

        backbone = models.resnet101(pretrained=True)

        self.backbone_layer1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1
        )

        self.backbone_layer2 = backbone.layer2

        if output_stride == 16:
            self.backbone_layer3 = backbone.layer3
            self.backbone_layer4 = self._make_layer_with_dilation(backbone.layer4, 2)
        else:
            self.backbone_layer3 = self._make_layer_with_dilation(backbone.layer3, 2)
            self.backbone_layer4 = self._make_layer_with_dilation(backbone.layer4, 4)

        for param in self.backbone_layer1.parameters():
            param.requires_grad = False
        for param in self.backbone_layer2.parameters():
            param.requires_grad = False
        for param in self.backbone_layer3.parameters():
            param.requires_grad = False

        self.aspp = ASPPModule(in_channels=2048, out_channels=256, dilations=[6, 12, 18])
        self.decoder = DeepLabV3PlusDecoder(num_classes=num_classes)

    def _make_layer_with_dilation(self, layer, dilation):
        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size[0] == 3:
                    m.dilation = (dilation, dilation)
                    m.padding = (dilation, dilation)
        return layer

    def forward(self, x):
        input_shape = x.shape[-2:]

        layer1_output = self.backbone_layer1(x)
        layer2_output = self.backbone_layer2(layer1_output)
        layer3_output = self.backbone_layer3(layer2_output)
        layer4_output = self.backbone_layer4(layer3_output)

        aspp_output = self.aspp(layer4_output)

        decoder_output = self.decoder(aspp_output, layer1_output)

        output = F.interpolate(decoder_output, size=input_shape, mode='bilinear', align_corners=False)

        return output

    def freeze_backbone(self):
        for param in self.backbone_layer1.parameters():
            param.requires_grad = False
        for param in self.backbone_layer2.parameters():
            param.requires_grad = False
        for param in self.backbone_layer3.parameters():
            param.requires_grad = False

    def unfreeze_decoder(self):
        for param in self.aspp.parameters():
            param.requires_grad = True
        for param in self.decoder.parameters():
            param.requires_grad = True

