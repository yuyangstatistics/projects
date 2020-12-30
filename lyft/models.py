import torch
from torch import nn
from torchvision.models.resnet import resnet50, resnet18, resnet34, resnet101
from efficientnet_pytorch import EfficientNet
from typing import Dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LyftEffnet(nn.Module):

    def __init__(self, cfg: Dict, num_modes=3):
        super().__init__()

        # architecture = cfg["model_params"]["model_architecture"]
        # backbone = eval(architecture)(pretrained=True, progress=True)
        backbone = EfficientNet.from_pretrained("efficientnet-b4")
        self.backbone = backbone

        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels

        # X, Y coords for the future positions (output shape: batch_sizex50x2)
        self.future_len = cfg["model_params"]["future_num_frames"]
        num_targets = 2 * self.future_len
        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes

        # modify the input channels and output channels of efficientnet
        self.backbone._conv_stem.in_channels = num_in_channels
        conv_weight = self.backbone._conv_stem.weight
        self.backbone._conv_stem.weight = nn.Parameter(conv_weight.repeat(1, 9, 1, 1)[:, 0:25, :, :])
        # rewrite the fc layer, don't use the backbone _fc, it doesn't work
        in_features = self.backbone._fc.in_features  
        self._fc = nn.Linear(in_features = in_features, \
            out_features = self.num_preds + num_modes, bias = True)

    def forward(self, x):
        # convolution layers
        x = self.backbone.extract_features(x)
        
        # pooling and final linear layer
        x = self.backbone._avg_pooling(x)
        x = x.flatten(start_dim = 1)
        x = self.backbone._dropout(x)
        x = self._fc(x)

        # pred (batch_size)x(modes)x(time)x(2D coords)
        # confidences (batch_size)x(modes)
        bs, _ = x.shape
        pred, confidences = torch.split(x, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences


class LyftDensenet(nn.Module):

    def __init__(self, cfg: Dict, num_modes=3):
        super().__init__()

        # architecture = cfg["model_params"]["model_architecture"]
        # backbone = eval(architecture)(pretrained=True, progress=True)
        backbone = torch.hub.load("pytorch/vision:v0.6.0", "densenet161", 
            pretrained = True)
        self.backbone = backbone

        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels

        # X, Y coords for the future positions (output shape: batch_sizex50x2)
        self.future_len = cfg["model_params"]["future_num_frames"]
        num_targets = 2 * self.future_len
        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes

        # modify the input channels and output channels of efficientnet
        self.backbone.features.conv0.in_channels = num_in_channels
        conv_weight = self.backbone.features.conv0.weight
        self.backbone.features.conv0.weight = nn.Parameter(conv_weight.repeat(1, 9, 1, 1)[:, 0:25, :, :])
        # rewrite the fc layer, don't use the backbone _fc, it doesn't work
        in_features = self.backbone.classifier.in_features  
        self.backbone.classifier = nn.Linear(in_features = in_features, \
            out_features = self.num_preds + num_modes, bias = True)
        

    def forward(self, x):
        # convolution layers
        x = self.backbone(x)

        # pred (batch_size)x(modes)x(time)x(2D coords)
        # confidences (batch_size)x(modes)
        bs, _ = x.shape
        pred, confidences = torch.split(x, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences


class LyftEffnetb7(nn.Module):

    def __init__(self, cfg: Dict, num_modes=3):
        super().__init__()

        # architecture = cfg["model_params"]["model_architecture"]
        # backbone = eval(architecture)(pretrained=True, progress=True)
        backbone = EfficientNet.from_pretrained("efficientnet-b7")
        self.backbone = backbone

        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels

        # X, Y coords for the future positions (output shape: batch_sizex50x2)
        self.future_len = cfg["model_params"]["future_num_frames"]
        num_targets = 2 * self.future_len
        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes

        # modify the input channels and output channels of efficientnet
        self.backbone._conv_stem.in_channels = num_in_channels
        conv_weight = self.backbone._conv_stem.weight
        self.backbone._conv_stem.weight = nn.Parameter(conv_weight.repeat(1, 9, 1, 1)[:, 0:25, :, :])
        # rewrite the fc layer, don't use the backbone _fc, it doesn't work
        in_features = self.backbone._fc.in_features  
        self._fc = nn.Linear(in_features = in_features, \
            out_features = self.num_preds + num_modes, bias = True)

    def forward(self, x):
        # convolution layers
        x = self.backbone.extract_features(x)
        
        # pooling and final linear layer
        x = self.backbone._avg_pooling(x)
        x = x.flatten(start_dim = 1)
        x = self.backbone._dropout(x)
        x = self._fc(x)

        # pred (batch_size)x(modes)x(time)x(2D coords)
        # confidences (batch_size)x(modes)
        bs, _ = x.shape
        pred, confidences = torch.split(x, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences