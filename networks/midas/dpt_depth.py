import torch
import torch.nn as nn

from .base_model import BaseModel
from .blocks import (
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_beit,
    forward_swin,
    forward_levit,
    forward_vit,
)
from .backbones.levit import stem_b4_transpose
from timm.models.layers import get_act_layer


def _make_fusion_block(features, use_bn, size = None):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )



class FeatLinearRefine(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.refine4 = FeatLinearRefineBlock(features)
        self.refine3 = FeatLinearRefineBlock(features)
        self.refine2 = FeatLinearRefineBlock(features)
        self.refine1 = FeatLinearRefineBlock(features)
        self.networks = [self.refine1, self.refine2,
                         self.refine3, self.refine4]

    def forward(self, feat_list):
        output_scale = []
        output_bias = []
        for x in range(len(feat_list)):
            scale, bias = self.networks[x](feat_list[x])
            output_scale.append(scale)
            output_bias.append(bias)
        return output_scale, output_bias

    def update_feat(self, feat_list, scales, biases):
        output_feat_list = []
        for x in range(len(feat_list)):
            output_feat_list.append(
                feat_list[x] * scales[x] + biases[x]
            )
        return output_feat_list


class FeatLinearRefineBlock(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.res_conv_1 = ResidualConvUnit(in_dim)
        self.mid_conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.output_weight = nn.Conv2d(
            in_dim//2, 1, kernel_size=3, stride=1, padding=1)
        self.output_bias = nn.Conv2d(
            in_dim//2, 1, kernel_size=3, stride=1, padding=1)
        self.output_weight.weight.data.normal_(0, 1e-4)
        self.output_weight.bias.data.fill_(1)
        self.output_bias.weight.data.normal_(0, 1e-4)
        self.output_bias.bias.data.fill_(0)

    def forward(self, x):
        x = self.res_conv_1(x)
        x = self.mid_conv(x)
        scale = self.output_weight(x)
        bias = self.output_bias(x)
        return scale, bias


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """
        y = F.relu(x, inplace=False)
        out = self.conv1(y)
        out = self.relu(out)
        out = self.conv2(out)

        return out + y



class DPT(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        readout="project",
        channels_last=False,
        use_bn=False,
        resize=None,
        **kwargs
    ):
        print("INITIALIZING DPT")
        backbone = 'beitl16_512'
        self.feature_dim = features
        self.resize = resize

        super(DPT, self).__init__()

        self.channels_last = channels_last

        # For the Swin, Swin 2, LeViT and Next-ViT Transformers, the hierarchical architectures prevent setting the 
        # hooks freely. Instead, the hooks have to be chosen according to the ranges specified in the comments.
        hooks = {
            "beitl16_512": [5, 11, 17, 23],
            "beitl16_384": [5, 11, 17, 23],
            "beitb16_384": [2, 5, 8, 11],
            "swin2l24_384": [1, 1, 17, 1],  # Allowed ranges: [0, 1], [0,  1], [ 0, 17], [ 0,  1]
            "swin2b24_384": [1, 1, 17, 1],                  # [0, 1], [0,  1], [ 0, 17], [ 0,  1]
            "swin2t16_256": [1, 1, 5, 1],                   # [0, 1], [0,  1], [ 0,  5], [ 0,  1]
            "swinl12_384": [1, 1, 17, 1],                   # [0, 1], [0,  1], [ 0, 17], [ 0,  1]
            "next_vit_large_6m": [2, 6, 36, 39],            # [0, 2], [3,  6], [ 7, 36], [37, 39]
            "levit_384": [3, 11, 21],                       # [0, 3], [6, 11], [14, 21]
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }[backbone]

        if "next_vit" in backbone:
            in_features = {
                "next_vit_large_6m": [96, 256, 512, 1024],
            }[backbone]
        else:
            in_features = None

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            False, # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks,
            use_readout=readout,
            in_features=in_features,
        )

        self.number_layers = len(hooks) if hooks is not None else 4
        size_refinenet3 = None
        self.scratch.stem_transpose = None

        # if "beit" in backbone:
        self.forward_transformer = forward_beit
        # elif "swin" in backbone:
        #     self.forward_transformer = forward_swin
        # elif "next_vit" in backbone:
        #     from .backbones.next_vit import forward_next_vit
        #     self.forward_transformer = forward_next_vit
        # elif "levit" in backbone:
        #     self.forward_transformer = forward_levit
        #     size_refinenet3 = 7
        #     self.scratch.stem_transpose = stem_b4_transpose(256, 128, get_act_layer("hard_swish"))
        # else:
        #     self.forward_transformer = forward_vit

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn, size_refinenet3)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = head


    def forward(self, x, inverse_depth=False, return_feat=False, freeze_backbone=False):

        # print(x.shape)

        orig_shape = x.shape[-2:]
        if self.resize is not None:
            x = torch.nn.functional.interpolate(
                x, size=self.resize, mode='bilinear', align_corners=False)

        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layers = self.forward_transformer(self.pretrained, x)
        if self.number_layers == 3:
            layer_1, layer_2, layer_3 = layers
        else:
            layer_1, layer_2, layer_3, layer_4 = layers

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        # if self.number_layers >= 4:
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # if self.number_layers == 3:
        #     path_3 = self.scratch.refinenet3(layer_3_rn, size=layer_2_rn.shape[2:])
        # else:
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        if self.scratch.stem_transpose is not None:
            path_1 = self.scratch.stem_transpose(path_1)

        out = self.scratch.output_conv(path_1)
        if not inverse_depth:
            out = torch.clamp(out, min=1e-2)
            out = 10000 / (out)
        if self.resize is not None:
            out = torch.nn.functional.interpolate(
                out, size=orig_shape, mode='bilinear', align_corners=False)
        if return_feat:
            return [layer_1, layer_2, layer_3, layer_4, layer_1_rn, layer_2_rn, layer_3_rn, layer_4_rn, path_1, path_2, path_3, path_4, out]

        return out
    

    def add_refine_branch(self, output_dim=32):
        self.refine_branch = nn.Module()

    def add_uncertainty_branch(self, output_channel=1):
        self.sigma_output = nn.Sequential(
            # nn.InstanceNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.feature_dim, 128,
                      kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 128,
                      kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(True),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(32, output_channel, kernel_size=1, stride=1, padding=0),
            nn.ELU(True)
        )

        # self.sigma_output = DummyBranch()
        # self.sigma_output.refinenet4 = FeatureFusionBlock(self.feature_dim)
        # self.sigma_output.refinenet3 = FeatureFusionBlock(self.feature_dim)
        # self.sigma_output.refinenet2 = FeatureFusionBlock(self.feature_dim)
        # self.sigma_output.refinenet1 = FeatureFusionBlock(self.feature_dim)
        # self.sigma_output.output_conv = nn.Sequential(
        #     nn.Conv2d(self.feature_dim, 1, kernel_size=3, stride=1, padding=1),
        #     nn.ELU(True),
        #     Interpolate(scale_factor=2, mode="bilinear"))

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                torch.nn.init.normal_(
                    m.weight.data, mean=0, std=1e-3)
                torch.nn.init.constant_(m.bias.data, 0.0)
        self.sigma_output.apply(init_func)
        # self.sigma_output.output_conv.apply(init_func)


    def forward_backbone(self, x):
        print(x.shape)
        with torch.no_grad():
            if self.channels_last == True:
                x.contiguous(memory_format=torch.channels_last)

            layers = self.forward_transformer(self.pretrained, x)
            # if self.number_layers == 3:
            #     layer_1, layer_2, layer_3 = layers
            # else:
            layer_1, layer_2, layer_3, layer_4 = layers

            layer_1_rn = self.scratch.layer1_rn(layer_1)
            layer_2_rn = self.scratch.layer2_rn(layer_2)
            layer_3_rn = self.scratch.layer3_rn(layer_3)
            # if self.number_layers >= 4:
            layer_4_rn = self.scratch.layer4_rn(layer_4)

        return [layer_4_rn, layer_3_rn, layer_2_rn, layer_1_rn]


    def forward_refine(self, feat_list):
        path_4 = self.scratch.refinenet4(feat_list[0], size=feat_list[1].shape[2:])
        path_3 = self.scratch.refinenet3(path_4, feat_list[1], size=feat_list[2].shape[2:])
        path_2 = self.scratch.refinenet2(path_3, feat_list[2], size=feat_list[3].shape[2:])
        path_1 = self.scratch.refinenet1(path_2, feat_list[3])

        if self.scratch.stem_transpose is not None:
            path_1 = self.scratch.stem_transpose(path_1)

        out = self.scratch.output_conv(path_1)
        if self.resize is not None:
            assert False
            out = torch.nn.functional.interpolate(
                out, size=self.orig_shape, mode='bilinear', align_corners=False)
        return out


    def get_uncertainty_feature(self, x):
        if hasattr(self.sigma_output, "refinenet4"):
            return self.forward_backbone(x)
        else:
            feat_list = self.forward_backbone(x)
            # path_4 = self.scratch.refinenet4(feat_list[0])
            # path_3 = self.scratch.refinenet3(path_4, feat_list[1])
            # path_2 = self.scratch.refinenet2(path_3, feat_list[2])
            # path_1 = self.scratch.refinenet1(path_2, feat_list[3])
            return feat_list[1]


    def predict_uncertainty(self, uncertainty_feat):
        if hasattr(self.sigma_output, 'refinenet4'):
            sigma_4 = self.sigma_output.refinenet4(uncertainty_feat[0])
            sigma_3 = self.sigma_output.refinenet3(
                sigma_4, uncertainty_feat[1])
            sigma_2 = self.sigma_output.refinenet2(
                sigma_3, uncertainty_feat[2])
            sigma_1 = self.sigma_output.refinenet1(
                sigma_2, uncertainty_feat[3])
            uncertainty = self.sigma_output.output_conv(sigma_1)+1
        else:
            uncertainty = self.sigma_output(uncertainty_feat)+1
        if self.resize is not None:
            uncertainty = torch.nn.functional.interpolate(
                uncertainty, size=self.orig_shape, mode='bilinear', align_corners=False)
        return uncertainty


    def forward_refine_with_uncertainty(self, feat_list, no_depth_grad=False):
        if no_depth_grad:
            with torch.no_grad():
                disp = self.forward_refine(feat_list)
        else:
            disp = self.forward_refine(feat_list)
        if hasattr(self.sigma_output, 'refinenet4'):
            sigma_4 = self.sigma_output.refinenet4(feat_list[1].detach())
            sigma_3 = self.sigma_output.refinenet3(
                sigma_4, feat_list[1].detach())
            sigma_2 = self.sigma_output.refinenet2(
                sigma_3, feat_list[2].detach())
            sigma_1 = self.sigma_output.refinenet1(
                sigma_2, feat_list[3].detach())
            uncertainty = self.sigma_output.output_conv(sigma_1)+1
        else:
            uncertainty = self.sigma_output(feat_list[1])+1
        if self.resize is not None:
            uncertainty = torch.nn.functional.interpolate(
                uncertainty, size=self.orig_shape, mode='bilinear', align_corners=False)
        return disp, uncertainty




class DPTDepthModel(DPT):
    def __init__(self, path=None, non_negative=True, **kwargs):
        features = kwargs["features"] if "features" in kwargs else 256
        head_features_1 = kwargs["head_features_1"] if "head_features_1" in kwargs else features
        head_features_2 = kwargs["head_features_2"] if "head_features_2" in kwargs else 32
        kwargs.pop("head_features_1", None)
        kwargs.pop("head_features_2", None)

        head = nn.Sequential(
            nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

        super().__init__(head, **kwargs)

        if path is not None:
           self.load(path)

    # def forward(self, x):
    #     assert False, "Shouldnt use this"
    #     return super().forward(x).squeeze(dim=1)
