from typing import Optional, Sequence, Tuple, Union

import torch
from monai.networks.blocks import UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.networks.layers.utils import get_act_layer
from monai.networks.nets import EfficientNetBNFeatures
from monai.networks.nets.basic_unet import UpCat
from monai.utils import InterpolateMode
from torch import nn


class UNetDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int],
        act: Union[str, tuple],
        norm: Union[str, tuple],
        dropout: Union[float, tuple],
        bias: bool,
        upsample: str,
        pre_conv: Optional[str],
        interp_mode: str,
        align_corners: Optional[bool],
    ):
        """
        UNet Decoder.
        This class refers to `segmentation_models.pytorch
        <https://github.com/qubvel/segmentation_models.pytorch>`_.

        Args:
            dim: number of spatial dimensions.
            encoder_channels: number of output channels for all feature maps in encoder.
                `len(encoder_channels)` should be no less than 2.
            decoder_channels: number of output channels for all feature maps in decoder.
                `len(decoder_channels)` should equal to `len(encoder_channels) - 1`.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            dropout: dropout ratio.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
            pre_conv: a conv block applied before upsampling.
                Only used in the "nontrainable" or "pixelshuffle" mode.
            interp_mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``}
                Only used in the "nontrainable" mode.
            align_corners: set the align_corners parameter for upsample. Defaults to True.
                Only used in the "nontrainable" mode.

        """
        super().__init__()
        if len(encoder_channels) < 2:
            raise ValueError("the length of `encoder_channels` should be no less than 2.")
        if len(decoder_channels) != len(encoder_channels) - 1:
            raise ValueError("`len(decoder_channels)` should equal to `len(encoder_channels) - 1`.")

        in_channels = [encoder_channels[-1]] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:-1][::-1]) + [0]
        halves = [True] * (len(skip_channels) - 1)
        halves.append(False)
        blocks = []
        for in_chn, skip_chn, out_chn, halve in zip(in_channels, skip_channels, decoder_channels, halves):
            blocks.append(
                UpCat(
                    dim=dim,
                    in_chns=in_chn,
                    cat_chns=skip_chn,
                    out_chns=out_chn,
                    act=act,
                    norm=norm,
                    dropout=dropout,
                    bias=bias,
                    upsample=upsample,
                    pre_conv=pre_conv,
                    interp_mode=interp_mode,
                    align_corners=align_corners,
                    halves=halve,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, features: Sequence[torch.Tensor]):
        skips = features[:-1][::-1]
        features = features[1:][::-1]

        x = features[0]
        for i, block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = block(x, skip)

        return x


class SegmentationHead(nn.Sequential):
    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        act: Optional[Union[Tuple, str]] = None,
        scale_factor: float = 1.0,
    ):
        """
        Segmentation head.
        This class refers to `segmentation_models.pytorch
        <https://github.com/qubvel/segmentation_models.pytorch>`_.

        Args:
            dim: number of spatial dimensions.
            in_channels: number of input channels for the block.
            out_channels: number of output channels for the block.
            kernel_size: kernel size for the conv layer.
            act: activation type and arguments.
            scale_factor: multiplier for spatial size. Has to match input size if it is a tuple.

        """
        conv_layer = Conv[Conv.CONV, dim](
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        up_layer: nn.Module = nn.Identity()
        if scale_factor > 1.0:
            up_layer = UpSample(
                spatial_dims=dim,
                scale_factor=scale_factor,
                mode="nontrainable",
                pre_conv=None,
                interp_mode=InterpolateMode.LINEAR,
            )
        if act is not None:
            act_layer = get_act_layer(act)
        else:
            act_layer = nn.Identity()
        super().__init__(conv_layer, up_layer, act_layer)


class RanzcrNetV2(nn.Module):
    def __init__(self, in_channels, out_channels, backbone="efficientnet-b0", pretrained=True, dropout=0.0):
        super().__init__()
        print(f"Using backbone {backbone}.")
        if "efficientnet_b8" in backbone:
            encoder_channels = (in_channels, 32, 56, 88, 248, 704)
            model_name = "efficientnet-b8"
        else:
            encoder_channels = (in_channels, 16, 24, 40, 112, 320)
            model_name = backbone
        if "ap" in backbone:
            adv_prop = True
        else:
            adv_prop = False

        self.backbone = backbone
        self.encoder = EfficientNetBNFeatures(
            model_name=model_name,
            pretrained=pretrained,
            in_channels=in_channels,
            norm=("batch", {"eps": 1e-3, "momentum": 0.1}),
            adv_prop=adv_prop,
        )
        self.decoder = UNetDecoder(
            dim=2,
            encoder_channels=encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            act=("relu", {"inplace": True}),
            norm="batch",
            bias=False,
            upsample="nontrainable",
            pre_conv=None,
            interp_mode="nearest",
            align_corners=None,
            dropout=dropout,
        )

        self.segmentation_head = SegmentationHead(
            dim=2,
            in_channels=16,
            out_channels=out_channels,
            kernel_size=3,
            act=None,
        )

        self.pool = Pool[Pool.ADAPTIVEMAX, 2](1)

    def forward(self, x):

        enc_out = self.encoder(x)
        decoder_out = self.decoder(enc_out)
        x_seg = self.segmentation_head(decoder_out)

        return x_seg
