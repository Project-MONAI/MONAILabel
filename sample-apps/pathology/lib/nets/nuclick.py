import torch
import torch.nn as nn
import torch.nn.functional as F

bn_axis = 1


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# --------------------New layers development in progress-----------------
"""(convolution => [BN] => ReLU/sigmoid)"""
"""No regularizer"""


class Conv_Bn_Relu(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=32,
        kernelSize=(3, 3),
        strds=(1, 1),
        useBias=False,
        dilatationRate=(1, 1),
        actv="relu",
        doBatchNorm=True,
    ):

        super().__init__()
        if isinstance(kernelSize, int):
            kernelSize = (kernelSize, kernelSize)
        if isinstance(strds, int):
            strds = (strds, strds)

        self.conv_bn_relu = self.get_block(
            in_channels, out_channels, kernelSize, strds, useBias, dilatationRate, actv, doBatchNorm
        )

    def forward(self, input):
        return self.conv_bn_relu(input)

    def get_block(self, in_channels, out_channels, kernelSize, strds, useBias, dilatationRate, actv, doBatchNorm):

        layers = []

        conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernelSize,
            stride=strds,
            dilation=dilatationRate,
            bias=useBias,
            padding="same",
            padding_mode="zeros",
        )

        if actv == "selu":
            # (Can't find 'lecun_normal' equivalent in PyTorch)
            torch.nn.init.xavier_normal_(conv1.weight)
        else:
            torch.nn.init.xavier_uniform_(conv1.weight)

        layers.append(conv1)

        if actv != "selu" and doBatchNorm:
            layers.append(nn.BatchNorm2d(num_features=out_channels, eps=1.001e-5))

        if actv == "relu":
            layers.append(nn.ReLU())
        elif actv == "sigmoid":
            layers.append(nn.Sigmoid())
        elif actv == "selu":
            layers.append(nn.SELU())

        block = nn.Sequential(*layers)
        return block


"""Multiscale Conv Block"""


class Multiscale_Conv_Block(nn.Module):
    def __init__(
        self,
        in_channels,
        kernelSizes,
        dilatationRates,
        out_channels=32,
        strds=(1, 1),
        actv="relu",
        useBias=False,
        isDense=True,
    ):

        super().__init__()

        # Initialise conv blocks
        if isDense:
            self.conv_block_0 = Conv_Bn_Relu(
                in_channels=in_channels,
                out_channels=4 * out_channels,
                kernelSize=1,
                strds=strds,
                actv=actv,
                useBias=useBias,
            )
            self.conv_block_5 = Conv_Bn_Relu(
                in_channels=in_channels,
                out_channels=out_channels,
                kernelSize=3,
                strds=strds,
                actv=actv,
                useBias=useBias,
            )
        else:
            self.conv_block_0 = None
            self.conv_block_5 = None

        self.conv_block_1 = Conv_Bn_Relu(
            in_channels=in_channels,
            out_channels=out_channels,
            kernelSize=kernelSizes[0],
            strds=strds,
            actv=actv,
            useBias=useBias,
            dilatationRate=(dilatationRates[0], dilatationRates[0]),
        )

        self.conv_block_2 = Conv_Bn_Relu(
            in_channels=in_channels,
            out_channels=out_channels,
            kernelSize=kernelSizes[1],
            strds=strds,
            actv=actv,
            useBias=useBias,
            dilatationRate=(dilatationRates[1], dilatationRates[1]),
        )

        self.conv_block_3 = Conv_Bn_Relu(
            in_channels=in_channels,
            out_channels=out_channels,
            kernelSize=kernelSizes[2],
            strds=strds,
            actv=actv,
            useBias=useBias,
            dilatationRate=(dilatationRates[2], dilatationRates[2]),
        )

        self.conv_block_4 = Conv_Bn_Relu(
            in_channels=in_channels,
            out_channels=out_channels,
            kernelSize=kernelSizes[3],
            strds=strds,
            actv=actv,
            useBias=useBias,
            dilatationRate=(dilatationRates[3], dilatationRates[3]),
        )

    def forward(self, input_map):
        # If isDense == True
        if self.conv_block_0 is not None:
            conv0 = self.conv_block_0(input_map)
        else:
            conv0 = input_map

        conv1 = self.conv_block_1(conv0)
        conv2 = self.conv_block_2(conv0)
        conv3 = self.conv_block_3(conv0)
        conv4 = self.conv_block_4(conv0)

        # (Not sure about bn_axis)
        output_map = torch.cat([conv1, conv2, conv3, conv4], dim=bn_axis)

        # If isDense == True
        if self.conv_block_5 is not None:
            output_map = self.conv_block_5(output_map)
            # (Not sure about bn_axis)
            output_map = torch.cat([input_map, output_map], dim=bn_axis)

        return output_map


"""Residual_Conv"""


class Residual_Conv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=32,
        kernelSize=(3, 3),
        strds=(1, 1),
        actv="relu",
        useBias=False,
        dilatationRate=(1, 1),
    ):
        super().__init__()

        if actv == "selu":
            self.conv_block_1 = Conv_Bn_Relu(
                in_channels,
                out_channels,
                kernelSize=kernelSize,
                strds=strds,
                actv="None",
                useBias=useBias,
                dilatationRate=dilatationRate,
                doBatchNorm=False,
            )
            self.conv_block_2 = Conv_Bn_Relu(
                in_channels,
                out_channels,
                kernelSize=kernelSize,
                strds=strds,
                actv="None",
                useBias=useBias,
                dilatationRate=dilatationRate,
                doBatchNorm=False,
            )
            self.activation = nn.SELU()
        else:
            self.conv_block_1 = Conv_Bn_Relu(
                in_channels,
                out_channels,
                kernelSize=kernelSize,
                strds=strds,
                actv="None",
                useBias=useBias,
                dilatationRate=dilatationRate,
                doBatchNorm=True,
            )
            self.conv_block_2 = Conv_Bn_Relu(
                out_channels,
                out_channels,
                kernelSize=kernelSize,
                strds=strds,
                actv="None",
                useBias=useBias,
                dilatationRate=dilatationRate,
                doBatchNorm=True,
            )

            if actv == "relu":
                self.activation = nn.ReLU()

            if actv == "sigmoid":
                self.activation = nn.Sigmoid()

    def forward(self, input):
        conv1 = self.conv_block_1(input)
        conv2 = self.conv_block_2(conv1)

        out = torch.add(conv1, conv2)
        out = self.activation(out)
        return out


class NuClick_NN(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.net_name = "NuClick"

        self.n_channels = n_channels
        self.n_classes = n_classes

        # -------------Conv_Bn_Relu blocks------------
        self.conv_block_1 = nn.Sequential(
            Conv_Bn_Relu(in_channels=self.n_channels, out_channels=64, kernelSize=7),
            Conv_Bn_Relu(in_channels=64, out_channels=32, kernelSize=5),
            Conv_Bn_Relu(in_channels=32, out_channels=32, kernelSize=3),
        )

        self.conv_block_2 = nn.Sequential(
            Conv_Bn_Relu(in_channels=64, out_channels=64),
            Conv_Bn_Relu(in_channels=64, out_channels=32),
            Conv_Bn_Relu(in_channels=32, out_channels=32),
        )

        self.conv_block_3 = Conv_Bn_Relu(
            in_channels=32, out_channels=self.n_classes, kernelSize=(1, 1), actv=None, useBias=True, doBatchNorm=False
        )

        # -------------Residual_Conv blocks------------
        self.residual_block_1 = nn.Sequential(
            Residual_Conv(in_channels=32, out_channels=64), Residual_Conv(in_channels=64, out_channels=64)
        )

        self.residual_block_2 = Residual_Conv(in_channels=64, out_channels=128)

        self.residual_block_3 = Residual_Conv(in_channels=128, out_channels=128)

        self.residual_block_4 = nn.Sequential(
            Residual_Conv(in_channels=128, out_channels=256),
            Residual_Conv(in_channels=256, out_channels=256),
            Residual_Conv(in_channels=256, out_channels=256),
        )

        self.residual_block_5 = nn.Sequential(
            Residual_Conv(in_channels=256, out_channels=512),
            Residual_Conv(in_channels=512, out_channels=512),
            Residual_Conv(in_channels=512, out_channels=512),
        )

        self.residual_block_6 = nn.Sequential(
            Residual_Conv(in_channels=512, out_channels=1024), Residual_Conv(in_channels=1024, out_channels=1024)
        )

        self.residual_block_7 = nn.Sequential(
            Residual_Conv(in_channels=1024, out_channels=512), Residual_Conv(in_channels=512, out_channels=256)
        )

        self.residual_block_8 = Residual_Conv(in_channels=512, out_channels=256)

        self.residual_block_9 = Residual_Conv(in_channels=256, out_channels=256)

        self.residual_block_10 = nn.Sequential(
            Residual_Conv(in_channels=256, out_channels=128), Residual_Conv(in_channels=128, out_channels=128)
        )

        self.residual_block_11 = Residual_Conv(in_channels=128, out_channels=64)

        self.residual_block_12 = Residual_Conv(in_channels=64, out_channels=64)

        # -------------Multiscale_Conv_Block blocks------------
        self.multiscale_block_1 = Multiscale_Conv_Block(
            in_channels=128, out_channels=32, kernelSizes=[3, 3, 5, 5], dilatationRates=[1, 3, 3, 6], isDense=False
        )

        self.multiscale_block_2 = Multiscale_Conv_Block(
            in_channels=256, out_channels=64, kernelSizes=[3, 3, 5, 5], dilatationRates=[1, 3, 2, 3], isDense=False
        )

        self.multiscale_block_3 = Multiscale_Conv_Block(
            in_channels=64, out_channels=16, kernelSizes=[3, 3, 5, 7], dilatationRates=[1, 3, 2, 6], isDense=False
        )

        # -------------MaxPool2d blocks------------
        self.pool_block_1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.pool_block_2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.pool_block_3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.pool_block_4 = nn.MaxPool2d(kernel_size=(2, 2))
        self.pool_block_5 = nn.MaxPool2d(kernel_size=(2, 2))

        # -------------ConvTranspose2d blocks------------
        self.conv_transpose_1 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=2,
            stride=(2, 2),
        )

        self.conv_transpose_2 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=256,
            kernel_size=2,
            stride=(2, 2),
        )

        self.conv_transpose_3 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=(2, 2),
        )

        self.conv_transpose_4 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=(2, 2),
        )

        self.conv_transpose_5 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=2,
            stride=(2, 2),
        )

    def forward(self, input):
        conv1 = self.conv_block_1(input)  # conv1: 32 channels
        pool1 = self.pool_block_1(conv1)  # poo1: 32 channels

        conv2 = self.residual_block_1(pool1)  # conv2: 64 channels
        pool2 = self.pool_block_2(conv2)  # pool2: 64 channels

        conv3 = self.residual_block_2(pool2)  # conv3: 128 channels
        conv3 = self.multiscale_block_1(conv3)  # conv3: 128 channels
        conv3 = self.residual_block_3(conv3)  # conv3: 128 channels
        pool3 = self.pool_block_3(conv3)  # pool3: 128 channels

        conv4 = self.residual_block_4(pool3)  # conv4: 256 channels
        pool4 = self.pool_block_4(conv4)  # pool4: 512 channels

        conv5 = self.residual_block_5(pool4)  # conv5: 512 channels
        pool5 = self.pool_block_5(conv5)  # pool5: 512  channels

        conv51 = self.residual_block_6(pool5)  # conv51: 1024 channels

        up61 = torch.cat([self.conv_transpose_1(conv51), conv5], dim=1)  # up61: 1024 channels
        conv61 = self.residual_block_7(up61)  # conv61: 256 channels

        up6 = torch.cat([self.conv_transpose_2(conv61), conv4], dim=1)  # up6: 512 channels
        conv6 = self.residual_block_8(up6)  # conv6: 256 channels
        conv6 = self.multiscale_block_2(conv6)  # conv6: 256 channels
        conv6 = self.residual_block_9(conv6)  # conv6: 256 channels

        up7 = torch.cat([self.conv_transpose_3(conv6), conv3], dim=1)  # up7: 256 channels
        conv7 = self.residual_block_10(up7)  # conv7: 128 channels

        up8 = torch.cat([self.conv_transpose_4(conv7), conv2], dim=1)  # up8: 128 channels
        conv8 = self.residual_block_11(up8)  # conv8: 64 channels
        conv8 = self.multiscale_block_3(conv8)  # conv8: 64 channels
        conv8 = self.residual_block_12(conv8)  # conv8: 64 channels

        up9 = torch.cat([self.conv_transpose_5(conv8), conv1], dim=1)  # up9: 64 channels
        conv9 = self.conv_block_2(up9)  # conv9: 32 channels

        conv10 = self.conv_block_3(conv9)  # conv10: out_channels

        return conv10
