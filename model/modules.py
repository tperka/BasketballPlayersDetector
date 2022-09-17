import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, channel_attention=False):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.channel_attention = False
        if channel_attention:
            self.eca = ChannelAttention(channel=out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        if self.channel_attention:
            input = self.eca(input)
        input = input + shortcut
        return nn.ReLU()(input)


class ChannelAttention(nn.Module):
    # Taken from: https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, b=1, gamma=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channels = channel
        self.b = b
        self.gamma = gamma
        self.kernel_size = self.kernel_size()
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def kernel_size(self):
        k = int(abs((math.log2(self.channels)/self.gamma)+ self.b/self.gamma))
        out = k if k % 2 else k + 1
        return out

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class FPN(nn.Module):
    def __init__(self, layers, out_channels, lateral_channels, return_layers=None):
        # return_layers: index of layers (numbered from 0) for which feature maps are returned
        super(FPN, self).__init__()

        assert len(layers) == len(out_channels)

        self.layers = layers
        self.out_channels = out_channels
        self.lateral_channels = lateral_channels
        self.lateral_layers = nn.ModuleList()
        self.smooth_layers = nn.ModuleList()
        if return_layers is None:
            # Feature maps fom all FPN levels are returned
            self.return_layers = list(range(len(layers)-1))
        else:
            self.return_layers = return_layers
        self.min_returned_layer = min(self.return_layers)

        # Make lateral layers (for channel reduction) and smoothing layers
        for i in range(self.min_returned_layer, len(self.layers)):
            self.lateral_layers.append(nn.Conv2d(out_channels[i], self.lateral_channels, kernel_size=1, stride=1,
                                                 padding=0))
        # Smoothing layers are not used. Because bilinear interpolation is used during the upsampling,
        # the resultant feature maps are free from artifacts

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        # Bottom-up pass, store all intermediary feature maps in list c
        c = []
        for m in self.layers:
            x = m(x)
            c.append(x)

        # Top-down pass
        p = [self.lateral_layers[-1](c[-1])]

        for i in range(len(c)-2, self.min_returned_layer-1, -1):
            temp = self._upsample_add(p[-1],  self.lateral_layers[i-self.min_returned_layer](c[i]))
            p.append(temp)

        # Reverse the order of tensors in p
        p = p[::-1]

        out_tensors = []
        for ndx, l in enumerate(self.return_layers):
            temp = p[l-self.min_returned_layer]
            out_tensors.append(temp)

        return out_tensors if len(out_tensors) > 1 else out_tensors[0]


class ConvModule(nn.Module):
    """module grouping a convolution, batchnorm, and activation function"""

    def __init__(self, n_in, n_out, kernel_size=3,
                 stride=1, padding=0, groups=1, dilation=1, bias=False,
                 bn=True, activation=nn.SiLU, channel_attention=False):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_in, n_out, kernel_size=kernel_size,
                              stride=stride, padding=padding,
                              groups=groups, bias=bias, dilation=dilation)
        self.bn = nn.BatchNorm2d(n_out) if bn else nn.Identity()
        self.act = activation(inplace=True) if activation != nn.LeakyReLU else activation(0.1, inplace=True)

        self.channel_attention = ChannelAttention(n_out, k_size=kernel_size) if channel_attention else None

    def forward(self, x):
        in_shape = x.shape
        x = self.conv(x)
        x = x[:, :, :in_shape[2] // self.stride, :in_shape[3] // self.stride]
        x = self.bn(x)
        x = self.channel_attention(x) if self.channel_attention else x
        x = self.act(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-excitation block"""

    def __init__(self, n_in, r=16):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(nn.Conv2d(n_in, n_in // r, kernel_size=1),
                                        nn.SiLU(),
                                        nn.Conv2d(n_in // r, n_in, kernel_size=1),
                                        nn.Sigmoid())

    def forward(self, x):
        y = self.squeeze(x)
        y = self.excitation(y)
        return x * y


class DropSample(nn.Module):
    """Drops each sample in x with probability p during training"""

    def __init__(self, p=0):
        super().__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.p = p
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, x):
        if (not self.p) or (not self.training):
            return x

        batch_size = len(x)
        random_tensor = torch.FloatTensor(batch_size, 1, 1, 1).uniform_()
        bit_mask = self.p < random_tensor
        bit_mask = bit_mask.to(self.device)
        x = x.div(1 - self.p)
        x = x * bit_mask
        return x


class MBConvN(nn.Module):
    """MBConv with an expansion factor of N, plus squeeze-and-excitation"""

    def __init__(self, n_in, n_out, expansion_factor,
                 kernel_size=3, stride=1, r=24, p=0):
        super().__init__()

        padding = (kernel_size - 1) // 2
        expanded = expansion_factor * n_in
        self.skip_connection = (n_in == n_out) and (stride == 1)

        self.expand_pw = nn.Identity() if (expansion_factor == 1) else ConvModule(n_in, expanded, kernel_size=1)
        self.depthwise = ConvModule(expanded, expanded, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=expanded)
        self.se = SEBlock(expanded, r=r)
        self.reduce_pw = ConvModule(expanded, n_out, kernel_size=1,
                                   activation=nn.Identity)
        self.dropsample = DropSample(p)

    def forward(self, x):
        residual = x

        x = self.expand_pw(x)
        x = self.depthwise(x)
        x = self.se(x)
        x = self.reduce_pw(x)

        if self.skip_connection:
            x = self.dropsample(x)
            x = x + residual
        return x


