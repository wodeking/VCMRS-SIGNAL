# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import torch.nn as nn
def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]

# def downsample_basic_block(x, planes, stride):
#   out = F.avg_pool3d(x, kernel_size=1, stride=stride)
#   zero_pads = torch.Tensor(
#       out.size(0), planes - out.size(1), out.size(2), out.size(3),
#       out.size(4)).zero_()
#   if isinstance(out.data, torch.cuda.FloatTensor):
#       zero_pads = zero_pads.cuda()

#   out = Variable(torch.cat([out.data, zero_pads], dim=1))

#   return out
class ResBlock3D(nn.Module):
    def __init__(self, in_planes, out_planes, stride, activation):
        super().__init__()
        if not isinstance(stride, tuple) and not isinstance(stride, list):
            stride = (stride, stride, stride)
        assert all([x == 1 or x == 2 for x in stride]), f"Expected stride of 1s and 2s. Got stride {stride}."

        self.in_planes, self.out_planes, self.activation, self.stride = in_planes, out_planes, activation, stride

        # Placeholder for subclasses
        self.conv_block = nn.ModuleList()
        self.block_shortcut = nn.Identity()
    
    def forward(self, x):
        residual = x
        residual = self.block_shortcut(x) # Do nothing by default (identity)
        for m in self.conv_block:
            x = m(x)
        if self.activation:
            x = self.activation(x)
        x += residual

        return x

    # def extra_repr(self):
    #     return "ASDASD"

'''    
##### Downsampling blocks (Encoder) ###############
'''

class BasicDownBlock3D(ResBlock3D):
    def __init__(self, in_planes, out_planes, stride, activation, kernel_size):
        super().__init__(in_planes, out_planes, stride, activation)
       
        # Main block structure
        padding = int((kernel_size - 1)/2)
        self.conv_block.append(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=self.stride, padding=padding))
        self.block_shortcut = nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride)

class SeparablekDownBlock3D(ResBlock3D):
    def __init__(self, in_planes, out_planes, stride, activation, kernel_size):
        super().__init__(in_planes, out_planes, stride, activation)
       
        # Main block structure
        padding = int((kernel_size - 1)/2)
        depthwise = nn.Conv3d(in_planes, in_planes, kernel_size=kernel_size, padding=padding, groups=in_planes, stride=self.stride)
        pointwise = nn.Conv3d(in_planes, out_planes, kernel_size=1, groups=1)

        self.conv_block.append(depthwise)
        self.conv_block.append(pointwise)
        self.block_shortcut = nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride)

'''    
##### Upsampling blocks (Decoder) ###############
'''

class BasicUpBlock3D(ResBlock3D):
    def __init__(self, in_planes, out_planes, stride, activation, kernel_size):
        super().__init__(in_planes, out_planes, stride, activation)
        output_padding = tuple(1 if s == 2 else 0 for s in self.stride)
        padding = int((kernel_size - 1)/2)

        # Main block structure
        self.conv_block.append(nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=self.stride, padding=padding, output_padding=output_padding))
        self.block_shortcut = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=1, stride=stride, output_padding=output_padding)

class SeparablekUpBlock3D(ResBlock3D):
    def __init__(self, in_planes, out_planes, stride, activation, kernel_size):
        super().__init__(in_planes, out_planes, stride, activation)
        output_padding = tuple(1 if s == 2 else 0 for s in self.stride)
        # Main block structure
        padding = int((kernel_size - 1)/2)
        depthwise = nn.ConvTranspose3d(in_planes, in_planes, kernel_size=kernel_size, padding=padding, groups=in_planes, stride=self.stride, output_padding=output_padding)
        pointwise = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=1, groups=1)

        self.conv_block.append(depthwise)
        self.conv_block.append(pointwise)
        self.block_shortcut = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=1, stride=stride, output_padding=output_padding)

