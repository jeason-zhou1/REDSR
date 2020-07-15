import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import common as mutil
from model import common
from torch.nn.parallel import DistributedDataParallel
import math

def make_model(args, parent=False):
    return NLSR(args,3,3,64,16)

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x



class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class NONLocalBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian',
                 sub_sample=False, bn_layer=False):
        super(NONLocalBlock2D, self).__init__()
        assert mode in ['embedded_gaussian', 'gaussian', 'dot_product', 'concatenation']

        self.mode = mode

        self.in_channels = in_channels
        self.inter_channels = in_channels // 2
        if self.inter_channels == 0:
            self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool = nn.MaxPool2d
        sub_sample = nn.Upsample
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                            kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

        self.theta = None
        self.phi = None

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                            kernel_size=1, stride=1, padding=0)

        self.operation_function = self._embedded_gaussian

        self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
        self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))

    def forward(self, x):

        output = self.operation_function(x)
        return output
        
    def _embedded_gaussian(self, x):
        batch_size,C,H,W = x.shape

        # g(x)同样把通道数减为了一半
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        # (b,hw,0.5c)
        g_x = g_x.permute(0, 2, 1)

        # 2D卷积 theta，此处的dimension是2，将通道数变成了原来的一半,(b,0.5c,hw)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        # 此处进行维度变换是为了方便进行矩阵相乘,(b,hw,0.5c)
        theta_x = theta_x.permute(0, 2, 1)
        # phi的操作也是将通道数变成原来的一半，phi和theta的操作是一样的,(b,0.5c,hw)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        # print(phi_x.shape,theta_x.shape)
        f = torch.matmul(theta_x, phi_x)
        # return f，
        # f_div_C相当于是一个在space上的一个的权重，（b,hw,hw）
        f_div_C = F.softmax(f, dim=-1)
        # return f_div_C
        # (b, hw,hw)dot(b,hw,0.5c) ==> (b,hw,0.5c)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        # (b,0.5c,h,w)
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        # W，将通道变回原来两倍 (b,0.5c,h,w)==> (b,c,h,w)
        W_y = self.W(y)
        z = W_y + x

        return z

class Nonlocal(nn.Module):
    def __init__(self, in_feat=64, inter_feat=32, reduction=8,sub_sample=False, bn_layer=True):
        super(Nonlocal, self).__init__()
        self.non_local = (NONLocalBlock2D(in_channels=in_feat,inter_channels=inter_feat, sub_sample=sub_sample,bn_layer=bn_layer))

        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        ## divide feature map into 4 part
        batch_size,C,H,W = x.shape
        H1 = int(H / 2)
        W1 = int(W / 2)
        nonlocal_feat = torch.zeros_like(x)

        feat_sub_lu = x[:, :, :H1, :W1]
        feat_sub_ld = x[:, :, H1:, :W1]
        feat_sub_ru = x[:, :, :H1, W1:]
        feat_sub_rd = x[:, :, H1:, W1:]


        nonlocal_lu = self.non_local(feat_sub_lu)
        nonlocal_ld = self.non_local(feat_sub_ld)
        nonlocal_ru = self.non_local(feat_sub_ru)
        nonlocal_rd = self.non_local(feat_sub_rd)
        nonlocal_feat[:, :, :H1, :W1] = nonlocal_lu
        nonlocal_feat[:, :, H1:, :W1] = nonlocal_ld
        nonlocal_feat[:, :, :H1, W1:] = nonlocal_ru
        nonlocal_feat[:, :, H1:, W1:] = nonlocal_rd

        return  nonlocal_feat


class NLSR(nn.Module):
    def __init__(self, args, in_nc, out_nc, nf, nb, gc=32):
        super(NLSR, self).__init__()
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        self.scale = args.scale[0]
        self.scale=4

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        # self.non_local = NONLocalBlock2D(nf)

        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.RRDB_trunk = mutil.make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling

        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if self.scale==4:
            self.upsample2 = nn.PixelShuffle(2)
            self.upconv2 = nn.Conv2d(nf, nf*4, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self,x):
        x = self.sub_mean(x)
        fea = self.conv_first(x)
        # fea = self.non_local(fea)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        if self.scale==2:
            fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        elif self.scale==3:
            fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=3, mode='nearest')))
        elif self.scale==4:
            fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
            fea = self.lrelu(self.upsample2(self.upconv2(fea)))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        out = self.add_mean(out)
        return out



    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
 

