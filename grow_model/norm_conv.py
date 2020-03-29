import torch
import torch.nn.functional as F

class FixWeightConv2d(torch.nn.modules.conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = torch.nn.modules.utils._pair(kernel_size)
        stride = torch.nn.modules.utils._pair(stride)
        padding = torch.nn.modules.utils._pair(padding)
        dilation = torch.nn.modules.utils._pair(dilation)
        super(FixWeightConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, torch.nn.modules.utils._pair(0), groups, bias, padding_mode)
        self.weight.requires_grad=False

    def conv2d_forward(self, input, weight):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, self.bias, self.stride,
                            torch.nn.modules.utils._pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self.conv2d_forward(input, self.weight)

class NormConv2dV1(torch.nn.Module):
    '''decouple the direction and length of weight and input x'''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros',
                stretch=True):
        super(NormConv2d, self).__init__()
        self.group_conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
            kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, 
            groups=in_channels, bias=False, padding_mode=padding_mode)
        self.norm_conv = FixWeightConv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, 
            bias=False, padding_mode=padding_mode)
        self.norm_conv.weight.data = self.norm_conv.weight.data.new_ones(1,1,kernel_size,kernel_size)
        self.stretch = stretch
        if stretch:
            self.out_1x1_conv = torch.nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        
        self.view_in, self.view_out = (in_channels, kernel_size*kernel_size), (in_channels, 1, 1, 1)
        
        self._init_weight()
        
        
    def forward(self, x):
        # 已将输入normalized， 其长度未受理
        if ((self.group_conv.weight.data.view(self.view_in).norm(dim=1) - 1).abs()>1e-4).any():
            self.group_conv.weight.data = self.group_conv.weight.data / self.group_conv.weight.data.view(
                self.view_in).norm(dim=1).view(self.view_out)
        N, C, H, W = x.shape
        x_1chn = x.view(N*C, 1, H, W)
        x_group = self.group_conv(x)
        N, C, H, W = x_group.shape
        # x_square = self.norm_conv(x_1chn * x_1chn)+1e-4
        x_square = self.norm_conv(torch.pow(x_1chn, 2))+1e-4
        if (x_square < 0).any():
            import pdb; pdb.set_trace()
        x_norm = x_square.sqrt().view(N, C, H, W).contiguous()
        x_normalized = x_group / x_norm
        # if (torch.isnan(x_normalized)).any():
        #     import pdb; pdb.set_trace()
        if  self.stretch:
            x_stretch = self.out_1x1_conv(x_normalized)
            return x_stretch
        return x_normalized
    

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class NormConv2d(torch.nn.Module):
    '''decouple the direction and length of weight and input x'''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros',
                stretch=True, merge_amp=True):
        super(NormConv2d, self).__init__()
        self.group_conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
            kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, 
            groups=in_channels, bias=False, padding_mode=padding_mode)
        # import pdb; pdb.set_trace()
        self.norm_conv = FixWeightConv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, 
            groups=in_channels, bias=False, padding_mode=padding_mode)
        # self.norm_conv.weight.data = self.norm_conv.weight.data.new_ones(1,1,kernel_size,kernel_size)
        self.norm_conv.weight.data.fill_(1)
        self.stretch = stretch
        self.merge_amp = merge_amp
        if stretch:
            self.out_1x1_conv = torch.nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        if merge_amp:        
            self.merge_amp_weight = torch.nn.Parameter(torch.Tensor(in_channels))
        self.view_in, self.view_out = (in_channels, kernel_size*kernel_size), (in_channels, 1, 1, 1)
        
        self._init_weight()
        
        
    def forward(self, x):
        # 已将输入normalized， 其长度未受理
        # 这是不是不对啊， w/|w| conv x/|x|
        # import pdb; pdb.set_trace()
        if ((self.group_conv.weight.data.view(self.view_in).norm(dim=1) - 1).abs()>1e-4).any():
            self.group_conv.weight.data = self.group_conv.weight.data / self.group_conv.weight.data.view(
                self.view_in).norm(dim=1).view(self.view_out)
        x_group = self.group_conv(x)
        x_sqr_in = torch.pow(x, 2)
        x_square = self.norm_conv(x_sqr_in)+1e-8
        x_norm = x_square.sqrt()
        x_normalized = x_group / x_norm
        # print(x_normalized[0,0,100,100],'\n',x_norm[0,0,100,100],'\n', x[0,0,99:102,99:102],'\n', self.group_conv.weight.data[0,0])
        # print('my calculate:', (x[0,0,99:102,99:102]/x_norm[0,0,100,100]*self.group_conv.weight.data[0,0]).sum())
        x_out = x_normalized
        if self.merge_amp:
            x_out = x_norm * self.merge_amp_weight[:, None, None] + x_normalized

        if  self.stretch:
            x_stretch = self.out_1x1_conv(x_out)
            return x_stretch
        return x_out
    

    def _init_weight(self):
        if self.merge_amp:
            self.merge_amp_weight.data.fill_(1)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ResNormConv2d(torch.nn.Module):
    '''decouple the direction and length of weight and input x'''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros',
                stretch=True):
        super(ResNormConv2d, self).__init__()
        self.group_conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
            kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, 
            groups=in_channels, bias=False, padding_mode=padding_mode)
        self.norm_conv = FixWeightConv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, 
            groups=in_channels, bias=False, padding_mode=padding_mode)
        self.norm_conv.weight.data.fill_(1)
        self.stretch = stretch
        if stretch:
            self.out_1x1_conv = torch.nn.Conv2d(in_channels*2, out_channels, 1, bias=bias)
        self.view_in, self.view_out = (in_channels, kernel_size*kernel_size), (in_channels, 1, 1, 1)
        
        # self._init_weight()
        
        
    def forward(self, x):
        # 已将输入normalized， 其长度未受理
        if ((self.group_conv.weight.data.view(self.view_in).norm(dim=1) - 1).abs()>1e-4).any():
            self.group_conv.weight.data = self.group_conv.weight.data / self.group_conv.weight.data.view(
                self.view_in).norm(dim=1).view(self.view_out)
        x_group = self.group_conv(x)
        x_sqr_in = torch.pow(x, 2)
        x_square = self.norm_conv(x_sqr_in)+1e-8
        x_norm = x_square.sqrt()
        x_normalized = x_group / x_norm
        x_out = x_normalized
        if  self.stretch:
            x_interp = x
            if x.shape != x_out.shape:
                x_interp = F.interpolate(x, x_out.shape[-2:], mode='bilinear', align_corners=True)
            x_interp = torch.cat((x_interp, x_out), dim=1)
            x_stretch = self.out_1x1_conv(x_interp)
            return x_stretch
        return x_out
    

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class MeanResNormConv2d(torch.nn.Module):
    '''decouple the direction and length of weight and input x'''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, padding_mode='zeros',
                stretch=True):
        super(MeanResNormConv2d, self).__init__()
        self.group_conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
            kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, 
            groups=in_channels, bias=False, padding_mode=padding_mode)
        self.norm_conv = FixWeightConv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, 
            groups=in_channels, bias=False, padding_mode=padding_mode)
        self.norm_conv.weight.data.fill_(1)
        self.stretch = stretch
        self.in_channels = in_channels
        if stretch:
            self.out_1x1_conv = torch.nn.Conv2d(in_channels*2, out_channels, 1, bias=False)
        self.view_in, self.view_out = (in_channels, kernel_size*kernel_size), (in_channels, 1, 1, 1)
        
        # self._init_weight()
        
        
    def forward(self, x):
        # 已将输入normalized， 其长度未受理
        if ((self.group_conv.weight.data.view(self.view_in).norm(dim=1) - 1).abs()>1e-4).any():
            self.group_conv.weight.data = self.group_conv.weight.data / self.group_conv.weight.data.view(
                self.view_in).norm(dim=1).view(self.view_out)
        x_group = self.group_conv(x)
        x_sqr_in = torch.pow(x, 2)
        x_square = self.norm_conv(x_sqr_in)+1e-8
        x_norm = x_square.sqrt()
        x_normalized = x_group / x_norm
        x_out = x_normalized
        if  self.stretch:
            x_interp = x
            if x.shape != x_out.shape:
                x_interp = F.interpolate(x, x_out.shape[-2:], mode='bilinear', align_corners=True)
            x_interp = torch.cat((x_interp, x_out), dim=1)
            x_stretch = self.out_1x1_conv(x_interp)
            # import pdb; pdb.set_trace()
            mean_bias =  x_stretch.mean(dim=(2,3))[...,None, None]
            x_stretch = x_stretch - mean_bias
            return x_stretch
        return x_out
    

    def _init_weight(self):
        # for m in self.modules():
        #     if isinstance(m, torch.nn.Conv2d):
        #         torch.nn.init.kaiming_normal_(m.weight)
        #     elif isinstance(m, torch.nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        import pdb; pdb.set_trace()

class MyConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = torch.nn.modules.utils._pair(kernel_size)
        stride = torch.nn.modules.utils._pair(stride)
        padding = torch.nn.modules.utils._pair(padding)
        dilation = torch.nn.modules.utils._pair(dilation)
        super(MyConv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.weight = torch.nn.Parameter(torch.Tensor(
            out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        # import pdb; pdb.set_trace()
        N, C, H, W = input.shape
        x_out = input.new_zeros(N, self.weight.shape[0], H+self.padding[0]*2-self.kernel_size[0]+1, W+self.padding[1]*2-self.kernel_size[1]+1)
        x_padded = input.new_zeros(N, C, H+self.padding[0]*2, W+self.padding[1]*2)
        x_padded[:,:, self.padding[0]:H+self.padding[0], self.padding[1]:W+self.padding[1]] = input
        for h_i in range(x_padded.shape[2]-self.kernel_size[0]//2-1):
            for w_i in range(x_padded.shape[3]-self.kernel_size[1]//2-1):
                # import pdb; pdb.set_trace()
                x_crop = x_padded[:,:,h_i:h_i+self.kernel_size[0], w_i:w_i+self.kernel_size[1]]
                x_match_mul = (x_crop[:,None]*self.weight[None])
                # x_match_mul = (x_crop[:,None]*self.weight[None, :, ::-1, ::-1])
                x_out[:,:,h_i, w_i] = x_match_mul.sum(-1).sum(-1).sum(-1)
        # import pdb; pdb.set_trace()
        return x_out

        # return F.conv2d(input, self.weight, self.bias, self.stride,
        #                 self.padding, self.dilation, self.groups)

class MyNormConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = torch.nn.modules.utils._pair(kernel_size)
        stride = torch.nn.modules.utils._pair(stride)
        padding = torch.nn.modules.utils._pair(padding)
        dilation = torch.nn.modules.utils._pair(dilation)
        super(MyNormConv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.weight = torch.nn.Parameter(torch.Tensor(
            out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        # import pdb; pdb.set_trace()
        N, C, H, W = input.shape
        x_out = input.new_zeros(N, self.weight.shape[0], H+self.padding[0]*2-self.kernel_size[0]+1, W+self.padding[1]*2-self.kernel_size[1]+1)
        x_padded = input.new_zeros(N, C, H+self.padding[0]*2, W+self.padding[1]*2)
        x_padded[:,:, self.padding[0]:H+self.padding[0], self.padding[1]:W+self.padding[1]] = input
        weight_normalized = self.weight.data / self.weight.data.view(3,9).norm(dim=-1).view(3,1,1,1)
        for h_i in range(x_padded.shape[2]-self.kernel_size[0]//2-1):
            for w_i in range(x_padded.shape[3]-self.kernel_size[1]//2-1):
                # import pdb; pdb.set_trace()
                x_crop = x_padded[:,:,h_i:h_i+self.kernel_size[0], w_i:w_i+self.kernel_size[1]].clone()
                # import pdb; pdb.set_trace()
                x_crop_normalized = x_crop / x_crop.view(N*C, self.kernel_size[0]*self.kernel_size[1]).norm(dim=-1).view(N,C,1,1)
                if self.groups:
                    x_match_mul = (x_crop_normalized*weight_normalized[None,:,0])
                    x_out[:,:,h_i, w_i] = x_match_mul.sum(-1).sum(-1)
                else:
                    x_match_mul = (x_crop_normalized[:,None]*weight_normalized[None])
                    # x_match_mul = (x_crop[:,None]*self.weight[None])
                    x_out[:,:,h_i, w_i] = x_match_mul.sum(-1).sum(-1).sum(-1)

                if h_i +1 == 100+self.padding[0] and w_i +1 ==100+self.padding[1]:
                    print(x_out[0,0,0,0])
                    print(x_crop.view(N*C, self.kernel_size[0]*self.kernel_size[1]).norm(dim=-1).view(N,C,1,1)[0,0,0,0])
                    print(x_crop[0,0])
                    print(weight_normalized[0,0])
                    print('my calculate:', (x_crop[0,0]/x_crop.view(N*C, self.kernel_size[0]*self.kernel_size[1]).norm(dim=-1).view(N,C,1,1)[0,0]*weight_normalized[0,0]).sum())
                    print('x_crop_normalizede:', x_crop_normalized)
        # import pdb; pdb.set_trace()
        return x_out

        # return F.conv2d(input, self.weight, self.bias, self.stride,
        #                 self.padding, self.dilation, self.groups)
