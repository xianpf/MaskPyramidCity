import sys, os, time, datetime, curses, logging
sys.path.insert(0, os.path.abspath(__file__+'/../..'))
import argparse, glob, re, cv2
from tqdm import tqdm
import numpy as np
import pynvml, subprocess

import torch
from torch import nn
import torch.nn.functional as F

from maskpyramid.config import cfg
# from maskpyramid.modeling.backbone import resnet
from dataloaders.build import make_data_loader
from maskpyramid.utils.metrics import Evaluator
from maskpyramid.utils.lr_scheduler import LR_Scheduler
from maskpyramid.utils.tools import setup_logger, mkdir, collect_env_info
from maskpyramid.utils.tools import TensorboardSummary, flatten_list
from grow_model.norm_conv import NormConv2d

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        # self.bn = BatchNorm(planes)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, inplanes = 512):
        super(ASPP, self).__init__()
        # inplanes = 512
        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU())
        # self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        # self.bn1 = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(1280, inplanes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class double_conv_ori(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            NormConv2d(in_ch, out_ch, 3, padding=1, bias=True, merge_amp=False),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            NormConv2d(out_ch, out_ch, 3, padding=1, bias=True, merge_amp=False),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(torch.nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.aspp = ASPP('drn', output_stride=16, BatchNorm=nn.BatchNorm2d)
        self.asppconv = nn.Conv2d(256, 512, 1)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.asppconv(self.aspp(x5))
        x = self.up1(x6, x4)
        # x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        # return F.sigmoid(x)
        return x

class UnetAspp(nn.Module):
    def __init__(self, num_classes=21):
        super(UnetAspp, self).__init__()
        self.unet = UNet(n_channels=3, n_classes=num_classes)

    def forward(self, x513):
        x_unet = self.unet(x513)
        return x_unet

class ResUnetAspp(torch.nn.Module):
    def __init__(self, n_channels=3, num_classes=21):
        super(ResUnetAspp, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.aspp = ASPP()
        # self.asppconv = nn.Conv2d(256, 512, 1)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, num_classes)
        self._init_weight()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # x6 = self.asppconv(self.aspp(x5))
        x6 = self.aspp(x5)
        x = self.up1(x6, x4)
        # x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        # return F.sigmoid(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class CategoryPyramid(nn.Module):
    def __init__(self, cfg):
        super(CategoryPyramid, self).__init__()
        num_classes = cfg.DATALOADER.NUM_CLASSES
        self.init_conv = double_conv(2+256, 32)
        self.up1 = up(256+32, 32)
        self.up2 = up(256+32, 32)
        self.up3 = up(256+32, 32)
        self.up4 = up(256+32, 32)
        self.up5 = up(256+32, 32)
        self.up256 = up(128+32, 32)
        self.up513 = up(64+32, 32)
        self.outc = outconv(32, 1)
        # self.up1 = up(256+256, 64)
        # self.up2 = up(256+64, 64)
        # self.up3 = up(256+64, 64)
        # self.up4 = up(256+64, 64)
        # self.up5 = up(256+64, 64)
        # self.up6 = up(128+64, 64)
        # self.up7 = up(64+64, 64)
        # self.outc = outconv(64, 1)
        self.up_list = nn.ModuleList([None, self.up1, self.up2, self.up3, 
            self.up4, self.up5])

class MaskPyramidsV8(nn.Module):
    def __init__(self, cfg):
        super(MaskPyramids, self).__init__()
        self.cfg = cfg
        self.resnet50 = ResNet50(cfg)
        num_classes = cfg.DATALOADER.NUM_CLASSES

        # self.res_7_category = nn.Conv2d()
        self.ins_cat_pyr_convs = [None,]*11
        for i in range(8):
            self.ins_cat_pyr_convs.append(
                CategoryPyramid(cfg)
            )
        self.ins_cat_pyr_convs = nn.ModuleList(self.ins_cat_pyr_convs)

        # self.sematic_conv_5 = Bottleneck(512+256, 64, 256, 1, False, 1, 1, nn.BatchNorm2d)
        self.sematic_conv_5 = Bottleneck(512, 64, 256, 1, False, 1, 1, nn.BatchNorm2d)
        self.sematic_conv_4 = Bottleneck(512, 64, 256, 1, False, 1, 1, nn.BatchNorm2d)
        self.sematic_conv_3 = Bottleneck(512, 64, 256, 1, False, 1, 1, nn.BatchNorm2d)
        self.sematic_conv_2 = Bottleneck(512, 64, 256, 1, False, 1, 1, nn.BatchNorm2d)
        self.sematic_conv_1 = Bottleneck(512, 64, 256, 1, False, 1, 1, nn.BatchNorm2d)
        self.sem_size5 = nn.Conv2d(256, num_classes, 1)
        self.sem_size9 = nn.Conv2d(256, num_classes, 1)
        self.sem_size17 = nn.Conv2d(256, num_classes, 1)
        self.sem_size33 = nn.Conv2d(256, num_classes, 1)
        self.sem_size65 = nn.Conv2d(256, num_classes, 1)
        self.sem_size129 = nn.Conv2d(256, num_classes, 1)
        self.sem_size513 = nn.Conv2d(256, num_classes, 1)


        self.cs_criteron = nn.CrossEntropyLoss()
        self.class_criteron = nn.CrossEntropyLoss()
        self.sematic_criteron = nn.CrossEntropyLoss(ignore_index=cfg.DATALOADER.IGNORE_INDEX)

        self.cs_loss_factor = 1.0
        self.miss_loss_factor = 1.0

        self.log_dict = {}
        

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def compute_mask(self, level, feature, pyramids):
        masks = []
        for pyr in pyramids:
            # import pdb; pdb.set_trace()
            pyr_conv = self.ins_cat_pyr_convs[pyr.tar_cat].convs[level - pyr.init_level]
            out_conv = self.ins_cat_pyr_convs[pyr.tar_cat].outs[level - pyr.init_level]
            if pyr.init_level == level:
                indicate_mask = pyr.get_indicate_mask(feature)[None]
                conv_input = torch.cat((indicate_mask, feature), dim=1)
                mask_logit = pyr_conv(conv_input)
                pyr.set_mask_logits(level, mask_logit)
                mask = out_conv(mask_logit)
            elif level == 6:
                # import pdb; pdb.set_trace()
                last_logits = pyr.get_mask_logits(level-1)
                up_last_logits = F.interpolate(last_logits, feature.shape[2:], mode='bilinear', align_corners=True)
                mask = out_conv(up_last_logits)
            else:
                # import pdb; pdb.set_trace()
                last_logits = pyr.get_mask_logits(level-1)
                up_last_logits = F.interpolate(last_logits, feature.shape[2:], mode='bilinear', align_corners=True)
                conv_input = torch.cat((feature, up_last_logits), dim=1)
                mask_logit = pyr_conv(conv_input)
                pyr.set_mask_logits(level, mask_logit)
                mask = out_conv(mask_logit)
            pyr.set_mask(level, mask)
            masks.append(mask)
        # import pdb; pdb.set_trace()
        masks = torch.cat(masks, dim=1)

        return masks

    def compute_loss(self, level, pyramids, target_levels, target_support_pyramids):
        # TODO: multi target cross entropy loss
        losses = []
        miss_losses = []
        # class_losses = []
        # covered_idx = []
        for pyramid in pyramids:
            mask_logits = pyramid.get_mask(level)
            if pyramid.target_idx is not None:
                target_mask = target_levels[7-level][0, [pyramid.target_idx]]
                loss_cs = self.cs_criteron(mask_logits, target_mask.squeeze(1).long())
                losses.append(loss_cs)

                # if loss_cs > 1e20 or torch.isnan(loss_cs):
                if torch.isnan(loss_cs):
                    import pdb; pdb.set_trace()

                # import pdb; pdb.set_trace()
                # if level == 3:
                #     import pdb; pdb.set_trace()
                #     target_label = target_levels['labels'][[pyramid.target_idx]]
                #     loss_class = self.class_criteron(pyramid.class_logits, target_label)
                #     class_losses.append(loss_class)

        # import pdb; pdb.set_trace()
        # TODO: 检查未被追踪的target_idx
        # TODO: 避免惩罚该target 指导的pyramid， 不存在这个问题。。。
        for i, t_match_list in  enumerate(target_support_pyramids):
            if not t_match_list:    # target 没有match到pyramid
                miss_target_map = target_levels[7-level][0, i]
                if miss_target_map.sum():
                    miss_pos = miss_target_map.nonzero()
                    # import pdb; pdb.set_trace()
                    all_masks = torch.cat([i_p.get_mask(level) for i_p in pyramids], dim=1)
                    loss_miss = all_masks[0,:,miss_pos[:,0], miss_pos[:,1]].mean()

        cs_loss = sum(loss for loss in losses)/len(losses)\
            if len(losses) else mask_logits.sum()*0
        miss_loss = sum(loss for loss in miss_losses) / len(miss_losses)\
            if len(miss_losses) else mask_logits.sum()*0
        # class_loss = sum(loss for loss in class_losses) / len(class_losses)\
        #     if len(class_losses) else mask_logits.sum()*0

        # resloss = cs_loss * self.cs_loss_factor + miss_loss * self.miss_loss_factor + \
        #     class_loss * self.class_loss_factor

        resloss = cs_loss * self.cs_loss_factor + miss_loss * self.miss_loss_factor
                    
        # resloss = (sum(loss for loss in losses)/len(losses)\
        #     if len(losses) else mask_logits.sum()*0) * self.cs_loss_factor + \
        #     (sum(loss for loss in miss_losses) / len(miss_losses)\
        #     if len(miss_losses) else mask_logits.sum()*0) * self.miss_loss_factor
        return resloss

    def prepare_targets_archive(self, target, down_size, mode='sematic', pyramids=[]):

        N, img_h, img_w = target['label'].shape
        h_ratio = int(np.ceil(img_h / down_size[0]))
        w_ratio = int(np.ceil(img_w / down_size[1]))
        h_match_size = h_ratio * down_size[0]
        w_match_size = w_ratio * down_size[1]

        inst_mode_num_stat = F.interpolate(target['instance'].unsqueeze(1).float(), (h_match_size, 
            w_match_size), mode='nearest').squeeze(1).long().view((N, down_size[0], 
            h_ratio, down_size[1], w_ratio)).permute(0,1,3,2,4).contiguous().view((N,
            )+down_size+(-1,))
        mode_num_instance = inst_mode_num_stat.mode(-1)[0]

        if mode == 'instance':
            # # import pdb; pdb.set_trace()
            # inst_mode_num_stat = F.interpolate(target['instance'].unsqueeze(1).float(), (h_match_size, 
            #     w_match_size), mode='nearest').squeeze(1).long().view((N, down_size[0], 
            #     h_ratio, down_size[1], w_ratio)).permute(0,1,3,2,4).contiguous().view((N,
            #     )+down_size+(-1,))
            # mode_num_instance = inst_mode_num_stat.mode(-1)[0]

            return mode_num_instance

        elif mode == 'sematic':
            # mode_num_stat = F.interpolate(target['label'].unsqueeze(1).float(), (h_match_size, 
            #     w_match_size), mode='nearest').squeeze(1).long().view((N, down_size[0], 
            #     h_ratio, down_size[1], w_ratio)).permute(0,1,3,2,4).contiguous().view((N,
            #     )+down_size+(-1,))
            # mode_num_sematic = mode_num_stat.mode(-1)[0]
            # import pdb; pdb.set_trace()
            mode_num_sematic = mode_num_instance.clone()
            for bid in range(len(mode_num_sematic)):
                for inst_num in mode_num_instance[bid].unique():
                    mode_num_sematic[bid][mode_num_instance[bid] == inst_num] = target['label']\
                        [bid][target['instance'][bid] == inst_num].unique().item()

            return mode_num_sematic


        else:
            import pdb; pdb.set_trace()
        return 0

    def prepare_targets(self, target, down_size, mode='sematic', pyramids=[]):

        N, img_h, img_w = target['label'].shape
        h_ratio = int(np.ceil(img_h / down_size[0]))
        w_ratio = int(np.ceil(img_w / down_size[1]))
        h_match_size = h_ratio * down_size[0]
        w_match_size = w_ratio * down_size[1]

        inst_mode_num_stat = F.interpolate(target['instance'].unsqueeze(1).float(), (h_match_size, 
            w_match_size), mode='nearest').squeeze(1).long().view((N, down_size[0], 
            h_ratio, down_size[1], w_ratio)).permute(0,1,3,2,4).contiguous().view((N,
            )+down_size+(-1,))
        mode_num_instance = inst_mode_num_stat.mode(-1)[0]

        if mode == 'instance':
            return mode_num_instance

        elif mode == 'sematic':
            mode_num_sematic = mode_num_instance.clone()
            for bid in range(len(mode_num_sematic)):
                for inst_num in mode_num_instance[bid].unique():
                    mode_num_sematic[bid][mode_num_instance[bid] == inst_num] = target['label']\
                        [bid][target['instance'][bid] == inst_num].unique().item()
            # import pdb; pdb.set_trace()
            return mode_num_sematic
        else:
            import pdb; pdb.set_trace()
        return 0

    def compute_sematic(self, feature, target):
        # import pdb; pdb.set_trace()
        x_sem_size5 = self.sem_size5(feature[5])
        up_6_5 = F.interpolate(feature[5], feature[4].shape[2:], mode='bilinear', align_corners=True)
        sematic_5 = self.sematic_conv_5(torch.cat((feature[4], up_6_5), dim=1))
        x_sem_size9 = self.sem_size9(sematic_5)
        up_5_4 = F.interpolate(sematic_5, feature[3].shape[2:], mode='bilinear', align_corners=True)
        sematic_4 = self.sematic_conv_4(torch.cat((feature[3], up_5_4), dim=1))
        x_sem_size17 = self.sem_size17(sematic_4)
        up_4_3 = F.interpolate(sematic_4, feature[2].shape[2:], mode='bilinear', align_corners=True)
        sematic_3 = self.sematic_conv_3(torch.cat((feature[2], up_4_3), dim=1))
        x_sem_size33 = self.sem_size33(sematic_3)
        up_3_2 = F.interpolate(sematic_3, feature[1].shape[2:], mode='bilinear', align_corners=True)
        sematic_2 = self.sematic_conv_2(torch.cat((feature[1], up_3_2), dim=1))
        x_sem_size65 = self.sem_size65(sematic_2)
        up_2_1 = F.interpolate(sematic_2, feature[0].shape[2:], mode='bilinear', align_corners=True)
        sematic_1 = self.sematic_conv_1(torch.cat((feature[0], up_2_1), dim=1))
        x_sem_size129 = self.sem_size129(sematic_1)
        up_final = F.interpolate(sematic_1, self.cfg.DATALOADER.CROP_SIZE, mode='bilinear', align_corners=True)
        x_sem_size513 = self.sem_size513(up_final)

        # import pdb; pdb.set_trace()
        sematic_targets_5 = self.prepare_targets(target, mode='sematic', down_size=(5,5))
        sematic_targets_9 = self.prepare_targets(target, mode='sematic', down_size=(9,9))
        sematic_targets_17 = self.prepare_targets(target, mode='sematic', down_size=(17,17))
        sematic_targets_33 = self.prepare_targets(target, mode='sematic', down_size=(33,33))
        sematic_targets_65 = self.prepare_targets(target, mode='sematic', down_size=(65,65))
        sematic_targets_129 = self.prepare_targets(target, mode='sematic', down_size=(129,129))


        sematic_loss_5 = self.sematic_criteron(x_sem_size5, sematic_targets_5)
        sematic_loss_9 = self.sematic_criteron(x_sem_size9, sematic_targets_9)
        sematic_loss_17 = self.sematic_criteron(x_sem_size17, sematic_targets_17)
        sematic_loss_33 = self.sematic_criteron(x_sem_size33, sematic_targets_33)
        sematic_loss_65 = self.sematic_criteron(x_sem_size65, sematic_targets_65)
        sematic_loss_129 = self.sematic_criteron(x_sem_size129, sematic_targets_129)
        sematic_loss_513 = self.sematic_criteron(x_sem_size513, target['label'].long())

        # import pdb; pdb.set_trace()
        sematic_loss = torch.tensor([sematic_loss_5, sematic_loss_9, sematic_loss_17, sematic_loss_33, 
            sematic_loss_65, sematic_loss_129, sematic_loss_513], requires_grad=True)
        # sematic_loss = torch.tensor([sematic_loss_5, sematic_loss_9, sematic_loss_17, sematic_loss_33, 
        #     sematic_loss_65, sematic_loss_129, sematic_loss_513])

        return sematic_loss, [x_sem_size5, x_sem_size9, x_sem_size17, x_sem_size33, x_sem_size65, 
            x_sem_size129, x_sem_size513]

    def forward_singel_level(self, curr_level, inst_pyramids, x_curr, i, level_sizes, 
        target_support_pyramids, target_levels, losses_i):
        new_pos_limit = [100, 50, 50, 50, 50, 50, 50]
        new_pos_quota = 80
        if x_curr[[i]].abs().max() > 1e20 or torch.isnan(x_curr[[i]].max()):
        # if torch.isnan(x_curr[[i]].max()):
            print(curr_level, '\n', x_curr[[i]])
            import pdb; pdb.set_trace()
        # 生成 upsample mask，对现有的mask pyramids
        self.compute_mask(curr_level, x_curr[[i]], inst_pyramids)
        # TODO: 考虑其他的new_masks计算方法，比如说 multi target cross entropy loss 中的单一channel
        new_masks_minus = torch.cat([i_p.get_mask(curr_level)[:,[1]] - i_p.get_mask(curr_level)[:,[0]] for i_p in inst_pyramids], dim=1)
        new_masks_softmax = F.softmax(new_masks_minus,dim=1)
        # avg_sharing = 1.0 / len(inst_pyramids)
        # num_pixels = int(new_masks_softmax.shape[-1]*new_masks_softmax.shape[-2])
        # top_percent = new_masks_softmax.view(-1).topk(int(num_pixels*(1-0.3)))[0][-1].item()
        # max_topk = new_masks_softmax.max(dim=1)[0].view(-1).topk(num_pixels-3)[0][-1].item()
        max_topk = new_masks_softmax.max(dim=1)[0].view(-1).topk(8, largest=False)[0][-1].item()
        # 这里非常的有趣，保证最少选拔8人，如果KOL话语权占不到5%，那就诞生新的KOL proposal
        # pending_thresh越高，新增的new_pos越多 所以 max_topk 应该是保底， 应该配合比例
        pending_thresh = max(0.02, max_topk)
        new_pos = torch.nonzero(new_masks_softmax[0].max(dim=0)[0] < pending_thresh)
        # if len(new_pos) > new_pos_limit[curr_level]:
        #     # import pdb; pdb.set_trace()
        #     raw_pos = new_masks_softmax.max(dim=1)[0].view(-1).topk(new_pos_limit[curr_level], largest=False)[1]
        #     new_pos_0 = raw_pos // x_curr.shape[-1]
        #     new_pos_1 = raw_pos % x_curr.shape[-1]
        #     new_pos = torch.cat((new_pos_0.view(-1,1), new_pos_1.view(-1,1)), dim=1)
        # import pdb; pdb.set_trace()
        if len(inst_pyramids) + len(new_pos) > new_pos_quota:
            available_number = max(0, new_pos_quota - len(inst_pyramids))
            if available_number:
                raw_pos = new_masks_softmax.max(dim=1)[0].view(-1).topk(available_number, largest=False)[1]
                new_pos_0 = raw_pos // x_curr.shape[-1]
                new_pos_1 = raw_pos % x_curr.shape[-1]
                new_pos = torch.cat((new_pos_0.view(-1,1), new_pos_1.view(-1,1)), dim=1)
            else:
                new_pos = []

        new_occupy = 1.0*len(new_pos) / x_curr.shape[-2] / x_curr.shape[-1]
        # if new_occupy > 0.5 or len(new_pos) <8:
        # if new_occupy > 0.5 or len(new_pos) <8-1:
        #     print('new_occupy:{}| len(new_pos):{}'.format(new_occupy, len(new_pos)))
            # import pdb; pdb.set_trace()
        new_pyramids = [InstancePyramid(pos, curr_level, level_sizes) for pos in new_pos]
        self.compute_mask(curr_level, x_curr[[i]], new_pyramids, True)
        # 出清没有领地的pyramid 在所有pixel都进不了前3
        # 统计没有pyramid的targets
        # 额外惩罚霸占位置的pyramid，保护弱势应得的 pyramid
        merit_pyramids_idx = new_masks_softmax.topk(2, dim=1)[1].unique()
        # merit_pyramids_idx = new_masks_softmax.topk(3, dim=1)[1].unique()
        merit_pyramids = [inst_pyramids[i] for i in range(len(inst_pyramids)) if i in merit_pyramids_idx]

        if self.training:
            target_len_before = sum([len(l) for l in target_support_pyramids])
            # target_len_1_before = sum([len(l) for l in target_support_pyramids_0])
            # import pdb; pdb.set_trace()
            for reduce_i in range(len(inst_pyramids)):
                if reduce_i not in merit_pyramids_idx:
                    die_id = inst_pyramids[reduce_i].idx
                    die_target_idx = inst_pyramids[reduce_i].target_idx
                    if die_target_idx:
                        target_support_pyramids[die_target_idx].remove(die_id)
                        # target_support_pyramids_0[die_target_idx].remove(die_id)
            target_len_after = sum([len(l) for l in target_support_pyramids])
            # target_len_1_after = sum([len(l) for l in target_support_pyramids_0])
            # if target_len_1_before != target_len_1_after:
            #     import pdb; pdb.set_trace()

        # import pdb; pdb.set_trace()
        inst_pyramids = merit_pyramids + new_pyramids
        # self.log_dict.update({'pyr_num_l1': len(inst_pyramids)})
        self.log_dict.update({'pyr_num_l'+str(curr_level): len(inst_pyramids)})
        if self.training:
            self.match_target(curr_level, new_pyramids, target_levels, target_support_pyramids)
            loss = self.compute_loss(curr_level, inst_pyramids, target_levels, target_support_pyramids)
            losses_i.append(loss)
        # import pdb; pdb.set_trace()
        
        return inst_pyramids

    def forward(self, image, targets=None):
        output_dict = dict()
        N, _, img_h, img_w = image.shape
        xs_r50 = self.resnet50(image)
        sematic_loss, sematic_out = self.compute_sematic(xs_r50, targets)
        if self.cfg.SOLVER.SEMATIC_ONLY:
            output_dict['loss_dict'] = {'sematic': sematic_loss.mean(), 'instance': 0}
            output_dict['sematic_out'] = sematic_out
            output_dict['targets'] = targets
            output_dict['ins_pyramids'] =[[]]
            return output_dict

        # import pdb; pdb.set_trace()
        instance_categories = [11, 12, 13, 14, 15, 16, 17, 18]
        lvl_sizes = [tuple(f.shape[-2:]) for f in xs_r50[::-1]]+[(513,513)]

        # 调换顺序，找最大值处后， 先分配target， target指定了cat
        pyr_id_own_ins_target = [{} for _ in range(N)]
        pyr_losses = [[[] for _ in range(N)] for __ in range(7)]
        ins_pyramids = [[] for _ in range(N)]
        in_dict_watches = [dict() for _ in range(N)]
        for sem_lvl in range(6):
            lvl_size = lvl_sizes[sem_lvl]
            if targets is not None:
                inst_targets = self.prepare_targets(targets, mode='instance', down_size=lvl_size)
                sematic_targets = self.prepare_targets(targets, mode='sematic', down_size=lvl_size)
            for bid in range(N):
                # in_dict_watches[bid]['sem_target'] = sematic_targets
                # in_dict_watches[bid]['ins_target'] = inst_targets
                sematic_logits = sematic_out[sem_lvl][bid]
                max_idx_sematic = sematic_logits.max(0)[1]
                # free_space = torch.ones_like(max_idx_sematic)
                if self.training:
                    inst_target = inst_targets[bid]
                    sematic_target = sematic_targets[bid]
                    embeded_target = []
                    for pyr in ins_pyramids[bid]:
                        if pyr.target_idx not in inst_target.unique():
                            # import pdb; pdb.set_trace()
                            pyr.set_ins_target(sem_lvl, F.interpolate(pyr.get_ins_target(sem_lvl-1).float()[None, None], lvl_size, mode='nearest')[0,0].bool())
                    for ins_tar_id in inst_target.unique():
                        target_cat = sematic_target[inst_target == ins_tar_id].unique().item()
                        if target_cat not in instance_categories:
                            if target_cat != -1:
                                embeded_target.append(inst_target == ins_tar_id)
                            continue
                        else:
                            if ins_tar_id.item() in pyr_id_own_ins_target[bid].keys():
                                # import pdb; pdb.set_trace()
                                pyr = [pyr for pyr in ins_pyramids[bid] if pyr.idx == \
                                    pyr_id_own_ins_target[bid][ins_tar_id.item()]][0]
                                pyr.set_ins_target(sem_lvl, inst_target == ins_tar_id)
                            else:
                                # if len([flatten_list(ins_pyramids)]) > 80:
                                if len(ins_pyramids[bid]) > 30:
                                    continue
                                cat_logit = sematic_logits[target_cat]
                                logit_within_instar = cat_logit * (inst_target == ins_tar_id).float()
                                max_poses = torch.nonzero(logit_within_instar == logit_within_instar.max())
                                new_pyr = InstancePyramid(max_poses[0], sem_lvl, lvl_sizes)
                                new_pyr.target_idx = ins_tar_id
                                new_pyr.tar_cat = target_cat
                                new_pyr.set_ins_target(sem_lvl, inst_target == ins_tar_id)
                                ins_pyramids[bid].append(new_pyr)

                                pyr_id_own_ins_target[bid][ins_tar_id.item()] = new_pyr.idx
                                # free_space[inst_target == ins_tar_id] = 0
                else:
                    # category_idx = 11
                    masked_inscat_logits = torch.zeros_like(sematic_out[sem_lvl])
                    for cat_id in range(19):
                        masked_inscat_logits[:,cat_id][max_idx_sematic == cat_id] = sematic_out[sem_lvl]\
                            [:, cat_id][max_idx_sematic == cat_id]
                    # import pdb; pdb.set_trace()
                    for cat_id in instance_categories:
                        for bid in range(N):
                            cat_max_resource = (max_idx_sematic[bid] == cat_id).int()
                            while cat_max_resource.sum() > 0:
                                resource_max = masked_inscat_logits[bid, cat_id].max()
                                max_poses = torch.nonzero(masked_inscat_logits[bid, cat_id] == resource_max)
                                if len(max_poses) > 1:
                                    import pdb; pdb.set_trace()
                                new_pyr = InstancePyramid(max_poses[0], sem_lvl, lvl_sizes)
                                inst_pyramids[bid].append(new_pyr)
                                # pyr get target and predict map and compute loss
                                import pdb; pdb.set_trace()
                                if self.training:
                                    pyr_target = self.prepare_targets(targets, mode='instance', down_size=lvl_sizes[sem_lvl])
                            
                            
                # 查缺补漏
                if targets is not None and len(ins_pyramids[bid]):
                    mask_mass = sematic_out[sem_lvl][[bid], :11]
                    embeded_target_0s = torch.zeros_like(inst_target).bool()
                    # try:
                    #     embeded_target_0s = torch.zeros_like(embeded_target[0])
                    # except:
                    #     import pdb; pdb.set_trace()
                    for i in range(11):
                        if i not in sematic_target.unique():
                            embeded_target.insert(i, embeded_target_0s)
                    if len(embeded_target) != 11:
                        import pdb; pdb.set_trace()
                    embeded_target += [pyr.get_ins_target(sem_lvl) for pyr in ins_pyramids[bid]]
                    mask_ins = self.compute_mask(sem_lvl, feature=([image]+xs_r50)[-1-sem_lvl][[bid]], pyramids=ins_pyramids[bid])
                    # try:
                    #     embeded_target += [pyr.get_ins_target(sem_lvl) for pyr in ins_pyramids[bid]]
                    #     mask_ins = self.compute_mask(sem_lvl, feature=xs_r50[-1-sem_lvl][[bid]], pyramids=ins_pyramids[bid])
                    #     # mask_ins = self.compute_mask(sem_lvl, feature=([image]+xs_r50)[-1-sem_lvl][[bid]], pyramids=ins_pyramids[bid])
                    # except:
                    #     import pdb; pdb.set_trace()
                    
                    embeded_masks = torch.cat((mask_mass, mask_ins), dim=1)
                    embeded_target = torch.stack(embeded_target)
                    embeded_target_idx = embeded_target.max(0)[1][None]
                    embeded_target_idx[0][sematic_target==-1] = -1
                    mask_loss = self.sematic_criteron(embeded_masks, embeded_target_idx)
                    pyr_losses[sem_lvl][bid].append(mask_loss)
            # print('sem_lvl', sem_lvl,'len pyr', len(ins_pyramids))
            # import pdb; pdb.set_trace()


        self.log_dict.update({'InstPyr_inst_count': InstancePyramid.inst_count})
        # import pdb; pdb.set_trace()
        pyr_losses = list(flatten_list(pyr_losses))
        instance_loss = torch.tensor(pyr_losses, requires_grad=True)

        output_dict['loss_dict'] = {'sematic': sematic_loss.mean(), 'instance': instance_loss.mean()}
        output_dict['sematic_out'] = sematic_out
        output_dict['targets'] = targets
        output_dict['ins_pyramids'] =ins_pyramids
        # output_dict['in_dict_watches'] =in_dict_watches

        return output_dict

class MaskPyramidsV10(nn.Module):
    def __init__(self, cfg):
        super(MaskPyramids, self).__init__()
        self.cfg = cfg
        num_classes = cfg.DATALOADER.NUM_CLASSES

        self.inc = inconv(3, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 256)
        self.down4 = down(256, 256)
        self.down5 = down(256, 256)
        self.down6 = down(256, 256)
        self.down7 = down(256, 256)
        # self.aspp = ASPP('drn', output_stride=16, BatchNorm=nn.BatchNorm2d)
        # self.asppconv = nn.Conv2d(256, 512, 1)
        self.up1 = up(512, 256)
        self.up2 = up(512, 256)
        self.up3 = up(512, 256)
        self.up4 = up(512, 256)
        self.up5 = up(512, 256)
        self.up6 = up(256+128, 256)
        self.up7 = up(256+64, 256)
        # self.up7 = up(512, 256)
        self._sematic_outc = outconv(256, num_classes)

        self.ins_cat_pyr_convs = [None for _ in range(11)]
        for i in range(8):
            self.ins_cat_pyr_convs.append(
                CategoryPyramid(cfg)
            )
        self.ins_cat_pyr_convs = nn.ModuleList(self.ins_cat_pyr_convs)

        self.cs_criteron = nn.CrossEntropyLoss()
        self.class_criteron = nn.CrossEntropyLoss()
        self.sematic_criteron = nn.CrossEntropyLoss(ignore_index=cfg.DATALOADER.IGNORE_INDEX)

        self.cs_loss_factor = 1.0
        self.miss_loss_factor = 1.0

        self.log_dict = {}
        

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def compute_mask(self, level, feature, pyramids):
        masks = []
        for pyr in pyramids:
            import pdb; pdb.set_trace()
            pyr_conv = self.ins_cat_pyr_convs[pyr.tar_cat].convs[level - pyr.init_level]
            out_conv = self.ins_cat_pyr_convs[pyr.tar_cat].outs[level - pyr.init_level]
            if pyr.init_level == level:
                indicate_mask = pyr.get_indicate_mask(feature)[None]
                conv_input = torch.cat((indicate_mask, feature), dim=1)
                mask_logit = pyr_conv(conv_input)
                pyr.set_mask_logits(level, mask_logit)
                mask = out_conv(mask_logit)
            elif level == 6:
                # import pdb; pdb.set_trace()
                last_logits = pyr.get_mask_logits(level-1)
                up_last_logits = F.interpolate(last_logits, feature.shape[2:], mode='bilinear', align_corners=True)
                mask = out_conv(up_last_logits)
            else:
                # import pdb; pdb.set_trace()
                last_logits = pyr.get_mask_logits(level-1)
                up_last_logits = F.interpolate(last_logits, feature.shape[2:], mode='bilinear', align_corners=True)
                conv_input = torch.cat((feature, up_last_logits), dim=1)
                mask_logit = pyr_conv(conv_input)
                pyr.set_mask_logits(level, mask_logit)
                mask = out_conv(mask_logit)
            pyr.set_mask(level, mask)
            masks.append(mask)
        # import pdb; pdb.set_trace()
        masks = torch.cat(masks, dim=1)

        return masks

    def compute_mask_v10(self, feature, pyramids, bid):
        masks = []
        for pyr in pyramids[bid]:
            # import pdb; pdb.set_trace()
            pyr_conv = self.ins_cat_pyr_convs[pyr.tar_cat]
            (x1, x2, x3, x4, x5, x6, x7, x8) = [f[[bid]] for f in feature]
            pyr_features = feature[::-1][pyr.init_level:]
            # print([f.shape for f in pyr_features])
            indicate_mask = pyr.get_indicate_mask(pyr_features[0])[None]
            init_input = torch.cat((indicate_mask, pyr_features[0]), dim=1)
            x_up = pyr_conv.init_conv(init_input)
            # import pdb; pdb.set_trace()
            for up_lvl in range(1, 6- pyr.init_level):
                # print('x{}, x_up.shape:{}'.format(9- pyr.init_level-up_lvl, x_up.shape))
                # import pdb; pdb.set_trace()
                x_up = pyr_conv.up_list[up_lvl](x_up, pyr_features[up_lvl])
            x_up_256 = pyr_conv.up256(x_up, x2)
            x_up_513 = pyr_conv.up513(x_up_256, x1)
            pyr_mask = pyr_conv.outc(x_up_513)
            masks.append(pyr_mask)
            pyr.set_mask(513, pyr_mask)
        masks = torch.cat(masks, dim=1)

        return masks

    def compute_loss(self, level, pyramids, target_levels, target_support_pyramids):
        # TODO: multi target cross entropy loss
        losses = []
        miss_losses = []
        # class_losses = []
        # covered_idx = []
        for pyramid in pyramids:
            mask_logits = pyramid.get_mask(level)
            if pyramid.target_idx is not None:
                target_mask = target_levels[7-level][0, [pyramid.target_idx]]
                loss_cs = self.cs_criteron(mask_logits, target_mask.squeeze(1).long())
                losses.append(loss_cs)

                # if loss_cs > 1e20 or torch.isnan(loss_cs):
                if torch.isnan(loss_cs):
                    import pdb; pdb.set_trace()

                # import pdb; pdb.set_trace()
                # if level == 3:
                #     import pdb; pdb.set_trace()
                #     target_label = target_levels['labels'][[pyramid.target_idx]]
                #     loss_class = self.class_criteron(pyramid.class_logits, target_label)
                #     class_losses.append(loss_class)

        # import pdb; pdb.set_trace()
        # TODO: 检查未被追踪的target_idx
        # TODO: 避免惩罚该target 指导的pyramid， 不存在这个问题。。。
        for i, t_match_list in  enumerate(target_support_pyramids):
            if not t_match_list:    # target 没有match到pyramid
                miss_target_map = target_levels[7-level][0, i]
                if miss_target_map.sum():
                    miss_pos = miss_target_map.nonzero()
                    # import pdb; pdb.set_trace()
                    all_masks = torch.cat([i_p.get_mask(level) for i_p in pyramids], dim=1)
                    loss_miss = all_masks[0,:,miss_pos[:,0], miss_pos[:,1]].mean()

        cs_loss = sum(loss for loss in losses)/len(losses)\
            if len(losses) else mask_logits.sum()*0
        miss_loss = sum(loss for loss in miss_losses) / len(miss_losses)\
            if len(miss_losses) else mask_logits.sum()*0
        # class_loss = sum(loss for loss in class_losses) / len(class_losses)\
        #     if len(class_losses) else mask_logits.sum()*0

        # resloss = cs_loss * self.cs_loss_factor + miss_loss * self.miss_loss_factor + \
        #     class_loss * self.class_loss_factor

        resloss = cs_loss * self.cs_loss_factor + miss_loss * self.miss_loss_factor
                    
        # resloss = (sum(loss for loss in losses)/len(losses)\
        #     if len(losses) else mask_logits.sum()*0) * self.cs_loss_factor + \
        #     (sum(loss for loss in miss_losses) / len(miss_losses)\
        #     if len(miss_losses) else mask_logits.sum()*0) * self.miss_loss_factor
        return resloss

    def prepare_targets_archive(self, target, down_size, mode='sematic', pyramids=[]):

        N, img_h, img_w = target['label'].shape
        h_ratio = int(np.ceil(img_h / down_size[0]))
        w_ratio = int(np.ceil(img_w / down_size[1]))
        h_match_size = h_ratio * down_size[0]
        w_match_size = w_ratio * down_size[1]

        inst_mode_num_stat = F.interpolate(target['instance'].unsqueeze(1).float(), (h_match_size, 
            w_match_size), mode='nearest').squeeze(1).long().view((N, down_size[0], 
            h_ratio, down_size[1], w_ratio)).permute(0,1,3,2,4).contiguous().view((N,
            )+down_size+(-1,))
        mode_num_instance = inst_mode_num_stat.mode(-1)[0]

        if mode == 'instance':
            # # import pdb; pdb.set_trace()
            # inst_mode_num_stat = F.interpolate(target['instance'].unsqueeze(1).float(), (h_match_size, 
            #     w_match_size), mode='nearest').squeeze(1).long().view((N, down_size[0], 
            #     h_ratio, down_size[1], w_ratio)).permute(0,1,3,2,4).contiguous().view((N,
            #     )+down_size+(-1,))
            # mode_num_instance = inst_mode_num_stat.mode(-1)[0]

            return mode_num_instance

        elif mode == 'sematic':
            # mode_num_stat = F.interpolate(target['label'].unsqueeze(1).float(), (h_match_size, 
            #     w_match_size), mode='nearest').squeeze(1).long().view((N, down_size[0], 
            #     h_ratio, down_size[1], w_ratio)).permute(0,1,3,2,4).contiguous().view((N,
            #     )+down_size+(-1,))
            # mode_num_sematic = mode_num_stat.mode(-1)[0]
            # import pdb; pdb.set_trace()
            mode_num_sematic = mode_num_instance.clone()
            for bid in range(len(mode_num_sematic)):
                for inst_num in mode_num_instance[bid].unique():
                    mode_num_sematic[bid][mode_num_instance[bid] == inst_num] = target['label']\
                        [bid][target['instance'][bid] == inst_num].unique().item()

            return mode_num_sematic


        else:
            import pdb; pdb.set_trace()
        return 0

    def prepare_targets(self, target, down_size, mode='sematic', pyramids=[]):

        N, img_h, img_w = target['label'].shape
        h_ratio = int(np.ceil(img_h / down_size[0]))
        w_ratio = int(np.ceil(img_w / down_size[1]))
        h_match_size = h_ratio * down_size[0]
        w_match_size = w_ratio * down_size[1]

        inst_mode_num_stat = F.interpolate(target['instance'].unsqueeze(1).float(), (h_match_size, 
            w_match_size), mode='nearest').squeeze(1).long().view((N, down_size[0], 
            h_ratio, down_size[1], w_ratio)).permute(0,1,3,2,4).contiguous().view((N,
            )+down_size+(-1,))
        mode_num_instance = inst_mode_num_stat.mode(-1)[0]

        if mode == 'instance':
            return mode_num_instance

        elif mode == 'sematic':
            mode_num_sematic = mode_num_instance.clone()
            for bid in range(len(mode_num_sematic)):
                for inst_num in mode_num_instance[bid].unique():
                    mode_num_sematic[bid][mode_num_instance[bid] == inst_num] = target['label']\
                        [bid][target['instance'][bid] == inst_num].unique().item()
            # import pdb; pdb.set_trace()
            return mode_num_sematic
        else:
            import pdb; pdb.set_trace()
        return 0

    def compute_sematic(self, feature, target):
        # import pdb; pdb.set_trace()
        x_sem_size5 = self.sem_size5(feature[5])
        up_6_5 = F.interpolate(feature[5], feature[4].shape[2:], mode='bilinear', align_corners=True)
        sematic_5 = self.sematic_conv_5(torch.cat((feature[4], up_6_5), dim=1))
        x_sem_size9 = self.sem_size9(sematic_5)
        up_5_4 = F.interpolate(sematic_5, feature[3].shape[2:], mode='bilinear', align_corners=True)
        sematic_4 = self.sematic_conv_4(torch.cat((feature[3], up_5_4), dim=1))
        x_sem_size17 = self.sem_size17(sematic_4)
        up_4_3 = F.interpolate(sematic_4, feature[2].shape[2:], mode='bilinear', align_corners=True)
        sematic_3 = self.sematic_conv_3(torch.cat((feature[2], up_4_3), dim=1))
        x_sem_size33 = self.sem_size33(sematic_3)
        up_3_2 = F.interpolate(sematic_3, feature[1].shape[2:], mode='bilinear', align_corners=True)
        sematic_2 = self.sematic_conv_2(torch.cat((feature[1], up_3_2), dim=1))
        x_sem_size65 = self.sem_size65(sematic_2)
        up_2_1 = F.interpolate(sematic_2, feature[0].shape[2:], mode='bilinear', align_corners=True)
        sematic_1 = self.sematic_conv_1(torch.cat((feature[0], up_2_1), dim=1))
        x_sem_size129 = self.sem_size129(sematic_1)
        up_final = F.interpolate(sematic_1, self.cfg.DATALOADER.CROP_SIZE, mode='bilinear', align_corners=True)
        x_sem_size513 = self.sem_size513(up_final)

        # import pdb; pdb.set_trace()
        sematic_targets_5 = self.prepare_targets(target, mode='sematic', down_size=(5,5))
        sematic_targets_9 = self.prepare_targets(target, mode='sematic', down_size=(9,9))
        sematic_targets_17 = self.prepare_targets(target, mode='sematic', down_size=(17,17))
        sematic_targets_33 = self.prepare_targets(target, mode='sematic', down_size=(33,33))
        sematic_targets_65 = self.prepare_targets(target, mode='sematic', down_size=(65,65))
        sematic_targets_129 = self.prepare_targets(target, mode='sematic', down_size=(129,129))


        sematic_loss_5 = self.sematic_criteron(x_sem_size5, sematic_targets_5)
        sematic_loss_9 = self.sematic_criteron(x_sem_size9, sematic_targets_9)
        sematic_loss_17 = self.sematic_criteron(x_sem_size17, sematic_targets_17)
        sematic_loss_33 = self.sematic_criteron(x_sem_size33, sematic_targets_33)
        sematic_loss_65 = self.sematic_criteron(x_sem_size65, sematic_targets_65)
        sematic_loss_129 = self.sematic_criteron(x_sem_size129, sematic_targets_129)
        sematic_loss_513 = self.sematic_criteron(x_sem_size513, target['label'].long())

        # import pdb; pdb.set_trace()
        sematic_loss = torch.tensor([sematic_loss_5, sematic_loss_9, sematic_loss_17, sematic_loss_33, 
            sematic_loss_65, sematic_loss_129, sematic_loss_513], requires_grad=True)
        # sematic_loss = torch.tensor([sematic_loss_5, sematic_loss_9, sematic_loss_17, sematic_loss_33, 
        #     sematic_loss_65, sematic_loss_129, sematic_loss_513])

        return sematic_loss, [x_sem_size5, x_sem_size9, x_sem_size17, x_sem_size33, x_sem_size65, 
            x_sem_size129, x_sem_size513]

    def forward_singel_level(self, curr_level, inst_pyramids, x_curr, i, level_sizes, 
        target_support_pyramids, target_levels, losses_i):
        new_pos_limit = [100, 50, 50, 50, 50, 50, 50]
        new_pos_quota = 80
        if x_curr[[i]].abs().max() > 1e20 or torch.isnan(x_curr[[i]].max()):
        # if torch.isnan(x_curr[[i]].max()):
            print(curr_level, '\n', x_curr[[i]])
            import pdb; pdb.set_trace()
        # 生成 upsample mask，对现有的mask pyramids
        self.compute_mask(curr_level, x_curr[[i]], inst_pyramids)
        # TODO: 考虑其他的new_masks计算方法，比如说 multi target cross entropy loss 中的单一channel
        new_masks_minus = torch.cat([i_p.get_mask(curr_level)[:,[1]] - i_p.get_mask(curr_level)[:,[0]] for i_p in inst_pyramids], dim=1)
        new_masks_softmax = F.softmax(new_masks_minus,dim=1)
        # avg_sharing = 1.0 / len(inst_pyramids)
        # num_pixels = int(new_masks_softmax.shape[-1]*new_masks_softmax.shape[-2])
        # top_percent = new_masks_softmax.view(-1).topk(int(num_pixels*(1-0.3)))[0][-1].item()
        # max_topk = new_masks_softmax.max(dim=1)[0].view(-1).topk(num_pixels-3)[0][-1].item()
        max_topk = new_masks_softmax.max(dim=1)[0].view(-1).topk(8, largest=False)[0][-1].item()
        # 这里非常的有趣，保证最少选拔8人，如果KOL话语权占不到5%，那就诞生新的KOL proposal
        # pending_thresh越高，新增的new_pos越多 所以 max_topk 应该是保底， 应该配合比例
        pending_thresh = max(0.02, max_topk)
        new_pos = torch.nonzero(new_masks_softmax[0].max(dim=0)[0] < pending_thresh)
        # if len(new_pos) > new_pos_limit[curr_level]:
        #     # import pdb; pdb.set_trace()
        #     raw_pos = new_masks_softmax.max(dim=1)[0].view(-1).topk(new_pos_limit[curr_level], largest=False)[1]
        #     new_pos_0 = raw_pos // x_curr.shape[-1]
        #     new_pos_1 = raw_pos % x_curr.shape[-1]
        #     new_pos = torch.cat((new_pos_0.view(-1,1), new_pos_1.view(-1,1)), dim=1)
        # import pdb; pdb.set_trace()
        if len(inst_pyramids) + len(new_pos) > new_pos_quota:
            available_number = max(0, new_pos_quota - len(inst_pyramids))
            if available_number:
                raw_pos = new_masks_softmax.max(dim=1)[0].view(-1).topk(available_number, largest=False)[1]
                new_pos_0 = raw_pos // x_curr.shape[-1]
                new_pos_1 = raw_pos % x_curr.shape[-1]
                new_pos = torch.cat((new_pos_0.view(-1,1), new_pos_1.view(-1,1)), dim=1)
            else:
                new_pos = []

        new_occupy = 1.0*len(new_pos) / x_curr.shape[-2] / x_curr.shape[-1]
        # if new_occupy > 0.5 or len(new_pos) <8:
        # if new_occupy > 0.5 or len(new_pos) <8-1:
        #     print('new_occupy:{}| len(new_pos):{}'.format(new_occupy, len(new_pos)))
            # import pdb; pdb.set_trace()
        new_pyramids = [InstancePyramid(pos, curr_level, level_sizes) for pos in new_pos]
        self.compute_mask(curr_level, x_curr[[i]], new_pyramids, True)
        # 出清没有领地的pyramid 在所有pixel都进不了前3
        # 统计没有pyramid的targets
        # 额外惩罚霸占位置的pyramid，保护弱势应得的 pyramid
        merit_pyramids_idx = new_masks_softmax.topk(2, dim=1)[1].unique()
        # merit_pyramids_idx = new_masks_softmax.topk(3, dim=1)[1].unique()
        merit_pyramids = [inst_pyramids[i] for i in range(len(inst_pyramids)) if i in merit_pyramids_idx]

        if self.training:
            target_len_before = sum([len(l) for l in target_support_pyramids])
            # target_len_1_before = sum([len(l) for l in target_support_pyramids_0])
            # import pdb; pdb.set_trace()
            for reduce_i in range(len(inst_pyramids)):
                if reduce_i not in merit_pyramids_idx:
                    die_id = inst_pyramids[reduce_i].idx
                    die_target_idx = inst_pyramids[reduce_i].target_idx
                    if die_target_idx:
                        target_support_pyramids[die_target_idx].remove(die_id)
                        # target_support_pyramids_0[die_target_idx].remove(die_id)
            target_len_after = sum([len(l) for l in target_support_pyramids])
            # target_len_1_after = sum([len(l) for l in target_support_pyramids_0])
            # if target_len_1_before != target_len_1_after:
            #     import pdb; pdb.set_trace()

        # import pdb; pdb.set_trace()
        inst_pyramids = merit_pyramids + new_pyramids
        # self.log_dict.update({'pyr_num_l1': len(inst_pyramids)})
        self.log_dict.update({'pyr_num_l'+str(curr_level): len(inst_pyramids)})
        if self.training:
            self.match_target(curr_level, new_pyramids, target_levels, target_support_pyramids)
            loss = self.compute_loss(curr_level, inst_pyramids, target_levels, target_support_pyramids)
            losses_i.append(loss)
        # import pdb; pdb.set_trace()
        
        return inst_pyramids

    def forward(self, image, targets=None):
        output_dict = dict()
        N, _, img_h, img_w = image.shape
        x1 = self.inc(image)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # x6 = self.asppconv(self.aspp(x5))
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.down7(x7)

        down_features = [x1, x2, x3, x4, x5, x6, x7, x8]
        x_sematic = self.up1(x8, x7)
        x_sematic = self.up2(x_sematic, x6)
        x_sematic = self.up3(x_sematic, x5)
        x_sematic = self.up4(x_sematic, x4)
        x_sematic = self.up5(x_sematic, x3)
        x_sematic = self.up6(x_sematic, x2)
        x_sematic = self.up7(x_sematic, x1)
        sematic_out = self._sematic_outc(x_sematic)

        sematic_loss = self.sematic_criteron(sematic_out, targets['label'].long())



        # xs_r50 = self.resnet50(image)
        # sematic_loss, sematic_out = self.compute_sematic(xs_r50, targets)
        if self.cfg.SOLVER.SEMATIC_ONLY:
            output_dict['loss_dict'] = {'sematic': sematic_loss.mean(), 'instance': 0}
            output_dict['sematic_out'] = sematic_out
            output_dict['targets'] = targets
            output_dict['ins_pyramids'] =[[]]
            return output_dict

        instance_categories = [11, 12, 13, 14, 15, 16, 17, 18]
        lvl_sizes = [tuple(f.shape[-2:]) for f in down_features[::-1]]

        # 调换顺序，找最大值处后， 先分配target， target指定了cat
        pyr_id_own_ins_target = [{} for _ in range(N)]
        pyr_losses = [[] for _ in range(N)]
        ins_pyramids = [[] for _ in range(N)]
        
        for bid in range(N):
            # import pdb; pdb.set_trace()
            embeded_target = []
            if self.training:
                for sem_lvl in range(5):
                # for sem_lvl in range(7):
                    lvl_size = lvl_sizes[sem_lvl]
                    # import pdb; pdb.set_trace()
                    # TODO: 简化其中的程序
                    inst_target = self.prepare_targets(targets, mode='instance', down_size=lvl_size)[bid]
                    sematic_target = self.prepare_targets(targets, mode='sematic', down_size=lvl_size)[bid]
                    sematic_lvl_resize = F.interpolate(sematic_out, lvl_size, mode='bilinear', align_corners=True)[bid]
                    for ins_tar_id in inst_target.unique():
                        target_cat = sematic_target[inst_target == ins_tar_id].unique().item()
                        if target_cat not in instance_categories:
                            if target_cat != -1:
                                embeded_target.append(inst_target == ins_tar_id)
                            continue
                        else:
                            # import pdb; pdb.set_trace()
                            if ins_tar_id.item() in pyr_id_own_ins_target[bid].keys():
                                # import pdb; pdb.set_trace()
                                # pyr = [pyr for pyr in ins_pyramids[bid] if pyr.idx == \
                                #     pyr_id_own_ins_target[bid][ins_tar_id.item()]][0]
                                # pyr.set_ins_target(sem_lvl, inst_target == ins_tar_id)
                                pass
                            else:
                                # if len([flatten_list(ins_pyramids)]) > 80:
                                if len(ins_pyramids[bid]) > 20:
                                    continue
                                cat_logit = sematic_lvl_resize[target_cat]
                                # logit_within_instar = cat_logit * (inst_target == ins_tar_id).float()
                                logit_within_instar = cat_logit.clone()
                                logit_within_instar[inst_target != ins_tar_id] = cat_logit.min()
                                max_poses = torch.nonzero(logit_within_instar == logit_within_instar.max())
                                new_pyr = InstancePyramid(max_poses[0], sem_lvl, lvl_sizes)
                                new_pyr.target_idx = ins_tar_id
                                new_pyr.tar_cat = target_cat
                                # new_pyr.set_ins_target(513, inst_target == ins_tar_id)
                                ins_pyramids[bid].append(new_pyr)
                                pyr_id_own_ins_target[bid][ins_tar_id.item()] = new_pyr.idx
            else:
                init_pos = torch.nonzero(torch.ones_like(down_features[-1][0][0]))

            mask_mass = sematic_out[[bid], :11]
            if len(ins_pyramids[bid]) == 0:
                # print('WWWWWWWWWWWWWarning! num is short', targets['label'][bid].unique())
                # import pdb; pdb.set_trace()
                # instance_loss = image.new_zeros(1)
                instance_loss = image.sum()* 0
            else:
                mask_ins = self.compute_mask_v10(feature=down_features, pyramids=ins_pyramids, bid=bid)
                mask_sem_ins = torch.cat((mask_mass, mask_ins), dim=1)
                sema_inst_target_11 = torch.zeros_like(down_features[0][bid, :11]).bool()
                for i in range(11):
                    sema_inst_target_11[i][targets['label'][bid] == i] = True
                # import pdb; pdb.set_trace()
                inst_target_masks = [sema_inst_target_11]
                for i, pyr in enumerate(ins_pyramids[bid]):
                    inst_target_masks.append((targets['instance'][bid] == pyr.target_idx)[None])
                # import pdb; pdb.set_trace()
                inst_target_idx = torch.cat(inst_target_masks).max(0)[1][None]
                instance_loss = self.sematic_criteron(mask_sem_ins, inst_target_idx)
            # import pdb; pdb.set_trace()
            pyr_losses[bid].append(instance_loss)
        self.log_dict.update({'InstPyr_inst_count': InstancePyramid.inst_count})
        # import pdb; pdb.set_trace()
        pyr_losses = list(flatten_list(pyr_losses))
        # instance_loss = torch.tensor(pyr_losses, requires_grad=True)

        output_dict['loss_dict'] = {'sematic': sematic_loss.mean(), 'instance': sum(pyr_losses)/len(pyr_losses)}
        # output_dict['loss_dict'] = {'sematic': sematic_loss.mean(), 'instance': pyr_losses.mean()}
        output_dict['sematic_out'] = sematic_out
        output_dict['targets'] = targets
        output_dict['ins_pyramids'] =ins_pyramids
        # output_dict['in_dict_watches'] =in_dict_watches

        return output_dict

class MaskPyramids(nn.Module):
    def __init__(self, cfg):
        super(MaskPyramids, self).__init__()
        self.cfg = cfg
        num_classes = cfg.DATALOADER.NUM_CLASSES

        self.inc = inconv(3, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 256)
        self.down4 = down(256, 256)
        self.down5 = down(256, 256)
        self.down6 = down(256, 256)
        self.down7 = down(256, 256)
        self.aspp = ASPP(inplanes=256)
        # self.asppconv = nn.Conv2d(256, 512, 1)
        self.up1 = up(512, 256)
        self.up2 = up(512, 256)
        self.up3 = up(512, 256)
        self.up4 = up(512, 256)
        self.up5 = up(512, 128)
        self.up6 = up(256, 64)
        self.up7 = up(128, 64)
        # self.up8 = up(128+64, 64)
        self.outc = outconv(64, num_classes)
        self._init_weight()

        # self.inc = inconv(3, 64)
        # self.down1 = down(64, 128)
        # self.down2 = down(128, 256)
        # self.down3 = down(256, 512)
        # self.down4 = down(512, 512)
        # self.aspp = ASPP()
        # # self.asppconv = nn.Conv2d(256, 512, 1)
        # self.up1 = up(1024, 256)
        # self.up2 = up(512, 128)
        # self.up3 = up(256, 64)
        # self.up4 = up(128, 64)
        # self.outc = outconv(64, num_classes)
        # self._init_weight()

        # self.inc = inconv(3, 64)
        # self.down1 = down(64, 128)
        # self.down2 = down(128, 256)
        # self.down3 = down(256, 256)
        # self.down4 = down(256, 256)
        # self.down5 = down(256, 256)
        # self.down6 = down(256, 256)
        # self.down7 = down(256, 256)
        # # self.aspp = ASPP('drn', output_stride=16, BatchNorm=nn.BatchNorm2d)
        # # self.asppconv = nn.Conv2d(256, 512, 1)
        # self.up1 = up(512, 256)
        # self.up2 = up(512, 256)
        # self.up3 = up(512, 256)
        # self.up4 = up(512, 256)
        # self.up5 = up(512, 256)
        # self.up6 = up(256+128, 256)
        # self.up7 = up(256+64, 256)
        # # self.up7 = up(512, 256)
        self._sematic_outc = outconv(256, num_classes)

        self.ins_cat_pyr_convs = [None for _ in range(11)]
        for i in range(8):
            self.ins_cat_pyr_convs.append(
                CategoryPyramid(cfg)
            )
        self.ins_cat_pyr_convs = nn.ModuleList(self.ins_cat_pyr_convs)

        self.cs_criteron = nn.CrossEntropyLoss()
        self.class_criteron = nn.CrossEntropyLoss()
        self.sematic_criteron = nn.CrossEntropyLoss(ignore_index=cfg.DATALOADER.IGNORE_INDEX)

        self.cs_loss_factor = 1.0
        self.miss_loss_factor = 1.0

        self.log_dict = {}
        

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def compute_mask(self, level, feature, pyramids):
        masks = []
        for pyr in pyramids:
            import pdb; pdb.set_trace()
            pyr_conv = self.ins_cat_pyr_convs[pyr.tar_cat].convs[level - pyr.init_level]
            out_conv = self.ins_cat_pyr_convs[pyr.tar_cat].outs[level - pyr.init_level]
            if pyr.init_level == level:
                indicate_mask = pyr.get_indicate_mask(feature)[None]
                conv_input = torch.cat((indicate_mask, feature), dim=1)
                mask_logit = pyr_conv(conv_input)
                pyr.set_mask_logits(level, mask_logit)
                mask = out_conv(mask_logit)
            elif level == 6:
                # import pdb; pdb.set_trace()
                last_logits = pyr.get_mask_logits(level-1)
                up_last_logits = F.interpolate(last_logits, feature.shape[2:], mode='bilinear', align_corners=True)
                mask = out_conv(up_last_logits)
            else:
                # import pdb; pdb.set_trace()
                last_logits = pyr.get_mask_logits(level-1)
                up_last_logits = F.interpolate(last_logits, feature.shape[2:], mode='bilinear', align_corners=True)
                conv_input = torch.cat((feature, up_last_logits), dim=1)
                mask_logit = pyr_conv(conv_input)
                pyr.set_mask_logits(level, mask_logit)
                mask = out_conv(mask_logit)
            pyr.set_mask(level, mask)
            masks.append(mask)
        # import pdb; pdb.set_trace()
        masks = torch.cat(masks, dim=1)

        return masks

    def compute_mask_v10(self, feature, pyramids, bid):
        masks = []
        for pyr in pyramids[bid]:
            # import pdb; pdb.set_trace()
            pyr_conv = self.ins_cat_pyr_convs[pyr.tar_cat]
            (x1, x2, x3, x4, x5, x6, x7, x8) = [f[[bid]] for f in feature]
            pyr_features = feature[::-1][pyr.init_level:]
            # print([f.shape for f in pyr_features])
            indicate_mask = pyr.get_indicate_mask(pyr_features[0])[None]
            init_input = torch.cat((indicate_mask, pyr_features[0]), dim=1)
            x_up = pyr_conv.init_conv(init_input)
            # import pdb; pdb.set_trace()
            for up_lvl in range(1, 6- pyr.init_level):
                # print('x{}, x_up.shape:{}'.format(9- pyr.init_level-up_lvl, x_up.shape))
                # import pdb; pdb.set_trace()
                x_up = pyr_conv.up_list[up_lvl](x_up, pyr_features[up_lvl])
            x_up_256 = pyr_conv.up256(x_up, x2)
            x_up_513 = pyr_conv.up513(x_up_256, x1)
            pyr_mask = pyr_conv.outc(x_up_513)
            masks.append(pyr_mask)
            pyr.set_mask(513, pyr_mask)
        masks = torch.cat(masks, dim=1)

        return masks

    def compute_loss(self, level, pyramids, target_levels, target_support_pyramids):
        # TODO: multi target cross entropy loss
        losses = []
        miss_losses = []
        # class_losses = []
        # covered_idx = []
        for pyramid in pyramids:
            mask_logits = pyramid.get_mask(level)
            if pyramid.target_idx is not None:
                target_mask = target_levels[7-level][0, [pyramid.target_idx]]
                loss_cs = self.cs_criteron(mask_logits, target_mask.squeeze(1).long())
                losses.append(loss_cs)

                # if loss_cs > 1e20 or torch.isnan(loss_cs):
                if torch.isnan(loss_cs):
                    import pdb; pdb.set_trace()

                # import pdb; pdb.set_trace()
                # if level == 3:
                #     import pdb; pdb.set_trace()
                #     target_label = target_levels['labels'][[pyramid.target_idx]]
                #     loss_class = self.class_criteron(pyramid.class_logits, target_label)
                #     class_losses.append(loss_class)

        # import pdb; pdb.set_trace()
        # TODO: 检查未被追踪的target_idx
        # TODO: 避免惩罚该target 指导的pyramid， 不存在这个问题。。。
        for i, t_match_list in  enumerate(target_support_pyramids):
            if not t_match_list:    # target 没有match到pyramid
                miss_target_map = target_levels[7-level][0, i]
                if miss_target_map.sum():
                    miss_pos = miss_target_map.nonzero()
                    # import pdb; pdb.set_trace()
                    all_masks = torch.cat([i_p.get_mask(level) for i_p in pyramids], dim=1)
                    loss_miss = all_masks[0,:,miss_pos[:,0], miss_pos[:,1]].mean()

        cs_loss = sum(loss for loss in losses)/len(losses)\
            if len(losses) else mask_logits.sum()*0
        miss_loss = sum(loss for loss in miss_losses) / len(miss_losses)\
            if len(miss_losses) else mask_logits.sum()*0
        # class_loss = sum(loss for loss in class_losses) / len(class_losses)\
        #     if len(class_losses) else mask_logits.sum()*0

        # resloss = cs_loss * self.cs_loss_factor + miss_loss * self.miss_loss_factor + \
        #     class_loss * self.class_loss_factor

        resloss = cs_loss * self.cs_loss_factor + miss_loss * self.miss_loss_factor
                    
        # resloss = (sum(loss for loss in losses)/len(losses)\
        #     if len(losses) else mask_logits.sum()*0) * self.cs_loss_factor + \
        #     (sum(loss for loss in miss_losses) / len(miss_losses)\
        #     if len(miss_losses) else mask_logits.sum()*0) * self.miss_loss_factor
        return resloss

    def prepare_targets_archive(self, target, down_size, mode='sematic', pyramids=[]):

        N, img_h, img_w = target['label'].shape
        h_ratio = int(np.ceil(img_h / down_size[0]))
        w_ratio = int(np.ceil(img_w / down_size[1]))
        h_match_size = h_ratio * down_size[0]
        w_match_size = w_ratio * down_size[1]

        inst_mode_num_stat = F.interpolate(target['instance'].unsqueeze(1).float(), (h_match_size, 
            w_match_size), mode='nearest').squeeze(1).long().view((N, down_size[0], 
            h_ratio, down_size[1], w_ratio)).permute(0,1,3,2,4).contiguous().view((N,
            )+down_size+(-1,))
        mode_num_instance = inst_mode_num_stat.mode(-1)[0]

        if mode == 'instance':
            # # import pdb; pdb.set_trace()
            # inst_mode_num_stat = F.interpolate(target['instance'].unsqueeze(1).float(), (h_match_size, 
            #     w_match_size), mode='nearest').squeeze(1).long().view((N, down_size[0], 
            #     h_ratio, down_size[1], w_ratio)).permute(0,1,3,2,4).contiguous().view((N,
            #     )+down_size+(-1,))
            # mode_num_instance = inst_mode_num_stat.mode(-1)[0]

            return mode_num_instance

        elif mode == 'sematic':
            # mode_num_stat = F.interpolate(target['label'].unsqueeze(1).float(), (h_match_size, 
            #     w_match_size), mode='nearest').squeeze(1).long().view((N, down_size[0], 
            #     h_ratio, down_size[1], w_ratio)).permute(0,1,3,2,4).contiguous().view((N,
            #     )+down_size+(-1,))
            # mode_num_sematic = mode_num_stat.mode(-1)[0]
            # import pdb; pdb.set_trace()
            mode_num_sematic = mode_num_instance.clone()
            for bid in range(len(mode_num_sematic)):
                for inst_num in mode_num_instance[bid].unique():
                    mode_num_sematic[bid][mode_num_instance[bid] == inst_num] = target['label']\
                        [bid][target['instance'][bid] == inst_num].unique().item()

            return mode_num_sematic


        else:
            import pdb; pdb.set_trace()
        return 0

    def prepare_targets(self, target, down_size, mode='sematic', pyramids=[]):

        N, img_h, img_w = target['label'].shape
        h_ratio = int(np.ceil(img_h / down_size[0]))
        w_ratio = int(np.ceil(img_w / down_size[1]))
        h_match_size = h_ratio * down_size[0]
        w_match_size = w_ratio * down_size[1]

        inst_mode_num_stat = F.interpolate(target['instance'].unsqueeze(1).float(), (h_match_size, 
            w_match_size), mode='nearest').squeeze(1).long().view((N, down_size[0], 
            h_ratio, down_size[1], w_ratio)).permute(0,1,3,2,4).contiguous().view((N,
            )+down_size+(-1,))
        mode_num_instance = inst_mode_num_stat.mode(-1)[0]

        if mode == 'instance':
            return mode_num_instance

        elif mode == 'sematic':
            mode_num_sematic = mode_num_instance.clone()
            for bid in range(len(mode_num_sematic)):
                for inst_num in mode_num_instance[bid].unique():
                    mode_num_sematic[bid][mode_num_instance[bid] == inst_num] = target['label']\
                        [bid][target['instance'][bid] == inst_num].unique().item()
            # import pdb; pdb.set_trace()
            return mode_num_sematic
        else:
            import pdb; pdb.set_trace()
        return 0

    def compute_sematic(self, feature, target):
        # import pdb; pdb.set_trace()
        x_sem_size5 = self.sem_size5(feature[5])
        up_6_5 = F.interpolate(feature[5], feature[4].shape[2:], mode='bilinear', align_corners=True)
        sematic_5 = self.sematic_conv_5(torch.cat((feature[4], up_6_5), dim=1))
        x_sem_size9 = self.sem_size9(sematic_5)
        up_5_4 = F.interpolate(sematic_5, feature[3].shape[2:], mode='bilinear', align_corners=True)
        sematic_4 = self.sematic_conv_4(torch.cat((feature[3], up_5_4), dim=1))
        x_sem_size17 = self.sem_size17(sematic_4)
        up_4_3 = F.interpolate(sematic_4, feature[2].shape[2:], mode='bilinear', align_corners=True)
        sematic_3 = self.sematic_conv_3(torch.cat((feature[2], up_4_3), dim=1))
        x_sem_size33 = self.sem_size33(sematic_3)
        up_3_2 = F.interpolate(sematic_3, feature[1].shape[2:], mode='bilinear', align_corners=True)
        sematic_2 = self.sematic_conv_2(torch.cat((feature[1], up_3_2), dim=1))
        x_sem_size65 = self.sem_size65(sematic_2)
        up_2_1 = F.interpolate(sematic_2, feature[0].shape[2:], mode='bilinear', align_corners=True)
        sematic_1 = self.sematic_conv_1(torch.cat((feature[0], up_2_1), dim=1))
        x_sem_size129 = self.sem_size129(sematic_1)
        up_final = F.interpolate(sematic_1, self.cfg.DATALOADER.CROP_SIZE, mode='bilinear', align_corners=True)
        x_sem_size513 = self.sem_size513(up_final)

        # import pdb; pdb.set_trace()
        sematic_targets_5 = self.prepare_targets(target, mode='sematic', down_size=(5,5))
        sematic_targets_9 = self.prepare_targets(target, mode='sematic', down_size=(9,9))
        sematic_targets_17 = self.prepare_targets(target, mode='sematic', down_size=(17,17))
        sematic_targets_33 = self.prepare_targets(target, mode='sematic', down_size=(33,33))
        sematic_targets_65 = self.prepare_targets(target, mode='sematic', down_size=(65,65))
        sematic_targets_129 = self.prepare_targets(target, mode='sematic', down_size=(129,129))


        sematic_loss_5 = self.sematic_criteron(x_sem_size5, sematic_targets_5)
        sematic_loss_9 = self.sematic_criteron(x_sem_size9, sematic_targets_9)
        sematic_loss_17 = self.sematic_criteron(x_sem_size17, sematic_targets_17)
        sematic_loss_33 = self.sematic_criteron(x_sem_size33, sematic_targets_33)
        sematic_loss_65 = self.sematic_criteron(x_sem_size65, sematic_targets_65)
        sematic_loss_129 = self.sematic_criteron(x_sem_size129, sematic_targets_129)
        sematic_loss_513 = self.sematic_criteron(x_sem_size513, target['label'].long())

        # import pdb; pdb.set_trace()
        sematic_loss = torch.tensor([sematic_loss_5, sematic_loss_9, sematic_loss_17, sematic_loss_33, 
            sematic_loss_65, sematic_loss_129, sematic_loss_513], requires_grad=True)
        # sematic_loss = torch.tensor([sematic_loss_5, sematic_loss_9, sematic_loss_17, sematic_loss_33, 
        #     sematic_loss_65, sematic_loss_129, sematic_loss_513])

        return sematic_loss, [x_sem_size5, x_sem_size9, x_sem_size17, x_sem_size33, x_sem_size65, 
            x_sem_size129, x_sem_size513]

    def forward_singel_level(self, curr_level, inst_pyramids, x_curr, i, level_sizes, 
        target_support_pyramids, target_levels, losses_i):
        new_pos_limit = [100, 50, 50, 50, 50, 50, 50]
        new_pos_quota = 80
        if x_curr[[i]].abs().max() > 1e20 or torch.isnan(x_curr[[i]].max()):
        # if torch.isnan(x_curr[[i]].max()):
            print(curr_level, '\n', x_curr[[i]])
            import pdb; pdb.set_trace()
        # 生成 upsample mask，对现有的mask pyramids
        self.compute_mask(curr_level, x_curr[[i]], inst_pyramids)
        # TODO: 考虑其他的new_masks计算方法，比如说 multi target cross entropy loss 中的单一channel
        new_masks_minus = torch.cat([i_p.get_mask(curr_level)[:,[1]] - i_p.get_mask(curr_level)[:,[0]] for i_p in inst_pyramids], dim=1)
        new_masks_softmax = F.softmax(new_masks_minus,dim=1)
        # avg_sharing = 1.0 / len(inst_pyramids)
        # num_pixels = int(new_masks_softmax.shape[-1]*new_masks_softmax.shape[-2])
        # top_percent = new_masks_softmax.view(-1).topk(int(num_pixels*(1-0.3)))[0][-1].item()
        # max_topk = new_masks_softmax.max(dim=1)[0].view(-1).topk(num_pixels-3)[0][-1].item()
        max_topk = new_masks_softmax.max(dim=1)[0].view(-1).topk(8, largest=False)[0][-1].item()
        # 这里非常的有趣，保证最少选拔8人，如果KOL话语权占不到5%，那就诞生新的KOL proposal
        # pending_thresh越高，新增的new_pos越多 所以 max_topk 应该是保底， 应该配合比例
        pending_thresh = max(0.02, max_topk)
        new_pos = torch.nonzero(new_masks_softmax[0].max(dim=0)[0] < pending_thresh)
        # if len(new_pos) > new_pos_limit[curr_level]:
        #     # import pdb; pdb.set_trace()
        #     raw_pos = new_masks_softmax.max(dim=1)[0].view(-1).topk(new_pos_limit[curr_level], largest=False)[1]
        #     new_pos_0 = raw_pos // x_curr.shape[-1]
        #     new_pos_1 = raw_pos % x_curr.shape[-1]
        #     new_pos = torch.cat((new_pos_0.view(-1,1), new_pos_1.view(-1,1)), dim=1)
        # import pdb; pdb.set_trace()
        if len(inst_pyramids) + len(new_pos) > new_pos_quota:
            available_number = max(0, new_pos_quota - len(inst_pyramids))
            if available_number:
                raw_pos = new_masks_softmax.max(dim=1)[0].view(-1).topk(available_number, largest=False)[1]
                new_pos_0 = raw_pos // x_curr.shape[-1]
                new_pos_1 = raw_pos % x_curr.shape[-1]
                new_pos = torch.cat((new_pos_0.view(-1,1), new_pos_1.view(-1,1)), dim=1)
            else:
                new_pos = []

        new_occupy = 1.0*len(new_pos) / x_curr.shape[-2] / x_curr.shape[-1]
        # if new_occupy > 0.5 or len(new_pos) <8:
        # if new_occupy > 0.5 or len(new_pos) <8-1:
        #     print('new_occupy:{}| len(new_pos):{}'.format(new_occupy, len(new_pos)))
            # import pdb; pdb.set_trace()
        new_pyramids = [InstancePyramid(pos, curr_level, level_sizes) for pos in new_pos]
        self.compute_mask(curr_level, x_curr[[i]], new_pyramids, True)
        # 出清没有领地的pyramid 在所有pixel都进不了前3
        # 统计没有pyramid的targets
        # 额外惩罚霸占位置的pyramid，保护弱势应得的 pyramid
        merit_pyramids_idx = new_masks_softmax.topk(2, dim=1)[1].unique()
        # merit_pyramids_idx = new_masks_softmax.topk(3, dim=1)[1].unique()
        merit_pyramids = [inst_pyramids[i] for i in range(len(inst_pyramids)) if i in merit_pyramids_idx]

        if self.training:
            target_len_before = sum([len(l) for l in target_support_pyramids])
            # target_len_1_before = sum([len(l) for l in target_support_pyramids_0])
            # import pdb; pdb.set_trace()
            for reduce_i in range(len(inst_pyramids)):
                if reduce_i not in merit_pyramids_idx:
                    die_id = inst_pyramids[reduce_i].idx
                    die_target_idx = inst_pyramids[reduce_i].target_idx
                    if die_target_idx:
                        target_support_pyramids[die_target_idx].remove(die_id)
                        # target_support_pyramids_0[die_target_idx].remove(die_id)
            target_len_after = sum([len(l) for l in target_support_pyramids])
            # target_len_1_after = sum([len(l) for l in target_support_pyramids_0])
            # if target_len_1_before != target_len_1_after:
            #     import pdb; pdb.set_trace()

        # import pdb; pdb.set_trace()
        inst_pyramids = merit_pyramids + new_pyramids
        # self.log_dict.update({'pyr_num_l1': len(inst_pyramids)})
        self.log_dict.update({'pyr_num_l'+str(curr_level): len(inst_pyramids)})
        if self.training:
            self.match_target(curr_level, new_pyramids, target_levels, target_support_pyramids)
            loss = self.compute_loss(curr_level, inst_pyramids, target_levels, target_support_pyramids)
            losses_i.append(loss)
        # import pdb; pdb.set_trace()
        
        return inst_pyramids

    def forward(self, image, targets=None):
        output_dict = dict()
        N, _, img_h, img_w = image.shape
        x1 = self.inc(image)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.down7(x7)
        # x6 = self.asppconv(self.aspp(x5))
        x9 = self.aspp(x8)
        x = self.up1(x9, x7)
        # x = self.up1(x5, x4)
        x = self.up2(x, x6)
        # import pdb; pdb.set_trace()
        x = self.up3(x, x5)
        x = self.up4(x, x4)
        x = self.up5(x, x3)
        x = self.up6(x, x2)
        x = self.up7(x, x1)
        sematic_out = self.outc(x)
        # # import pdb; pdb.set_trace()
        # x1 = self.inc(image)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # # x6 = self.asppconv(self.aspp(x5))
        # x6 = self.aspp(x5)
        # x = self.up1(x6, x4)
        # # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        # sematic_out = self.outc(x)

        # x1 = self.inc(image)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # # x6 = self.asppconv(self.aspp(x5))
        # x6 = self.down5(x5)
        # x7 = self.down6(x6)
        # x8 = self.down7(x7)

        # down_features = [x1, x2, x3, x4, x5, x6, x7, x8]
        # x_sematic = self.up1(x8, x7)
        # x_sematic = self.up2(x_sematic, x6)
        # x_sematic = self.up3(x_sematic, x5)
        # x_sematic = self.up4(x_sematic, x4)
        # x_sematic = self.up5(x_sematic, x3)
        # x_sematic = self.up6(x_sematic, x2)
        # x_sematic = self.up7(x_sematic, x1)
        # sematic_out = self._sematic_outc(x_sematic)

        sematic_loss = self.sematic_criteron(sematic_out, targets['label'].long())



        # xs_r50 = self.resnet50(image)
        # sematic_loss, sematic_out = self.compute_sematic(xs_r50, targets)
        if self.cfg.SOLVER.SEMATIC_ONLY:
            output_dict['loss_dict'] = {'sematic': sematic_loss.mean(), 'instance': 0}
            output_dict['sematic_out'] = sematic_out
            output_dict['targets'] = targets
            output_dict['ins_pyramids'] =[[]]
            return output_dict

        instance_categories = [11, 12, 13, 14, 15, 16, 17, 18]
        lvl_sizes = [tuple(f.shape[-2:]) for f in down_features[::-1]]

        # 调换顺序，找最大值处后， 先分配target， target指定了cat
        pyr_id_own_ins_target = [{} for _ in range(N)]
        pyr_losses = [[] for _ in range(N)]
        ins_pyramids = [[] for _ in range(N)]
        
        for bid in range(N):
            # import pdb; pdb.set_trace()
            embeded_target = []
            if self.training:
                for sem_lvl in range(5):
                # for sem_lvl in range(7):
                    lvl_size = lvl_sizes[sem_lvl]
                    # import pdb; pdb.set_trace()
                    # TODO: 简化其中的程序
                    inst_target = self.prepare_targets(targets, mode='instance', down_size=lvl_size)[bid]
                    sematic_target = self.prepare_targets(targets, mode='sematic', down_size=lvl_size)[bid]
                    sematic_lvl_resize = F.interpolate(sematic_out, lvl_size, mode='bilinear', align_corners=True)[bid]
                    for ins_tar_id in inst_target.unique():
                        target_cat = sematic_target[inst_target == ins_tar_id].unique().item()
                        if target_cat not in instance_categories:
                            if target_cat != -1:
                                embeded_target.append(inst_target == ins_tar_id)
                            continue
                        else:
                            # import pdb; pdb.set_trace()
                            if ins_tar_id.item() in pyr_id_own_ins_target[bid].keys():
                                # import pdb; pdb.set_trace()
                                # pyr = [pyr for pyr in ins_pyramids[bid] if pyr.idx == \
                                #     pyr_id_own_ins_target[bid][ins_tar_id.item()]][0]
                                # pyr.set_ins_target(sem_lvl, inst_target == ins_tar_id)
                                pass
                            else:
                                # if len([flatten_list(ins_pyramids)]) > 80:
                                if len(ins_pyramids[bid]) > 20:
                                    continue
                                cat_logit = sematic_lvl_resize[target_cat]
                                # logit_within_instar = cat_logit * (inst_target == ins_tar_id).float()
                                logit_within_instar = cat_logit.clone()
                                logit_within_instar[inst_target != ins_tar_id] = cat_logit.min()
                                max_poses = torch.nonzero(logit_within_instar == logit_within_instar.max())
                                new_pyr = InstancePyramid(max_poses[0], sem_lvl, lvl_sizes)
                                new_pyr.target_idx = ins_tar_id
                                new_pyr.tar_cat = target_cat
                                # new_pyr.set_ins_target(513, inst_target == ins_tar_id)
                                ins_pyramids[bid].append(new_pyr)
                                pyr_id_own_ins_target[bid][ins_tar_id.item()] = new_pyr.idx
            else:
                init_pos = torch.nonzero(torch.ones_like(down_features[-1][0][0]))

            mask_mass = sematic_out[[bid], :11]
            if len(ins_pyramids[bid]) == 0:
                # print('WWWWWWWWWWWWWarning! num is short', targets['label'][bid].unique())
                # import pdb; pdb.set_trace()
                # instance_loss = image.new_zeros(1)
                instance_loss = image.sum()* 0
            else:
                mask_ins = self.compute_mask_v10(feature=down_features, pyramids=ins_pyramids, bid=bid)
                mask_sem_ins = torch.cat((mask_mass, mask_ins), dim=1)
                sema_inst_target_11 = torch.zeros_like(down_features[0][bid, :11]).bool()
                for i in range(11):
                    sema_inst_target_11[i][targets['label'][bid] == i] = True
                # import pdb; pdb.set_trace()
                inst_target_masks = [sema_inst_target_11]
                for i, pyr in enumerate(ins_pyramids[bid]):
                    inst_target_masks.append((targets['instance'][bid] == pyr.target_idx)[None])
                # import pdb; pdb.set_trace()
                inst_target_idx = torch.cat(inst_target_masks).max(0)[1][None]
                instance_loss = self.sematic_criteron(mask_sem_ins, inst_target_idx)
            # import pdb; pdb.set_trace()
            pyr_losses[bid].append(instance_loss)
        self.log_dict.update({'InstPyr_inst_count': InstancePyramid.inst_count})
        # import pdb; pdb.set_trace()
        pyr_losses = list(flatten_list(pyr_losses))
        # instance_loss = torch.tensor(pyr_losses, requires_grad=True)

        output_dict['loss_dict'] = {'sematic': sematic_loss.mean(), 'instance': sum(pyr_losses)/len(pyr_losses)}
        # output_dict['loss_dict'] = {'sematic': sematic_loss.mean(), 'instance': pyr_losses.mean()}
        output_dict['sematic_out'] = sematic_out
        output_dict['targets'] = targets
        output_dict['ins_pyramids'] =ins_pyramids
        # output_dict['in_dict_watches'] =in_dict_watches

        return output_dict


class InstancePyramid():
    inst_count = 0
    def __init__(self, pos, init_pub_level, level_sizes):
        self.idx = InstancePyramid.inst_count
        InstancePyramid.inst_count += 1
        self.init_level = init_pub_level
        self.level_sizes = level_sizes
        self.init_size = self.level_sizes[self.init_level]
        self.pos = pos
        self.masks = {}
        self.ins_target = {}
        self.mask_logits = {}
        self.class_logits = None
        self.target_idx = None
        self.tar_cat = -1
        self.is_alive = True
        # torch.tensor(800.0/2**(7-self.init_level)).ceil().long().item()
        self.feature_scales = [7, 13, 25, 50, 100, 200]
        # self.gaussian_masks = self.generate_gaussian_masks()
        # import pdb; pdb.set_trace()
        # self.shared_gaussian_mask = self.shared_gaussian_masks()
        self.init_gaussian_mask = self.init_gaussian_masks()

    def set_mask(self, lvl, mask):
        self.masks[lvl] = mask

    def get_mask(self, lvl):
        return self.masks[lvl]

    def set_ins_target(self, lvl, ins_target):
        self.ins_target[lvl] = ins_target

    def get_ins_target(self, lvl):
        return self.ins_target[lvl]

    def set_mask_logits(self, lvl, mask_logits):
        self.mask_logits[lvl] = mask_logits

    def get_mask_logits(self, lvl):
        return self.mask_logits[lvl]

    def bind_target(self, idx):
        self.target_idx = idx

    def get_root_level_pos(self, pub_level):
        init_size = self.level_sizes[self.init_level]
        req_size = self.level_sizes[pub_level]

        h = (self.pos[0].float() / init_size[0] * req_size[0]).round().long()
        w = (self.pos[1].float() / init_size[1] * req_size[1]).round().long()

        return (h.item(), w.item())
        
    def get_rel_pos(self, pub_level):
        init_size = self.level_sizes[self.init_level]
        req_size = self.level_sizes[pub_level]

        h = ((self.pos[0].float()+0.5) / init_size[0] * req_size[0]-0.5).round().long()
        w = ((self.pos[1].float()+0.5) / init_size[1] * req_size[1]-0.5).round().long()

        return (h.item(), w.item())

    def shared_gaussian_masks(self):
        # feature_scales = [7, 13, 25, 50, 100, 200]
        xs = torch.arange(7*4)
        ys = torch.arange(7*4).view(-1,1)
        ln2 = torch.Tensor(2.0, requires_grad=False).log()
        # shared_gaussian_mask = (-4*ln2*((xs.float()-7*2+1)**2+(ys.float()-7*2+1)**2)/7**2).exp()
        shared_gaussian_mask = (-ln2*((xs.float()-7*2+1)**2+(ys.float()-7*2+1)**2)/7**2).exp()
        return shared_gaussian_mask

    def init_gaussian_masks(self):
        xs = torch.arange(7*4)
        ys = torch.arange(7*4).view(-1,1)
        # ln2 = torch.tensor(2.0, requires_grad=False).log()
        ln2 = torch.Tensor([2.0]).log()[0]
        gaussian_mask_28 = (-ln2*((xs.float()-7*2+1)**2+(ys.float()-7*2+1)**2)/7**2).exp()

        init_gaussian_mask = torch.zeros(self.level_sizes[self.init_level])

        src_x0 = max(13-self.pos[0], 0)
        src_y0 = max(13-self.pos[1], 0)
        src_x1 = min(13+self.level_sizes[self.init_level][0]-self.pos[0], 28)
        src_y1 = min(13+self.level_sizes[self.init_level][1]-self.pos[1], 28)
        res_x0 = max(0, self.pos[0]-13)    #+1?
        res_y0 = max(0, self.pos[1]-13)
        res_x1 = res_x0+src_x1-src_x0
        res_y1 = res_y0+src_y1-src_y0

        init_gaussian_mask[res_x0:res_x1, res_y0:res_y1] = gaussian_mask_28\
            [src_x0:src_x1, src_y0:src_y1]

        return init_gaussian_mask

    def get_indicate_mask(self, feature):
        indicate = torch.zeros_like(feature[0,0])
        indicate[tuple(self.pos)] = 1.0
        mask = torch.ones_like(feature[0,0]) * feature[0,0][tuple(self.pos)]
        return torch.stack((indicate, mask))

class Trainer(object):
    def __init__(self, cfg, output_dir):
        self.cfg = cfg
        self.output_dir = output_dir
        self.logger = logging.getLogger("MaskPyramid")
        self.tbSummary = TensorboardSummary(output_dir)
        self.writer = self.tbSummary.create_summary()
        self.model = MaskPyramids(cfg)
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(cfg)

        train_params = [{'params': self.model.parameters(), 'lr': cfg.SOLVER.BASE_LR}]
        self.optimizer = torch.optim.SGD(train_params, momentum=cfg.SOLVER.MOMENTUM,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY, nesterov=cfg.SOLVER.NESTEROV)

        self.evaluator = Evaluator(self.nclass)
        self.scheduler = LR_Scheduler(cfg.SOLVER.SCHEDULE_TYPE, cfg.SOLVER.BASE_LR,
            cfg.SOLVER.EPOCHES, len(self.train_loader))
        self.start_epoch = 0
        self.best_pred = 0.0
        self.meters = {'start_time': time.time(),
            'total_iters': cfg.SOLVER.EPOCHES*len(self.train_loader)}

        # log_gpu_stat(self.logger)

    def load_weights(self, path=None, subdict='model', continue_train=False):
        state_dict = torch.load(path if path else self.cfg.MODEL.WEIGHT)
        if subdict:
            weights = state_dict[subdict]
            self.model.load_state_dict(weights)
            if self.optimizer is not None and 'optimizer' in state_dict.keys():
                self.optimizer.load_state_dict(state_dict["optimizer"])
            if 'best_pred' in state_dict.keys():
                self.best_pred = state_dict["best_pred"]
            if continue_train and 'epoch' in state_dict.keys():
                self.start_epoch = state_dict["epoch"]
        else:
            self.model.load_state_dict(state_dict)

    def mask_inst_img(self, images_np, sematic_a_image_np, output, idx, level=-1, label_mode=1, insts_a_image=None):
        pyramids = output['all_pyramids'][idx]
        inst_target = None
        if level>=0:
            import pdb; pdb.set_trace()
        bg_and_masks = [F.interpolate(pyramids[0].get_mask(3)[:,[0]], 513, mode='bilinear', align_corners=True)]
        pred_labels = [0]
        sematic_out = output['sematic_out'][idx]
        for pyr in pyramids:
            # import pdb; pdb.set_trace()
            pos = tuple(pyr.pos.tolist())
            largest_mask = pyr.get_mask(3)
            mask_513 = F.interpolate(largest_mask[:,[1]], 513, mode='bilinear', align_corners=True)
            # mask_513_map = mask_513[0].max(0)[1]
            bg_and_masks.append(mask_513)
            # prepare label
            scaled_sematic_out = F.interpolate(sematic_out[None], pyr.init_size, mode='bilinear', align_corners=True)
            label = scaled_sematic_out[0].max(0)[1][pos[0], pos[1]].item()
            pred_labels.append(label)

        # import pdb; pdb.set_trace()
        bg_and_masks_np = torch.cat(bg_and_masks).squeeze(1).max(0)[1].detach().cpu().numpy()
        inst_output = self.rend_on_image(images_np, bg_and_masks_np, pred_labels)



        return  inst_output, inst_target

    def rend_on_image_v8(self, image_np, masks_np, labels_list):
        # label 从0开始，0 表示unlabeled
        alpha = 0.5
        color_bias = 1
        # import pdb; pdb.set_trace()
        # print('masks_np unique',np.unique(masks_np), 'labels_list', labels_list)
        class_count = [0 for _ in range(20)]
        colors = np.array([
            # [0, 100, 0],
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [0, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32]])
        # import pdb; pdb.set_trace()
        masked = image_np.copy()
        # import pdb; pdb.set_trace()
        for i, label in enumerate(labels_list):
            if label == -1:
                continue
            mask_idx = np.nonzero(masks_np == i)
            masked[mask_idx[0], mask_idx[1], :] *= 1.0 - alpha
            color = (colors[label]+class_count[label]*color_bias)/255.0
            # try:
            #     color = (colors[label]+class_count[label]*color_bias)/255.0
            # except:
            #     import pdb; pdb.set_trace()

            masked[mask_idx[0], mask_idx[1], :] += alpha * color
            # import pdb; pdb.set_trace()
            contours, hierarchy = cv2.findContours(
                (masks_np == i).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            # masked = cv2.drawContours(masked, contours, -1, (1.,1.,1.), -1)
            masked = cv2.drawContours(masked, contours, -1, (1.,1.,1.), 1)
            class_count[label] += 1

        return masked

    def show_image_v8(self, images, output, targets):
        N,C,H,W = images.shape
        masked_imgs = []
        origin_image_ON = True
        ori_sematic_target_ON = False
        ori_instance_target_ON = True
        car_only_ON = False
        pred_sematic_ON = True
        pred_instance_ON = True
        pred_stat_ON = True
        for i, image in enumerate(images):
            parts_to_show = []
            images_np = image.permute(1,2,0).detach().cpu().numpy()
            images_np = ((images_np - images_np.min()) / (images_np.max() - images_np.min()))[...,::-1]
            if origin_image_ON:
                parts_to_show.append(images_np)
            semat_a_target, insts_a_target = targets['label'][i], targets['instance'][i]

            # targets prepared in forwards
            # out_one = output['in_dict_watches'][i]
            # target_fwd_label = (out_one['target_levels']['labels']+1).tolist()
            sematic_target = output['targets']['label'][i]
            instacne_target = output['targets']['instance'][i]
            # sematic_out_513 = output['sematic_out'][6]
            sematic_out_513 = output['sematic_out']

            if ori_sematic_target_ON:
                sematic_target_np = sematic_target.detach().cpu().numpy()
                ori_sematic_target = self.rend_on_image_v8(images_np, sematic_target_np, range(19))
                parts_to_show.append(ori_sematic_target)

            if ori_instance_target_ON:
                instacne_target_np = instacne_target.detach().cpu().numpy()
                ins_cats = [sematic_target[instacne_target == ins_idx].unique().item() for ins_idx in instacne_target.unique()]
                if car_only_ON:
                    ins_cats = [13 if label==13 else -1 for label in ins_cats]
                ori_instacne_target = self.rend_on_image_v8(images_np, instacne_target_np, ins_cats)
                parts_to_show.append(ori_instacne_target)

            # import pdb; pdb.set_trace()
            # prediction    - sematic
            if pred_sematic_ON:
                sematic_predict_np = sematic_out_513[0].max(0)[1].detach().cpu().numpy()
                sematic_predict = self.rend_on_image_v8(images_np, sematic_predict_np, range(19))
                parts_to_show.append(sematic_predict)

            if not self.cfg.SOLVER.SEMATIC_ONLY:
                parymids = output['ins_pyramids']
                # prediction    - instance
                if pred_instance_ON:
                    # import pdb; pdb.set_trace()
                    instance_predict = torch.cat([pyr.get_mask(513) for pyr in  parymids[i]], dim=1) if len(parymids[i]) else torch.empty((1,1,513,513))
                    instance_predict_513 = F.interpolate(instance_predict, 513, mode='bilinear', align_corners=True)
                    instance_predict_np = instance_predict_513[0].max(0)[1].detach().cpu().numpy()
                    ins_cats_predict = [pyr.tar_cat for pyr in  parymids[i]]
                    # import pdb; pdb.set_trace()
                    pred_instance = self.rend_on_image_v8(images_np, instance_predict_np, ins_cats_predict)
                    # pred_instance = self.rend_on_image_v8(images_np, instance_predict_np, range(19))
                    if pred_stat_ON:
                        cv2.putText(pred_instance,'#:{}'.format(len(parymids[i])),(30,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
                        for pyr in parymids[i]:
                            # import pdb; pdb.set_trace()
                            pyr_pos_513 = pyr.get_rel_pos(7)[::-1]
                            cv2.circle(pred_instance, pyr_pos_513, 3, (0,0,255), -1)
                parts_to_show.append(pred_instance)

            masked_imgs.append(np.hstack(parts_to_show))

        masked_show = np.vstack(masked_imgs)
        cv2.imshow('Observation V5', masked_show)
        cv2.waitKey(10)

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        end = time.time()
        # tbar = tqdm(self.train_loader)
        # for i, sample in enumerate(tbar):
        for i, sample in enumerate(self.train_loader):
            # image = sample['image'].to(self.device)
            # target = {'label': sample['label'].to(self.device),
            #         'instance': sample['instance'].to(self.device)}
            image, label, instance = sample
            image, label, instance = image.to(self.device), label.to(self.device), instance.to(self.device)
            target = {'label': label, 'instance': instance}
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output_dict = self.model(image, target)
            # try:
            #     output_dict = self.model(image, target)
            # except:
            #     log_gpu_stat(self.logger)
            #     print('Num of instances:', self.model.log_dict['InstPyr_inst_count'])
            #     exit()
            # import pdb; pdb.set_trace()
            loss_dict = output_dict['loss_dict']
            losses = sum(loss for loss in loss_dict.values())
            self.model.log_dict['loss_dict'] = loss_dict
            # import pdb; pdb.set_trace()
            losses.backward()
            self.optimizer.step()
            batch_time = time.time() - end
            end = time.time()
            self.model.log_dict['loss_dict'] = loss_dict
            train_loss += losses.item()

            if i % 20 == 0 or i == len(self.train_loader) -1:
                curr_iter = epoch*len(self.train_loader)+ i + 1
                sepent_time = time.time() - self.meters['start_time']
                eta_seconds = sepent_time * self.meters['total_iters'] / curr_iter - sepent_time
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if self.cfg.SOLVER.SEMATIC_ONLY:
                    self.logger.info(('Ep:{}/{}|Iter:{}/{}|Eta:{}|SematicLoss:{:2.4}|Class:{:2.4}').format(
                        epoch, self.cfg.SOLVER.EPOCHES, i, len(self.train_loader),
                        eta_string, losses.item(), loss_dict['sematic'].item()))
                else:
                    self.logger.info(('Ep:{}/{}|Iter:{}/{}|Eta:{}|SematicLoss:{:2.4}|Class:{:2.4}').format(
                        epoch, self.cfg.SOLVER.EPOCHES, i, len(self.train_loader),
                        eta_string, losses.item(), loss_dict['sematic'].item()))
                    # self.logger.info(('Ep:{}/{}|Iter:{}/{}|Eta:{}|Loss:{:2.4}|Class:{:2.4}|'+\
                    #     'L0:{:2.4}|L1:{:2.4}|L2:{:2.4}|L3:{:2.4}|PyrNum:{}|#pyr0:{}|'+\
                    #     '#pyr1:{}|#pyr2:{}|#pyr3:{}|').format(
                    #     epoch, self.cfg.SOLVER.EPOCHES, i, len(self.train_loader),
                    #     eta_string, losses.item(), loss_dict['class_loss'].item(), loss_dict['level_0'].item(),
                    #     loss_dict['level_1'].item(), loss_dict['level_2'].item(), loss_dict['level_3'].item(),
                    #     self.model.log_dict['InstPyr_inst_count'], self.model.log_dict['pyr_num_l0'], 
                    #     self.model.log_dict['pyr_num_l1'], self.model.log_dict['pyr_num_l2'], 
                    #     self.model.log_dict['pyr_num_l3'], 
                    # ))
            if self.cfg.SOLVER.SHOW_IMAGE and i % 10 == 0:
                self.show_image_v8(image, output_dict, target)
        self.writer.add_scalar('train/loss_epoch', train_loss, epoch)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        test_loss = 0.0
        tbar = tqdm(self.val_loader, desc='\r')
        for i, sample in enumerate(tbar):
            image, label, instance = sample
            image, label, instance = image.to(self.device), label.to(self.device), instance.to(self.device)
            target = {'label': label, 'instance': instance}
            with torch.no_grad():
                output = self.model(image, target)
            # import pdb; pdb.set_trace()
            sematic_out = output['sematic_out']
            pred = sematic_out.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(label.detach().cpu().numpy(), pred)

        # Fast test during the training
        # import pdb; pdb.set_trace()
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)

        self.logger.info('Evalueat report: mIoU: {:3.4}| Acc: {:3.4}| Acc_class: {:3.4}| fwIoU: {:3.4}| previousBest: {:3.4}|'.format(
            mIoU, Acc, Acc_class, FWIoU, float(self.best_pred)
        ))
        if mIoU > self.best_pred:
            is_best = True
            self.best_pred = mIoU
            save_data = {}
            save_data["epoch"] = epoch + 1
            save_data["best_pred"] = self.best_pred
            save_data["model"] = self.model.state_dict()
            if self.optimizer is not None:
                save_data["optimizer"] = self.optimizer.state_dict()
            if self.scheduler is not None:
                save_data["scheduler"] = self.scheduler.__dict__ 
            torch.save(save_data, self.output_dir+"/model_Epoch_{}.pth".format(epoch))


def log_gpu_stat(logger):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used_mem = (meminfo.used / 1024) /1024
    pynvml.nvmlShutdown()
    gpu_info = subprocess.check_output(["nvidia-smi"])
    logger.info(("\nThe pid of current job is {} and {}, the used memory before we run is {}MB,"+\
        " the <nvidia-smi> shows:\n{}").format(os.getpid(),os.getppid(), used_mem, gpu_info.decode("utf-8")))


def main():
    parser = argparse.ArgumentParser(description="PyTorch MaskPyramid Training for Cityscapes")
    parser.add_argument(        # args.config
        "--config",
        default="maskpyramid/config/gallery/MaskPyramidV5.yml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(        # args.opts
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    # cfg.merge_from_list(['SOLVER.SHOW_IMAGE', False])
    cfg.merge_from_list(['DATALOADER.BATCH_SIZE_TRAIN', 2])
    # cfg.merge_from_list(['DATALOADER.NUM_WORKERS', 0])
    cfg.merge_from_list(['SOLVER.SEMATIC_ONLY', True])
    cfg.merge_from_list(['MULTI_RUN', True])
    # cfg.merge_from_list(['MODEL.WEIGHT', 'run/mpc_v9/Every_5_model_Epoch_130.pth'])
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    extra_exp_name = 'sematic' if cfg.SOLVER.SEMATIC_ONLY else ''
    experiment_name = os.path.basename(__file__).split('.py')[0]
    experiment_name += '_'+extra_exp_name if extra_exp_name else ''
    if cfg.MULTI_RUN:
        run_names = glob.glob(cfg.OUTPUT_DIR + '/*')
        matched_nums = []
        for name in run_names:
            matched_nums += re.findall(r'%s_(\d+)'%experiment_name, name)
        experiment_name += '_' + str(max([int(n) for n in matched_nums])+1) if matched_nums else '_1'
    output_dir = os.path.join(cfg.OUTPUT_DIR, experiment_name)
    
    mkdir(output_dir)
    with open(os.path.join(output_dir, experiment_name+'_config.yml'), 'w') as f:
        f.write(cfg.dump())
    logger = setup_logger("MaskPyramid", output_dir, 0, experiment_name+"_log.txt")  # get_rank())
    logger.info(args)
    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())
    logger.info("Loaded and archive configuration file at '{}'".format(os.path.join(output_dir, experiment_name+'_config.yml')))
    log_gpu_stat(logger)

    trainer = Trainer(cfg, output_dir)
    logger.info('Starting Epoch:{} / Total Epoches:{}'.format(trainer.start_epoch, cfg.SOLVER.EPOCHES))
    if cfg.MODEL.WEIGHT:
        trainer.load_weights(cfg.MODEL.WEIGHT)

    for epoch in range(trainer.start_epoch, cfg.SOLVER.EPOCHES):
        trainer.training(epoch)
        trainer.validation(epoch)
        log_gpu_stat(logger)
        if epoch == 5 or epoch % 20 == 0:
            save_data = {}
            save_data["epoch"] = epoch + 1
            save_data["best_pred"] = -1
            save_data["model"] = trainer.model.state_dict()
            if trainer.optimizer is not None:
                save_data["optimizer"] = trainer.optimizer.state_dict()
            if trainer.scheduler is not None:
                save_data["scheduler"] = trainer.scheduler.__dict__ 
            torch.save(save_data, trainer.output_dir+"/Every_5_model_Epoch_{}.pth".format(epoch))



    trainer.writer.close()

if __name__ == "__main__":
    main()


# gm_v2 batchNorm --> bias only
# v10_5_resunet_v2_sematic_1 层次加深到 feature map size 最小为4 
# v10.2: imshow 展示pyr num 与实际的instance target
# v10: Unet 下，不再多层输出，仅使用最终mask 通过放缩来匹配筛选各级mask

# v9: resnet 换成 Unet

# v8: 新的策略：
# 首先使用sematic segmentation 的 cross entropy 得出19个类的logits，
# 然后，对于可数的类，找未instance化的极值，初始化为instance，同时，instance 在计算时侵蚀（×0）掉该class logis
# 注意， 这个侵蚀， training时用target 侵蚀， testing 用instance 高于其它类别的值侵蚀
# 要有个while循环来把所有sematic 部分instance化
