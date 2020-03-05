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
from maskpyramid.modeling.backbone import resnet
from maskpyramid.solver.build import make_optimizer, make_lr_scheduler
from dataloaders.build import make_data_loader
from maskpyramid.utils.metrics import Evaluator
from maskpyramid.utils.lr_scheduler import LR_Scheduler
from maskpyramid.utils.tools import setup_logger, mkdir, collect_env_info
from maskpyramid.utils.tools import TensorboardSummary


class ResNet50(nn.Module):
    def __init__(self, cfg):
        super(ResNet50, self).__init__()
        self.stem = resnet.BaseStem(cfg, nn.BatchNorm2d)


        self.res_layer_1 = nn.Sequential(
            resnet.Bottleneck(64, 64, 256, 1, True, 1, 1, nn.BatchNorm2d),
            resnet.Bottleneck(256, 64, 256, 1, True, 1, 1, nn.BatchNorm2d),
            resnet.Bottleneck(256, 64, 256, 1, True, 1, 1, nn.BatchNorm2d),
        )

        self.res_layer_2 = nn.Sequential(
            resnet.Bottleneck(256, 128, 512, 1, True, 2, 1, nn.BatchNorm2d),
            resnet.Bottleneck(512, 128, 512, 1, True, 1, 1, nn.BatchNorm2d),
            resnet.Bottleneck(512, 128, 512, 1, True, 1, 1, nn.BatchNorm2d),
            resnet.Bottleneck(512, 128, 512, 1, True, 1, 1, nn.BatchNorm2d),
        )

        self.res_layer_3 = nn.Sequential(
            resnet.Bottleneck(512, 128, 256, 1, True, 2, 1, nn.BatchNorm2d),
            resnet.Bottleneck(256, 128, 256, 1, True, 1, 1, nn.BatchNorm2d),
            resnet.Bottleneck(256, 128, 256, 1, True, 1, 1, nn.BatchNorm2d),
            resnet.Bottleneck(256, 128, 256, 1, True, 1, 1, nn.BatchNorm2d),
            resnet.Bottleneck(256, 128, 256, 1, True, 1, 1, nn.BatchNorm2d),
            resnet.Bottleneck(256, 128, 256, 1, True, 1, 1, nn.BatchNorm2d),
        )

        self.res_layer_4 = nn.Sequential(
            resnet.Bottleneck(256, 128, 512, 1, True, 2, 1, nn.BatchNorm2d),
            resnet.Bottleneck(512, 128, 512, 1, True, 1, 1, nn.BatchNorm2d),
            resnet.Bottleneck(512, 128, 512, 1, True, 1, 1, nn.BatchNorm2d),
        )
        self.res_layer_5 = nn.Sequential(
            resnet.Bottleneck(512, 128, 256, 1, True, 2, 1, nn.BatchNorm2d),
            resnet.Bottleneck(256, 128, 256, 1, True, 1, 1, nn.BatchNorm2d),
            resnet.Bottleneck(256, 128, 256, 1, True, 1, 1, nn.BatchNorm2d),
        )
        self.res_layer_6 = nn.Sequential(
            resnet.Bottleneck(256, 128, 512, 1, True, 2, 1, nn.BatchNorm2d),
            resnet.Bottleneck(512, 128, 512, 1, True, 1, 1, nn.BatchNorm2d),
            resnet.Bottleneck(512, 128, 512, 1, True, 1, 1, nn.BatchNorm2d),
        )

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        x = self.res_layer_1(x)
        outputs.append(x)
        x = self.res_layer_2(x)
        outputs.append(x)
        x = self.res_layer_3(x)
        outputs.append(x)
        x = self.res_layer_4(x)
        outputs.append(x)
        x = self.res_layer_5(x)
        outputs.append(x)
        x = self.res_layer_6(x)
        outputs.append(x)
        
        return outputs

# v4: show running instance
class MaskPyramids(nn.Module):
    def __init__(self, cfg):
        super(MaskPyramids, self).__init__()
        self.cfg = cfg
        # self.r50 = resnet.ResNet(cfg)
        self.resnet50 = ResNet50(cfg)
        num_classes = cfg.DATALOADER.NUM_CLASSES

        self.init_pyramid = nn.Sequential(
            resnet.Bottleneck(256, 256, 256, 1, True, 1, 1, nn.BatchNorm2d),
            resnet.Bottleneck(256, 256, 256, 1, True, 1, 1, nn.BatchNorm2d),
            resnet.Bottleneck(256, 256, 256, 1, True, 1, 1, nn.BatchNorm2d),
        )
        self.class_logits = nn.Sequential(
            resnet.Bottleneck(256, 64, 128, 1, True, 2, 1, nn.BatchNorm2d),
            resnet.Bottleneck(128, 64, 64, 1, True, 2, 1, nn.BatchNorm2d),
            resnet.Bottleneck(64, 32, 32, 1, True, 2, 1, nn.BatchNorm2d),
        )
        self.cls_score = nn.Linear(32*5*5, num_classes)

        self.chn256_6 = nn.Conv2d(512+2, 256, 1)
        self.chn256_5 = nn.Conv2d(256+2, 256, 1)
        self.chn256_4 = nn.Conv2d(512+2, 256, 1)
        self.chn256_3 = nn.Conv2d(256+2, 256, 1)
        self.chn256_2 = nn.Conv2d(512+2, 256, 1)
        self.chn256_s = [self.chn256_6, self.chn256_5, self.chn256_4, self.chn256_3, self.chn256_2]

        self.conv_6 = resnet.Bottleneck(256, 64, 256, 1, True, 1, 1, nn.BatchNorm2d)
        self.conv_5 = resnet.Bottleneck(256, 64, 256, 1, True, 1, 1, nn.BatchNorm2d)
        self.conv_4 = resnet.Bottleneck(256, 64, 256, 1, True, 1, 1, nn.BatchNorm2d)
        self.conv_3 = resnet.Bottleneck(256, 64, 256, 1, True, 1, 1, nn.BatchNorm2d)
        self.conv_2 = resnet.Bottleneck(256, 64, 256, 1, True, 1, 1, nn.BatchNorm2d)
        self.conv_s = [self.conv_6, self.conv_5, self.conv_4, self.conv_3, self.conv_2]

        self.out_6 = nn.Conv2d(256, 1, 1)
        self.out_5 = nn.Conv2d(256, 1, 1)
        self.out_4 = nn.Conv2d(256, 1, 1)
        self.out_3 = nn.Conv2d(256, 1, 1)
        self.out_2 = nn.Conv2d(256, 1, 1)
        self.out_s = [self.out_6, self.out_5, self.out_4, self.out_3, self.out_2]

        self.bg_6 = nn.Conv2d(512, 1, 1)
        self.bg_5 = nn.Conv2d(256, 1, 1)
        self.bg_4 = nn.Conv2d(512, 1, 1)
        self.bg_3 = nn.Conv2d(256, 1, 1)
        self.bg_2 = nn.Conv2d(512, 1, 1)
        self.bg_s = [self.bg_6, self.bg_5, self.bg_4, self.bg_3, self.bg_2]

        self.upmix256_5 = nn.Conv2d(256+256+2, 256, 1)
        self.upmix256_4 = nn.Conv2d(512+256+2, 256, 1)
        self.upmix256_3 = nn.Conv2d(256+256+2, 256, 1)
        self.upmix256_2 = nn.Conv2d(512+256+2, 256, 1)
        self.upmix256_s = [None, self.upmix256_5, self.upmix256_4, self.upmix256_3, self.upmix256_2]

        # self.sematic_conv_5 = resnet.Bottleneck(512+256, 64, 256, 1, True, 1, 1, nn.BatchNorm2d)
        self.sematic_conv_5 = resnet.Bottleneck(512+256, 64, 256, 1, True, 1, 1, nn.BatchNorm2d)
        self.sematic_conv_4 = resnet.Bottleneck(512+256, 64, 256, 1, True, 1, 1, nn.BatchNorm2d)
        self.sematic_conv_3 = resnet.Bottleneck(256+256, 64, 256, 1, True, 1, 1, nn.BatchNorm2d)
        self.sematic_conv_2 = resnet.Bottleneck(512+256, 64, 256, 1, True, 1, 1, nn.BatchNorm2d)
        self.sematic_conv_1 = resnet.Bottleneck(256+256, 64, 256, 1, True, 1, 1, nn.BatchNorm2d)
        self.sematic_final = nn.Conv2d(256, num_classes, 1)


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

    def _init_target(self, img_tensor_shape, device, target=None, run_idx=0):
        target_levels = {}
        if self.cfg.DATALOADER.DATASET == "coco":
            target_ori_mask = target.get_field('masks').get_mask_tensor().unsqueeze(0).to(device)
            target_levels['labels'] = target.get_field('labels')
            target_shape = (1, target_ori_mask.shape[-3]) + img_tensor_shape
            target_mask_pad_to_img = target_ori_mask.new(*target_shape).zero_()
            target_mask_pad_to_img[:,:,:target.size[1], :target.size[0]] = target_ori_mask
        elif self.cfg.DATALOADER.DATASET == "cityscapes":
            # import pdb; pdb.set_trace()
            target_sematic = target['label'][run_idx]
            ins_mask = target['instance'][run_idx]
            target_ori_mask = torch.cat([ins_mask[None] == lbl for lbl in ins_mask.unique()])[None]
            # target_levels['labels'] = torch.cat([(mask.float()*target_sematic).max().view(1) \
            target_levels['labels'] = torch.cat([(mask.long()*target_sematic).max().view(1) \
                    for mask in target_ori_mask[0]]).long()
            # if target_sematic.max()> 18:
            #     import pdb; pdb.set_trace()
            # target_levels['labels'][target_levels['labels']==255]=0
            target_mask_pad_to_img = target_ori_mask
        else:
            import pdb; pdb.set_trace()

        target_levels[0] = target_mask_pad_to_img
        level_shape = ((img_tensor_shape[0]+1)//2, (img_tensor_shape[1]+1)//2)
        target_levels[1] = F.interpolate(target_mask_pad_to_img.float(), level_shape, mode='bilinear').type(target_mask_pad_to_img.dtype)
        level_shape = ((level_shape[0]+1)//2, (level_shape[1]+1)//2)
        target_levels[2] = F.interpolate(target_mask_pad_to_img.float(), level_shape, mode='bilinear').type(target_mask_pad_to_img.dtype)
        level_shape = ((level_shape[0]+1)//2, (level_shape[1]+1)//2)
        target_levels[3] = F.interpolate(target_mask_pad_to_img.float(), level_shape, mode='bilinear').type(target_mask_pad_to_img.dtype)
        level_shape = ((level_shape[0]+1)//2, (level_shape[1]+1)//2)
        target_levels[4] = F.interpolate(target_mask_pad_to_img.float(), level_shape, mode='bilinear').type(target_mask_pad_to_img.dtype)
        level_shape = ((level_shape[0]+1)//2, (level_shape[1]+1)//2)
        target_levels[5] = F.interpolate(target_mask_pad_to_img.float(), level_shape, mode='bilinear').type(target_mask_pad_to_img.dtype)
        level_shape = ((level_shape[0]+1)//2, (level_shape[1]+1)//2)
        target_levels[6] = F.interpolate(target_mask_pad_to_img.float(), level_shape, mode='bilinear').type(target_mask_pad_to_img.dtype)
        level_shape = ((level_shape[0]+1)//2, (level_shape[1]+1)//2)
        target_levels[7] = F.interpolate(target_mask_pad_to_img.float(), level_shape, mode='bilinear').type(target_mask_pad_to_img.dtype)

        return target_levels
   
    # 只能给新来的 pyramids match target， 旧的match 要保持连贯性
    def match_target(self, level, pyramids, target_levels, target_support_pyramids):
        for pyr in pyramids:
            target_idxs = target_levels[7-level][0, :, pyr.pos[0], pyr.pos[1]].nonzero()
            for i, target_idx in enumerate(target_idxs):
                target_idx_int = target_idx[0].item()
                target_support_pyramids[target_idx_int].append(pyr.idx)
                # 解决一个pixel 多个target 的问题， 核心：(小target优先)， 已分配的不改
                if pyr.target_idx is not None:
                    # print('target_idxs', target_idxs)
                    # print('pyr.target_idx:', pyr.target_idx)
                    target_map_last = target_levels[7-level][0, pyr.target_idx]
                    target_map_now = target_levels[7-level][0, target_idx_int]
                    # 重叠的上一个已经有其他pixel选项了， 让其让路
                    if len(target_support_pyramids[pyr.target_idx]) > 1:
                        target_support_pyramids[pyr.target_idx].remove(pyr.idx)
                        pyr.bind_target(target_idx_int)
                    elif len(target_support_pyramids[target_idx_int]) > 1:
                        target_support_pyramids[target_idx_int].remove(pyr.idx)
                        continue
                    elif (target_map_now == target_map_last).all():
                        target_support_pyramids[target_idx_int].remove(pyr.idx)
                        continue
                    elif target_map_now.sum() < target_map_last.sum():
                        target_support_pyramids[pyr.target_idx].remove(pyr.idx)
                        pyr.bind_target(target_idx_int)
                    elif target_map_now.sum() > target_map_last.sum():
                        target_support_pyramids[target_idx_int].remove(pyr.idx)
                        continue
                    else:
                        target_support_pyramids[target_idx_int].remove(pyr.idx)
                        continue
                        # import pdb; pdb.set_trace()
                else:
                    pyr.bind_target(target_idx_int)

    def compute_mask(self, level, feature, pyramids, is_init=False):
        x_shared_bg = self.bg_s[level](feature)
        for pyramid in pyramids:
            # import pdb; pdb.set_trace()
            root_pos = pyramid.get_root_level_pos(level)
            root_value = feature.new_ones((feature.shape[0],1,feature.shape[2],feature.shape[3]))\
                 * feature[0, 0, root_pos[0], root_pos[1]]
            feature_gaussian_mask = F.interpolate(pyramid.init_gaussian_mask[None,None,...]\
                .to(feature.device), tuple(feature[0,0].shape), mode='bilinear', align_corners=False)[0,0]
            if is_init:
                conv_in = torch.cat((feature, feature_gaussian_mask[None, None,:,:], root_value), dim=1)
                chn256 = self.chn256_s[level](conv_in)
                x_init = self.init_pyramid(chn256)
                mask_logits = self.conv_s[level](x_init)
                pyramid.set_mask_logits(level, mask_logits)
                mask_1 = self.out_s[level](mask_logits)
                mask = torch.cat((x_shared_bg, mask_1), dim=1)
                pyramid.set_mask(level, mask)
                
            else:
                last_mask_logits = pyramid.get_mask_logits(level-1)
                up_size = tuple(feature.shape[-2:])
                last_mask_logits_up = F.interpolate(last_mask_logits, up_size, mode='bilinear', align_corners=False)

                conv_in = torch.cat((last_mask_logits_up, feature, feature_gaussian_mask[None, None,:,:], root_value), dim=1)
                out = self.upmix256_s[level](conv_in)
                mask_logits = self.conv_s[level](out)
                pyramid.set_mask_logits(level, mask_logits)
                mask_1 = self.out_s[level](mask_logits)
                mask = torch.cat((x_shared_bg, mask_1), dim=1)
                pyramid.set_mask(level, mask)

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

    def compute_class(self, pyramids, target_levels):
        class_losses = []
        for pyramid in pyramids:
            mask_3 = pyramid.get_mask(3).max(1)[1][None].float()
            mask_logits_3 = pyramid.get_mask_logits(3)
            masked_class_logits = mask_logits_3 * mask_3
            class_logits = self.class_logits(masked_class_logits)
            if self.cfg.DATALOADER.DATASET == 'coco':
                class_logits_5x5 = F.interpolate(class_logits, (5, 5), mode='bilinear')
                class_logits = self.cls_score(class_logits_5x5.view(class_logits_5x5.shape[0],-1))
            elif self.cfg.DATALOADER.DATASET == 'cityscapes':
                class_logits = self.cls_score(class_logits.view(class_logits.shape[0],-1))

            pyramid.class_logits = class_logits
            pyramid.masked_class_logits = masked_class_logits

            if pyramid.target_idx is not None:
                # import pdb; pdb.set_trace()
                target_label = target_levels['labels'][[pyramid.target_idx]]
                loss_class = self.class_criteron(class_logits, target_label)
                class_losses.append(loss_class)

        class_loss = sum(loss for loss in class_losses) / len(class_losses)\
            if len(class_losses) else mask_logits_3.sum()*0

        return class_loss

    def compute_sematic(self, feature, target):
        up_6_5 = F.interpolate(feature[5], feature[4].shape[2:], mode='bilinear')
        sematic_5 = self.sematic_conv_5(torch.cat((feature[4], up_6_5), dim=1))
        up_5_4 = F.interpolate(sematic_5, feature[3].shape[2:], mode='bilinear')
        sematic_4 = self.sematic_conv_4(torch.cat((feature[3], up_5_4), dim=1))
        up_4_3 = F.interpolate(sematic_4, feature[2].shape[2:], mode='bilinear')
        sematic_3 = self.sematic_conv_3(torch.cat((feature[2], up_4_3), dim=1))
        up_3_2 = F.interpolate(sematic_3, feature[1].shape[2:], mode='bilinear')
        sematic_2 = self.sematic_conv_2(torch.cat((feature[1], up_3_2), dim=1))
        up_2_1 = F.interpolate(sematic_2, feature[0].shape[2:], mode='bilinear')
        sematic_1 = self.sematic_conv_1(torch.cat((feature[0], up_2_1), dim=1))
        up_final = F.interpolate(sematic_1, self.cfg.DATALOADER.CROP_SIZE, mode='bilinear')
        final = self.sematic_final(up_final)

        sematic_loss = self.sematic_criteron(final, target['label'].long())

        return sematic_loss, final

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
        xs_r50 = self.resnet50(image)
        N, _, img_size_h, img_size_w = image.shape
        device = image.device
        level_sizes = [tuple(f.shape[-2:]) for f in xs_r50[::-1]]

        losses = {}
        losses_0 = []
        losses_1 = []
        losses_2 = []
        losses_3 = []
        losses_4 = []
        losses_class = []
        all_pyramids = []
        test_masks = []
        target_support_pyramids = None
        for i in range(N):
            InstancePyramid.inst_count = 0
            curr_level = 0
            x_curr = xs_r50[-1]
            init_pos = torch.nonzero(torch.ones_like(x_curr[0][0]))
            inst_pyramids = [InstancePyramid(pos, curr_level, level_sizes) for pos in init_pos]
            self.compute_mask(curr_level, x_curr[[i]], inst_pyramids, True)
            self.log_dict.update({'pyr_num_l0': len(inst_pyramids)})
            target_levels = None
            if self.training:
                # target_levels = self._init_target((img_size_h, img_size_w ), device, targets[i])
                target_levels = self._init_target((img_size_h, img_size_w ), device, targets, i)
                target_support_pyramids = [[] for k in range(target_levels[7].shape[1])]
                # 统计 target 匹配
                self.match_target(0, inst_pyramids, target_levels, target_support_pyramids)
                
                loss_0 = self.compute_loss(0, inst_pyramids, target_levels, target_support_pyramids)
                losses_0.append(loss_0)
    
            inst_pyramids = self.forward_singel_level(1, inst_pyramids, xs_r50[-2], i, level_sizes, 
                    target_support_pyramids, target_levels, losses_1)

            inst_pyramids = self.forward_singel_level(2, inst_pyramids, xs_r50[-3], i, level_sizes, 
                    target_support_pyramids, target_levels, losses_2)

            inst_pyramids = self.forward_singel_level(3, inst_pyramids, xs_r50[-4], i, level_sizes, 
                    target_support_pyramids, target_levels, losses_3)

            # import pdb; pdb.set_trace()
            # class_loss = self.compute_class(inst_pyramids, target_levels)


            # losses_class.append(class_loss)
            all_pyramids.append(inst_pyramids)

            if not self.training:
                test_masks.append(inst_pyramids)

        sematic_loss, sematic_out = self.compute_sematic(xs_r50, targets)
            
        self.log_dict.update({'InstPyr_inst_count': InstancePyramid.inst_count})
        # import pdb; pdb.set_trace()
        losses['class_loss']= sematic_loss
        # losses['class_loss']= sum(loss for loss in losses_class)
        losses['level_0']= sum(loss for loss in losses_0)
        losses['level_1']= sum(loss for loss in losses_1)
        losses['level_2']= sum(loss for loss in losses_2)
        losses['level_3']= sum(loss for loss in losses_3)
        # losses['level_4']= sum(loss for loss in losses_4)

        # if self.training:
        #     return losses
        # else:
        #     output = dict()
        #     # import pdb; pdb.set_trace()
        #     for pyrs in test_masks:
        #         masks = torch.cat([pyr.get_mask(3) for pyr in pyrs])
        #     # test_masks[0]
        #     output['sematic_out'] = sematic_out

        #     return output

        output_dict = dict()
        output_dict['loss_dict'] = losses
        output_dict['sematic_out'] = sematic_out
        output_dict['all_pyramids'] = all_pyramids

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
        self.mask_logits = {}
        self.class_logits = None
        self.target_idx = None
        self.is_alive = True
        # torch.tensor(800.0/2**(7-self.init_level)).ceil().long().item()
        self.feature_scales = [7, 13, 25, 50, 100, 200]
        # self.gaussian_masks = self.generate_gaussian_masks()
        # import pdb; pdb.set_trace()
        # self.shared_gaussian_mask = self.shared_gaussian_masks()
        self.init_gaussian_mask = self.init_gaussian_masks()

    def set_mask(self, pub_level, mask):
        self.masks[pub_level - self.init_level] = mask

    def get_mask(self, pub_level):
        return self.masks[pub_level - self.init_level]

    def set_mask_logits(self, pub_level, mask_logits):
        self.mask_logits[pub_level - self.init_level] = mask_logits

    def get_mask_logits(self, pub_level):
        return self.mask_logits[pub_level - self.init_level]

    def bind_target(self, idx):
        self.target_idx = idx

    def get_root_level_pos(self, pub_level):
        init_size = self.level_sizes[self.init_level]
        req_size = self.level_sizes[pub_level]

        h = (self.pos[0].float() / init_size[0] * req_size[0]).round().long()
        w = (self.pos[1].float() / init_size[1] * req_size[1]).round().long()

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


class Trainer(object):
    def __init__(self, cfg, output_dir):
        self.cfg = cfg
        self.output_dir = output_dir
        self.logger = logging.getLogger("MaskPyramid")
        self.tbSummary = TensorboardSummary(output_dir)
        self.writer = self.tbSummary.create_summary()
        self.model = MaskPyramids(cfg)
        # self.optimizer = make_optimizer(cfg, self.model)
        # self.scheduler = make_lr_scheduler(cfg, self.optimizer)
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

        log_gpu_stat(self.logger)

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

    def rend_on_image(self, image_np, masks_np, labels):
        alpha = 0.5
        colors = np.array([
            [0, 100, 0],
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
        for i, label in enumerate(labels):
            mask_idx = np.nonzero(masks_np == i)
            masked[mask_idx[0], mask_idx[1], :] *= 1.0 - alpha
            color = (colors[label+1]+i)/255.0
            masked[mask_idx[0], mask_idx[1], :] += alpha * color
            # import pdb; pdb.set_trace()
            contours, hierarchy = cv2.findContours(
                (masks_np == i).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            # masked = cv2.drawContours(masked, contours, -1, (1.,1.,1.), -1)
            masked = cv2.drawContours(masked, contours, -1, (1.,1.,1.), 1)

        return masked

    def show_image(self, images, output, targets):
        N,C,H,W = images.shape
        masked_imgs = []
        for i, image in enumerate(images):
            images_np = image.permute(1,2,0).detach().cpu().numpy()
            images_np = ((images_np - images_np.min()) / (images_np.max() - images_np.min()))[...,::-1]
            semat_a_image, insts_a_image = targets['label'][i], targets['instance'][i]
            if max([len(semat_a_image[insts_a_image == j].unique()) for j in range(len(insts_a_image.unique()))]) > 1:
                import pdb; pdb.set_trace()
            class_of_inst = [semat_a_image[insts_a_image == j].unique().item() for j in range(len(insts_a_image.unique()))]
            class_names = self.train_loader.dataset.class_names
            insts_a_image_np = insts_a_image.detach().cpu().numpy()
            masked_target = self.rend_on_image(images_np, insts_a_image_np, class_of_inst)
            # prediction
            sematic_a_image = output['sematic_out'][i]
            sematic_a_image_np = sematic_a_image.max(0)[1].detach().cpu().numpy()
            masked_sematic = self.rend_on_image(images_np, sematic_a_image_np, range(19))
            # instance
            pyramids = output['all_pyramids'][i]
            inst_masks = torch.cat([pyramids[0].get_mask(3)[:,0]]+[pyr.get_mask(3)[:,1] for pyr in pyramids], dim=0)
            inst_a_image = F.interpolate(inst_masks[None], self.cfg.DATALOADER.CROP_SIZE, mode='bilinear', align_corners=True)[0]
            inst_a_image_np = inst_a_image.max(0)[1].detach().cpu().numpy()
            # TODO: 标label的3种逻辑：1.以root处为准；2.以mask响应最高处为准；3.以本mask出头范围内响应最高处为准
            # import pdb; pdb.set_trace()
            inst_label = [-1,] + [F.interpolate(semat_a_image[None,None].float(), pyr.init_size, mode='nearest')[0,\
                0,pyr.pos[0], pyr.pos[1]].long().item() for pyr in pyramids]
            masked_instance = self.rend_on_image(images_np, sematic_a_image_np, inst_label)

            masked_imgs.append(np.hstack((images_np, masked_target, masked_sematic, masked_instance)))

        masked_show = np.vstack(masked_imgs)
        cv2.imshow('Observation V4', masked_show)
        cv2.waitKey(1)

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
            loss_dict = output_dict['loss_dict']
            losses = sum(loss for loss in loss_dict.values())
            self.model.log_dict['loss_dict'] = loss_dict
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
                self.logger.info(('Ep:{}/{}|Iter:{}/{}|Eta:{}|Loss:{:2.4}|Class:{:2.4}|'+\
                    'L0:{:2.4}|L1:{:2.4}|L2:{:2.4}|L3:{:2.4}|PyrNum:{}|#pyr0:{}|'+\
                    '#pyr1:{}|#pyr2:{}|#pyr3:{}|').format(
                    epoch, self.cfg.SOLVER.EPOCHES, i, len(self.train_loader),
                    eta_string, losses.item(), loss_dict['class_loss'].item(), loss_dict['level_0'].item(),
                    loss_dict['level_1'].item(), loss_dict['level_2'].item(), loss_dict['level_3'].item(),
                    self.model.log_dict['InstPyr_inst_count'], self.model.log_dict['pyr_num_l0'], 
                    self.model.log_dict['pyr_num_l1'], self.model.log_dict['pyr_num_l2'], 
                    self.model.log_dict['pyr_num_l3'], 
                ))
            if self.cfg.SOLVER.SHOW_IMAGE and i % 2 == 0:
                self.show_image(image, output_dict, target)
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
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)

        self.logger.info('Evalueat report: mIoU: {:3.4}| Acc: {:3.4}| Acc_class: {:3.4}| fwIoU: {:3.4}|'.format(
            mIoU, Acc, Acc_class, FWIoU
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
        default="maskpyramid/config/gallery/MaskPyramidV1.yml",
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
    # cfg.merge_from_list(['DATALOADER.BATCH_SIZE_TRAIN', 4])
    cfg.merge_from_list(['DATALOADER.NUM_WORKERS', 0])
    # cfg.merge_from_list(['MODEL.WEIGHT', ''])
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    experiment_name = os.path.basename(__file__).split('.py')[0]
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
    trainer.load_weights('run/mpc_v3/model_Epoch_  9.pth')

    for epoch in range(trainer.start_epoch, cfg.SOLVER.EPOCHES):
        trainer.training(epoch)
        log_gpu_stat(logger)
        trainer.validation(epoch)


    import pdb; pdb.set_trace()
    trainer.writer.close()

if __name__ == "__main__":
    main()