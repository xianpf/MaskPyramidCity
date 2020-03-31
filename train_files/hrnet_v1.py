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
from grow_model.norm_conv import MeanResNormConv2d
import functools
import torch.backends.cudnn as cudnn

from hrnet.sync_bn.inplace_abn.bn import InPlaceABNSync

# BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=False)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3, 
                                            momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear')
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse

blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self, config, **kwargs):
        self.cfg = config
        extra = config.MODEL.EXTRA
        super(HighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)

        self.stage1_cfg = extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion*num_channels

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)
        
        last_inp_channels = np.int(np.sum(pre_stage_channels))

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=last_inp_channels,
                # out_channels=config.DATASET.NUM_CLASSES,
                out_channels=config.DATALOADER.NUM_CLASSES,
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0)
        )
        self.sematic_criteron = nn.CrossEntropyLoss(ignore_index=cfg.DATALOADER.IGNORE_INDEX)

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=False)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=False)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                      block,
                                      num_blocks,
                                      num_inchannels,
                                      num_channels,
                                      fuse_method,
                                      reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    # def forward(self, x):
    def forward(self, image, targets=None):
        # import pdb; pdb.set_trace()
        output_dict = dict()
        in_N, in_C, in_h, in_w = image.shape
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.upsample(x[1], size=(x0_h, x0_w), mode='bilinear')
        x2 = F.upsample(x[2], size=(x0_h, x0_w), mode='bilinear')
        x3 = F.upsample(x[3], size=(x0_h, x0_w), mode='bilinear')

        x = torch.cat([x[0], x1, x2, x3], 1)

        x = self.last_layer(x)

        # return x
        # import pdb; pdb.set_trace()
        sematic_out = F.upsample(x, size=(in_h, in_w), mode='bilinear')
        sematic_loss = self.sematic_criteron(sematic_out, targets['label'].long())


        if self.cfg.SOLVER.SEMATIC_ONLY:
            output_dict['loss_dict'] = {'sematic': sematic_loss.mean(), 'instance': 0}
            output_dict['sematic_out'] = sematic_out
            output_dict['targets'] = targets
            output_dict['ins_pyramids'] =[[]]
            return output_dict

    def init_weights(self, pretrained='',):
        logger = logging.getLogger("MaskPyramid")
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, InPlaceABNSync):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            # for k, _ in pretrained_dict.items():
            #     logger.info(
            #         '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


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
        # self.model = MaskPyramids(cfg)
        self.model = HighResolutionNet(cfg)
        self.model.init_weights(cfg.MODEL.PRETRAINED)
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
        if subdict == 'hrnet':
            # model_state_file = config.TEST.MODEL_FILE
            pretrained_dict = torch.load(path)
            if list(pretrained_dict.keys())[0].startswith('model.'):
                model_dict = self.model.state_dict()
                pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                                    if k[6:] in model_dict.keys()}
                # for k, _ in pretrained_dict.items():
                #     self.logger.info(
                #         '=> loading {} from pretrained model'.format(k))
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict)

            # model_dict = self.model.state_dict()

        elif subdict == 'model':
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
                sematic_predict_np = sematic_out_513[i].max(0)[1].detach().cpu().numpy()
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
            # import pdb; pdb.set_trace()
            losses.backward()
            self.optimizer.step()

            batch_time = time.time() - end
            end = time.time()
            # self.model.log_dict['loss_dict'] = loss_dict
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
                # output = self.model(image)
            # import pdb; pdb.set_trace()
            sematic_out = output['sematic_out']
            pred = sematic_out.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(label.detach().cpu().numpy(), pred)
            if self.cfg.SOLVER.SHOW_IMAGE and i % 10 == 0:
                self.show_image_v8(image, output, target)

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
        default="hrnet/hrnet.yml",
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
    # # cfg.merge_from_list(['SOLVER.SHOW_IMAGE', False])
    # cfg.merge_from_list(['DATALOADER.BATCH_SIZE_TRAIN', 2])
    # # cfg.merge_from_list(['DATALOADER.NUM_WORKERS', 0])
    # cfg.merge_from_list(['SOLVER.SEMATIC_ONLY', True])
    # cfg.merge_from_list(['SOLVER.BASE_LR', 1e-7])
    # cfg.merge_from_list(['TEST_ONLY', True])
    cfg.merge_from_list(['MULTI_RUN', True])
    # cfg.merge_from_list(['MODEL.WEIGHT', 'run/gm_v4_sematic_21/Every_5_model_Epoch_5.pth'])
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


    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLED

    trainer = Trainer(cfg, output_dir)
    logger.info('Starting Epoch:{} / Total Epoches:{}'.format(trainer.start_epoch, cfg.SOLVER.EPOCHES))
    if cfg.MODEL.WEIGHT:
        trainer.load_weights(cfg.MODEL.WEIGHT, subdict='hrnet')

    if cfg.TEST_ONLY:
        assert cfg.MODEL.WEIGHT, "You MUST provide a weight file for the testing mode!"
        trainer.validation(0)
    else:
        for epoch in range(trainer.start_epoch, cfg.SOLVER.EPOCHES):
            trainer.training(epoch)
            trainer.validation(epoch)
            log_gpu_stat(logger)
            if epoch % 5 == 0:
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


# gm_v4 mean threshed res norm conv
# v10_5_resunet_v2_sematic_1 层次加深到 feature map size 最小为4 
# v10.2: imshow 展示pyr num 与实际的instance target
# v10: Unet 下，不再多层输出，仅使用最终mask 通过放缩来匹配筛选各级mask

# v9: resnet 换成 Unet

# v8: 新的策略：
# 首先使用sematic segmentation 的 cross entropy 得出19个类的logits，
# 然后，对于可数的类，找未instance化的极值，初始化为instance，同时，instance 在计算时侵蚀（×0）掉该class logis
# 注意， 这个侵蚀， training时用target 侵蚀， testing 用instance 高于其它类别的值侵蚀
# 要有个while循环来把所有sematic 部分instance化
