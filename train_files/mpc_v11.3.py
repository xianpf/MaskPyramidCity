import sys, os, time, datetime, curses, logging
sys.path.insert(0, os.path.abspath(__file__+'/../..'))
import argparse, glob, re, cv2
from tqdm import tqdm
import numpy as np
import pynvml, subprocess
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import torch
from torch import nn
import torch.nn.functional as F

from maskpyramid.config import cfg
from maskpyramid.backbone.unet_baseline.unet_purebase import UNet, down, up
# from maskpyramid.modeling.backbone import resnet
from dataloaders.build import make_data_loader
from maskpyramid.utils.metrics import Evaluator
from maskpyramid.utils.lr_scheduler import LR_Scheduler
from maskpyramid.utils.tools import setup_logger, mkdir, collect_env_info
from maskpyramid.utils.tools import TensorboardSummary, flatten_list


class KeepDown(nn.Module):
    def __init__(self, cfg):
        super(KeepDown, self).__init__()
        self.down5 = down(512, 512)
        self.down6 = down(512, 512)
        self.down7 = down(512, 512)
        self.down8 = down(512, 512)
        self.down9 = down(512, 512)

    def forward(self, x5):
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.down7(x7)
        x9 = self.down8(x8)

        return [x6, x7, x8, x9]

class CategoryPyramid(nn.Module):
    def __init__(self, cfg):
        super(CategoryPyramid, self).__init__()
        num_classes = cfg.DATALOADER.NUM_CLASSES
        self.f_down_1x1 = nn.ModuleList([nn.ModuleList([nn.Conv2d(512,256,1) for \
            i in range(j+1)]) for j in range(6)])
        self.f_down_1x1_129 = nn.Conv2d(256,256,1)
        self.up1 = up(512, 256)    # 3
        self.up2 = up(512, 256)    # 5
        self.up3 = up(512, 256)    # 9
        self.up4 = up(512, 256)    # 17
        self.up5 = up(512, 256)    # 33
        self.up6 = up(512, 256)    # 3
        self.up129 = up(512, 256)    # 3
        self.out_conv = nn.Conv2d(256, 1, 1)
        
        self.up_list = [None, self.up1, self.up2, self.up3, self.up4, self.up5, self.up6]
        # self.up_list = nn.ModuleList([None, self.up1, self.up2, self.up3, 
        #     self.up4, self.up5, self.up6])

class MaskPyramids(nn.Module):
    def __init__(self, cfg):
        super(MaskPyramids, self).__init__()
        self.cfg = cfg
        num_classes = cfg.DATALOADER.NUM_CLASSES
        self.sematic_bone = UNet(num_classes)
        self.sematic_bone.load_state_dict(torch.load('maskpyramid/backbone/unet_baseline/checkpoint_MIOU54.4.pth.tar')['state_dict'])
        
        self.keep_down = KeepDown(cfg)
        cat_conv = [None for _ in range(11)] + [CategoryPyramid(cfg) for _ in range(8)]
        self.cat_conv = nn.ModuleList(cat_conv)
        self.sematic_criteron = nn.CrossEntropyLoss(ignore_index=cfg.DATALOADER.IGNORE_INDEX)
        self.sem_ins_criteron = nn.CrossEntropyLoss(ignore_index=cfg.DATALOADER.IGNORE_INDEX)

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
            return mode_num_sematic
        else:
            import pdb; pdb.set_trace()
        return 0

    def compute_mask(self, level, bid, feature, pyramids):
        masks = []
        involved_features = feature[::-1][level:]
        for pyr in pyramids:
            cat_conv = self.cat_conv[pyr.tar_cat]
            feat_cover_mask, p_rge = pyr.get_cover_mask(level)
            f_down_0 = involved_features[0][[bid], :, p_rge[0]:p_rge[1], p_rge[2]:p_rge[3]]
            x = cat_conv.f_down_1x1[level][0](f_down_0)
            # 0-3 1-5 2-9 3-17 4-33 5-65 6-129
            for i in range(1, 6-level):
                feat_cover_mask, p_rge = pyr.get_cover_mask(level+i)
                f_down = involved_features[i][[bid], :, p_rge[0]:p_rge[1], p_rge[2]:p_rge[3]]
                x_down = cat_conv.f_down_1x1[level+i][i](f_down)
                x = cat_conv.up_list[i](x, x_down)
            feat_cover_mask, p_rge = pyr.get_cover_mask(6)
            f_down_129 = feature[2][[bid], :, p_rge[0]:p_rge[1], p_rge[2]:p_rge[3]]
            x_down_129 = cat_conv.f_down_1x1_129(f_down_129)
            x_up_129 = cat_conv.up129(x, x_down_129)
            pad_129 = feature[0].new_zeros((1,x_up_129.shape[1], 129, 129))
            pad_129[0,:,p_rge[0]:p_rge[1], p_rge[2]:p_rge[3]] = x_up_129
            mask_129 = cat_conv.out_conv(pad_129)
            pyr.set_mask_129(mask_129)
            masks.append(mask_129)

        return masks

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
        masked = image_np.copy()
        for i, label in enumerate(labels_list):
            if label == -1:
                continue
            mask_idx = np.nonzero(masks_np == i)
            masked[mask_idx[0], mask_idx[1], :] *= 1.0 - alpha
            color = (colors[label]+class_count[label]*color_bias)/255.0
            
            masked[mask_idx[0], mask_idx[1], :] += alpha * color
            contours, hierarchy = cv2.findContours(
                (masks_np == i).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            masked = cv2.drawContours(masked, contours, -1, (1.,1.,1.), 1)
            class_count[label] += 1

        return masked

    def show_image_v8(self, images, output, targets, tool='cv2'):
        N,C,H,W = images.shape
        masked_imgs = []
        origin_image_ON = True
        ori_sematic_target_ON = False
        ori_instance_target_ON = True
        car_only_ON = False
        pred_sematic_ON = True
        pred_instance_ON = False
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
        if tool=='cv2':
            cv2.imshow('Observation V5', masked_show)
            cv2.waitKey(10)
        elif tool=='plt':
            plt.figure('show', figsize=(16,12))
            plt.imshow(masked_show)
            plt.tight_layout()
            plt.show()

    def make_observation(self, images, sem_targets=None, ins_target=None,
            size_limit=None, s_image=True, s_sem_out=False, sematic_out=None,
            typical_size=513, s_sem_tar=False, s_ins_tar=False, s_ins_out=True,
            ins_out=None, ins_out_label=None, pyramids=None,
            s_run_mask=False, run_mask=None):
        show_dict = dict()
        N, _, img_h, img_w = images.shape
        out_img = []
        for bid in range(N):
            per_bid_show_img = []
            image_np = images[bid].permute(1,2,0).detach().cpu().numpy()
            image_np = ((image_np - image_np.min()) / (image_np.max() - image_np.min()))
            if s_image:
                per_bid_show_img.append(image_np)
            if s_sem_tar:
                sem_target_np = sem_targets[bid].detach().cpu().numpy()
                res_img = self.rend_on_image_v8(image_np, sem_target_np, range(19))
                per_bid_show_img.append(res_img)
            if s_sem_out:
                sematic_out_np = sematic_out[bid].max(0)[1].detach().cpu().numpy()
                res_img = self.rend_on_image_v8(image_np, sematic_out_np, range(19))
                per_bid_show_img.append(res_img)
            if s_ins_tar:
                ins_target_np = ins_target[bid].detach().cpu().numpy()
                ins_cats = [sem_targets[bid][ins_target[bid] == ins_idx].unique().item() for ins_idx in ins_target[bid].unique()]
                res_img = self.rend_on_image_v8(image_np, ins_target_np, ins_cats)
                per_bid_show_img.append(res_img)
            if s_ins_out:
                ins_out_np = ins_out[bid][0].max(0)[1].detach().cpu().numpy()
                res_img = self.rend_on_image_v8(image_np, ins_out_np, ins_out_label[bid])
                cv2.putText(res_img,'#:{}'.format(len(pyramids[bid])),(30,30),cv2.FONT_HERSHEY_COMPLEX,1,(1,0,0),1)
                for pyr in pyramids[bid]:
                    # import pdb; pdb.set_trace()
                    # mask_513 = pyr.get_mask(513)[0,0]
                    # accurate_pos_513 = tuple(torch.nonzero(mask_513 == mask_513.max())[0].tolist())
                    pyr_pos_513 = pyr.get_rel_pos(8)[::-1]
                    # cv2.circle(res_img, accurate_pos_513, 5, (1,0,0), -1)
                    cv2.circle(res_img, pyr_pos_513, 5, (1,0,0), -1)
                per_bid_show_img.append(res_img)

            # if s_run_mask:
            #     import pdb; pdb.set_trace()
            #     run_mask_np = run_mask


            for oimg in per_bid_show_img:
                if oimg.shape[:2] != (typical_size, typical_size):
                    import pdb; pdb.set_trace()
            if len(per_bid_show_img) < 8:
                out_img.append(np.hstack(per_bid_show_img))
            else:
                import pdb; pdb.set_trace()
        out = np.vstack(out_img)
        if size_limit:
            import pdb; pdb.set_trace()
        return out

        return show_dict

    def show_image(self, images, out_dict=None, show_via='cv2'):
        show_img = out_dict['show_img']
        if show_via == 'cv2':
            show_img = show_img[:,:,::-1]
            cv2.imshow('Observation CV2', show_img)
            cv2.waitKey(10)
        elif show_via == 'plt':
            plt.figure('show', figsize=(16,12))
            plt.imshow(show_img)
            plt.tight_layout()
            plt.show()
        elif show_via == 'output':
            return show_img[:,:,::-1]

    def forward(self, image, targets=None):
        output_dict = dict()
        N, _, img_h, img_w = image.shape
        sematic_info = self.sematic_bone(image)
        sematic_out = sematic_info[0]
        sematic_loss = self.sematic_criteron(sematic_out, targets['label'].long())
        sematic_info += (sematic_loss,)
        sematic_out, down_features, up_features, sematic_loss = sematic_info

        output_dict['loss_dict'] = {'sematic': sematic_loss.mean(), 'instance': 0}
        output_dict['sematic_out'] = sematic_out
        output_dict['targets'] = targets
        output_dict['ins_pyramids'] =[[] for _ in range(N)]
        output_dict['observe_images'] =[[] for _ in range(N)]

        if self.cfg.SOLVER.SEMATIC_ONLY:
            output_dict['show_img'] = self.make_observation(image, sematic_out=sematic_out, 
                    sem_targets=targets['label'], ins_target=targets['instance'], 
                    s_sem_tar=True, s_ins_tar=True, s_sem_out=True, )
            # self.show_image(image, output_dict, show_via='plt')
            return output_dict

        keep_down_features = self.keep_down(down_features[-1])
        down_features = down_features + keep_down_features
        lvl_sizes = [tuple(f.shape[-2:]) for f in down_features[::-1]]
        
        pyr_losses = []
        ins_out = []
        ins_pyramids = [[] for _ in range(N)]
        instance_categories = [11, 12, 13, 14, 15, 16, 17, 18]
        for bid in range(N):
            if targets:
                ins_target_catch_pyr_id = [-1 for i in range(targets['instance'].unique().max()+1)]
                ins_idx_of_sem = [targets['instance'][bid][targets['label'][bid] == i].unique().tolist() for i in range(19)]
            scratch_sematic_129 = F.interpolate(sematic_out, 129, mode='bilinear', align_corners=True)[bid]
            stuff_max_129 = scratch_sematic_129[:11].max(0)[0]
            for sem_lvl in range(6):
                lvl_size = lvl_sizes[sem_lvl]
                if targets:
                    inst_target = self.prepare_targets(targets, mode='instance', down_size=lvl_size)[bid]
                    sematic_target = self.prepare_targets(targets, mode='sematic', down_size=lvl_size)[bid]
                    sematic_lvl_resize = F.interpolate(sematic_out, lvl_size, mode='bilinear', align_corners=True)[bid]
                
                running_sem_ins_129 = torch.cat([scratch_sematic_129]+[pyr.get_mask_129()[0] for pyr in ins_pyramids[bid]], dim=0)
                running_sem_ins_lvl = F.interpolate(running_sem_ins_129[None], lvl_size, mode='bilinear', align_corners=True)[0]
                max_value, max_idx = running_sem_ins_lvl.max(0)
                running_mask = (max_idx >= 11) * (max_idx < 19)
                # 策略1： 新生只与11个sematic类作比较
                pos_list = torch.nonzero(running_mask)
                # 这个new pyr 的生成方式要改
                for pos_t in pos_list:
                    pos = tuple(pos_t.tolist())
                    tar_cat = int(max_idx[pos])
                    if len(ins_pyramids[bid]) > 30:
                        break
                    new_pyr = InstancePyramid(pos, sem_lvl, lvl_sizes, tar_cat=tar_cat)
                    if targets:
                        for ins_idx in ins_idx_of_sem[tar_cat]:
                            if ins_target_catch_pyr_id[ins_idx] < 0 and (inst_target == ins_idx)[pos] > 0:
                                new_pyr.ins_tar_id=ins_idx
                                ins_target_catch_pyr_id[ins_idx] = new_pyr.idx
                                break

                    self.compute_mask(sem_lvl, bid, down_features, [new_pyr])

                    new_pyr_over_mask = new_pyr.get_mask_129()[0, 0] > stuff_max_129
                    # if not self.training and new_pyr_over_mask.sum() == 0:
                    #     continue
                    ins_pyramids[bid].append(new_pyr)
                    scratch_sematic_129 = scratch_sematic_129 * (1-new_pyr_over_mask.float())
                if targets:
                    for ins_tar_id in inst_target.unique():
                        target_cat = sematic_target[inst_target == ins_tar_id].unique().item()
                        if ins_target_catch_pyr_id[ins_tar_id] >= 0 or   \
                            target_cat not in instance_categories:
                            continue
                        else:
                            if (inst_target == ins_tar_id).sum() > 1:
                                pos_candis = torch.zeros_like(inst_target).float()
                                pos_candis[inst_target == ins_tar_id] = sematic_lvl_resize[target_cat][inst_target == ins_tar_id]
                                pos = tuple(torch.nonzero(pos_candis == pos_candis.max())[0].tolist())
                            else:
                                pos = tuple(torch.nonzero(inst_target == ins_tar_id)[0].tolist())
                            new_pyr = InstancePyramid(pos, sem_lvl, lvl_sizes, ins_tar_id, target_cat)
                            self.compute_mask(sem_lvl, bid, down_features, [new_pyr])
                            ins_target_catch_pyr_id[ins_tar_id] = new_pyr.idx
                            ins_pyramids[bid].append(new_pyr)

            mask_stuff = sematic_out[[bid], :11]
            if len(ins_pyramids[bid]) == 0:
                instance_loss = image.sum()* 0
                mask_sem_ins = mask_stuff
            else:
                # TODO: 合在一起来减少显存
                # mask_ins_129 = torch.cat([pyr.get_mask_129() for pyr in ins_pyramids[bid]], dim=1)
                # mask_ins = F.interpolate(mask_ins_129, 513, mode='bilinear', align_corners=True)
                # mask_sem_ins = torch.cat((mask_stuff, mask_ins), dim=1)
                mask_sem_ins_list = [mask_stuff]
                for pyr in ins_pyramids[bid]:
                    mask_129 = pyr.get_mask_129()
                    mask_513 = F.interpolate(mask_129, 513, mode='bilinear', align_corners=True)
                    pyr.set_mask(513, mask_513)
                    mask_sem_ins_list.append(mask_513)
                mask_sem_ins = torch.cat(mask_sem_ins_list, dim=1)

                if targets:
                    # target sem int
                    inst_target_idx = targets['label'][bid].clone()
                    inst_target_idx[inst_target_idx>=11] = -2
                    for i, pyr in enumerate(ins_pyramids[bid], start=11):
                        if pyr.ins_tar_id is not None:
                            inst_target_idx[targets['instance'][bid]==pyr.ins_tar_id] = i
                    if -2 in inst_target_idx.unique():
                        # print('Warning! Uncovered instance mask left on its sematic mask!, sum={}'.format(
                        #     (inst_target_idx==-2).sum()))
                        # import pdb; pdb.set_trace()
                        inst_target_idx[inst_target_idx==-2] = -1
                    inst_target_idx = inst_target_idx[None]
                    instance_loss = self.sem_ins_criteron(mask_sem_ins, inst_target_idx)
                # import pdb; pdb.set_trace()
            pyr_losses.append(instance_loss)
            ins_out.append(mask_sem_ins)

        ins_out_label = [[i for i in range(11)]+[pyr.tar_cat for pyr in pyrs] for pyrs in ins_pyramids]
        output_dict['show_img'] = self.make_observation(image, sematic_out=sematic_out, 
                sem_targets=targets['label'], ins_target=targets['instance'],
                ins_out = ins_out, ins_out_label=ins_out_label, pyramids=ins_pyramids,
                s_sem_tar=False, s_ins_tar=True, s_sem_out=True, s_ins_out=True)
        # self.show_image(image, output_dict, show_via='plt')
        
        output_dict['loss_dict'] = {'sematic': sematic_loss.mean(), 'instance': sum(pyr_losses)/N}
        output_dict['ins_pyramids'] =ins_pyramids

        return output_dict

class InstancePyramid():
    inst_count = 0
    def __init__(self, pos, init_lvl, lvl_sizes, ins_tar_id=-1, tar_cat=-1):
        self.idx = InstancePyramid.inst_count
        InstancePyramid.inst_count += 1
        self.pos = pos  # tuple required
        self.init_lvl = init_lvl
        self.lvl_sizes = lvl_sizes
        self.init_size = self.lvl_sizes[self.init_lvl]
        self.ins_tar_id = ins_tar_id
        self.tar_cat = tar_cat
        self.mask_129 = None
        self.masks = {}

        init_mask = torch.zeros(self.init_size)
        init_mask[self.pos] = 1
        cover_conv = nn.Conv2d(1,1,3,1,1, bias=False)
        cover_conv.weight.requires_grad = False
        cover_conv.weight.data.fill_(1)
        self.init_cover = cover_conv(init_mask[None,None])
        self.init_center_cover = cover_conv(init_mask[None,None])
        # center_mask = torch.ones(3,3)
        # if self.pos[0] <= 0:
        #     center_mask[:1, :] = 0
        # elif self.pos[0] >= self.init_size[0] - 1:
        #     center_mask[-1:, :] = 0
        # if self.pos[1] <= 0:
        #     center_mask[:, :1] = 0
        # elif self.pos[1] >= self.init_size[1] - 1:
        #     center_mask[:, -1:] = 0
        # self.center_mask = center_mask[None,None]
        

    def get_cover_mask_with_pyr_mask(self, lvl):
        f_mask = F.interpolate(self.init_cover, self.lvl_sizes[lvl], mode='nearest')[0,0]
        # pyr_mask = F.interpolate(self.center_mask, self.lvl_sizes[lvl-self.init_lvl], mode='nearest')
        pyr_mask = F.interpolate(self.center_mask, 2**(lvl-self.init_lvl+1)+1, mode='nearest')[0,0]
        return f_mask, pyr_mask
        
    def get_cover_mask(self, lvl):
        f_mask = F.interpolate(self.init_cover, self.lvl_sizes[lvl], mode='nearest')[0,0]
        # import pdb; pdb.set_trace()
        positive_pos = torch.nonzero(f_mask)
        pos_range = (positive_pos[:,0].min().item(), positive_pos[:,0].max().item()+1,
                    positive_pos[:,1].min().item(), positive_pos[:,1].max().item()+1)
        return f_mask, pos_range
        
    def set_mask(self, lvl, mask):
        self.masks[lvl] = mask

    def get_mask(self, lvl):
        return self.masks[lvl]

    def set_mask_129(self, mask):
        self.mask_129 = mask

    def get_mask_129(self):
        return self.mask_129

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
        init_size = self.lvl_sizes[self.init_lvl]
        req_size = self.lvl_sizes[pub_level]

        h = (self.pos[0].float() / init_size[0] * req_size[0]).round().long()
        w = (self.pos[1].float() / init_size[1] * req_size[1]).round().long()

        return (h.item(), w.item())
        
    def get_rel_pos(self, pub_level):
        init_size = self.lvl_sizes[self.init_lvl]
        req_size = self.lvl_sizes[pub_level]

        h = round((self.pos[0]+0.5) / init_size[0] * req_size[0]-0.5)
        w = round((self.pos[1]+0.5) / init_size[1] * req_size[1]-0.5)

        return (h, w)

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

        train_params = [{'params': self.model.keep_down.parameters(), 'lr': cfg.SOLVER.BASE_LR},
            {'params': self.model.cat_conv.parameters(), 'lr': cfg.SOLVER.BASE_LR},]
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
        for i, sample in enumerate(self.train_loader):
            image, label, instance = sample
            image, label, instance = image.to(self.device), label.to(self.device), instance.to(self.device)
            target = {'label': label, 'instance': instance}
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output_dict = self.model(image, target)
            loss_dict = output_dict['loss_dict']
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            self.optimizer.step()
            batch_time = time.time() - end
            end = time.time()
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

            if self.cfg.SOLVER.SHOW_IMAGE and i % 10 == 0:
                self.model.show_image(image, output_dict)
                if i // 10 % 10 == 0:
                    observe_out = self.model.show_image(image, output_dict, show_via='output')
                    cv2.imwrite(self.output_dir+"/ObserveOut_E_{}_I{}.png".format(epoch, i), observe_out*255)
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
                output_dict = self.model(image, target)
            # import pdb; pdb.set_trace()
            sematic_out = output_dict['sematic_out']
            pred = sematic_out.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(label.detach().cpu().numpy(), pred)

            if self.cfg.SOLVER.SHOW_IMAGE and i % 5 == 0:
                self.model.show_image(image, output_dict)

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
    # cfg.merge_from_list(['SOLVER.SEMATIC_ONLY', True])
    cfg.merge_from_list(['MULTI_RUN', True])
    cfg.merge_from_list(['MODEL.WEIGHT', 'run/mpc_v11.3_1/Every_5_model_Epoch_5.pth'])
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


# v11.3 解决了大片覆盖问题，但是存在pos list遍历导致init point过多
# v11.2 完成了validation，但是由于train val 不统一导致的大片覆盖问题下一版要解决
# v11.1 train 编码完成； 无validation; observe 完善
# v11 backbone 与 maskpyramid 分开， 直接利用好的backbone weight
# v10_5 instance 引入
# v10_5_resunet_v3 没有aspp
# v10_5_resunet_v2_sematic_1 层次加深到 feature map size 最小为4 
# v10.2: imshow 展示pyr num 与实际的instance target
# v10: Unet 下，不再多层输出，仅使用最终mask 通过放缩来匹配筛选各级mask

# v9: resnet 换成 Unet

# v8: 新的策略：
# 首先使用sematic segmentation 的 cross entropy 得出19个类的logits，
# 然后，对于可数的类，找未instance化的极值，初始化为instance，同时，instance 在计算时侵蚀（×0）掉该class logis
# 注意， 这个侵蚀， training时用target 侵蚀， testing 用instance 高于其它类别的值侵蚀
# 要有个while循环来把所有sematic 部分instance化
