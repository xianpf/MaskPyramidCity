###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2018
###########################################################################

import os
import sys
import random
import numpy as np
from tqdm import tqdm, trange
from PIL import Image, ImageOps, ImageFilter

import torch
import torch.utils.data as data
import torchvision.transforms as transform

class BaseDataset(data.Dataset):
    def __init__(self, root, split, mode=None, transform=None, 
                 target_transform=None, base_size=520, crop_size=480):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size
        if self.mode == 'train':
            print('BaseDataset: base_size {}, crop_size {}'. \
                format(base_size, crop_size))

    def __getitem__(self, index):
        raise NotImplementedError

    # @property
    # def num_class(self):
    #     return self.NUM_CLASS

    @property
    def pred_offset(self):
        raise NotImplementedError

    def make_pred(self, x):
        return x + self.pred_offset

    def _val_sync_transform(self, img, mask, inst):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        inst = inst.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1+outsize, y1+outsize))
        mask = mask.crop((x1, y1, x1+outsize, y1+outsize))
        inst = inst.crop((x1, y1, x1+outsize, y1+outsize))
        # final transform
        return img, self._mask_transform(mask), self._inst_transform(inst)

    def _sync_transform(self, img, mask, inst):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            inst = inst.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        w, h = img.size
        long_size = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        inst = inst.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=-1)
            inst = ImageOps.expand(inst, border=(0, 0, padw, padh), fill=-1)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1+crop_size, y1+crop_size))
        mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))
        inst = inst.crop((x1, y1, x1+crop_size, y1+crop_size))
        # final transform
        return img, self._mask_transform(mask), self._inst_transform(inst)

    def _mask_transform(self, mask):
        return torch.from_numpy(np.array(mask)).long()

    def _inst_transform(self, mask):
        import pdb; pdb.set_trace()

class CitySegmentation(BaseDataset):
    NUM_CLASS = 19
    # def __init__(self, root=os.path.expanduser('~/.encoding/data'), split='train',
    def __init__(self, cfg, split='train',
                 mode=None, transform=None, target_transform=None, **kwargs):
        self.cfg = cfg
        self.root = cfg.DATALOADER.DATASET_PATH
        super(CitySegmentation, self).__init__(
            self.root, split, mode, transform, target_transform, base_size=cfg.DATALOADER.BASE_SIZE, 
            crop_size=cfg.DATALOADER.CROP_SIZE, **kwargs)
        #self.root = os.path.join(root, self.BASE_DIR)
        self.images, self.mask_paths = get_city_pairs(self.root, self.split)
        self.inst_paths = [mpath.replace('_gtFine_labelIds.png','_gtFine_instanceIds.png') for mpath in self.mask_paths]
        assert (len(self.images) == len(self.mask_paths) == len(self.inst_paths))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of: \
                " + self.root + "\n")
        self._indices = np.array(range(-1, 19))
        self._classes = np.array([0, 7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                                  23, 24, 25, 26, 27, 28, 31, 32, 33])
        self._key = np.array([-1, -1, -1, -1, -1, -1,
                              -1, -1,  0,  1, -1, -1, 
                              2,   3,  4, -1, -1, -1,
                              5,  -1,  6,  7,  8,  9,
                              10, 11, 12, 13, 14, 15,
                              -1, -1, 16, 17, 18])
        self._mapping = np.array(range(-1, len(self._key)-1)).astype('int32')
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']

    def _class_to_index(self, mask):
        # assert the values
        values = np.unique(mask)
        for i in range(len(values)):
            assert(values[i] in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def _inst_reindex(self, inst):
        # assert the values
        values = sorted(np.unique(inst))
        for i in range(len(values)):
            inst[inst==values[i]] = i
        return inst

    def _preprocess(self, mask_file):
        if os.path.exists(mask_file):
            masks = torch.load(mask_file)
            return masks
        masks = []
        print("Preprocessing mask, this will take a while." + \
            "But don't worry, it only run once for each split.")
        tbar = tqdm(self.mask_paths)
        for fname in tbar:
            tbar.set_description("Preprocessing masks {}".format(fname))
            mask = Image.fromarray(self._class_to_index(
                np.array(Image.open(fname))).astype('int8'))
            masks.append(mask)
        torch.save(masks, mask_file)
        return masks

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        #mask = self.masks[index]
        mask = Image.open(self.mask_paths[index])
        inst = Image.open(self.inst_paths[index])
        # synchrosized transform
        if self.mode == 'train':
            img, mask, inst = self._sync_transform(img, mask, inst)
        elif self.mode == 'val':
            img, mask, inst = self._val_sync_transform(img, mask, inst)
        else:
            assert self.mode == 'testval'
            import pdb; pdb.set_trace()
            mask = self._mask_transform(mask)
            inst = self._inst_transform(inst)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
            import pdb; pdb.set_trace()
            inst = self.target_transform(inst)
        return img, mask, inst

    def _mask_transform(self, mask):
        #target = np.array(mask).astype('int32') - 1
        target = self._class_to_index(np.array(mask).astype('int32'))
        return torch.from_numpy(target).long()

    def _inst_transform(self, inst):
        target = self._inst_reindex(np.array(inst))
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.images)

    def make_pred(self, mask):
        values = np.unique(mask)
        for i in range(len(values)):
            assert(values[i] in self._indices)
        index = np.digitize(mask.ravel(), self._indices, right=True)
        return self._classes[index].reshape(mask.shape)


def get_city_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []  
        mask_paths = []  
        for root, directories, files in os.walk(img_folder):
            for filename in files:
                if filename.endswith(".png"):
                    imgpath = os.path.join(root, filename)
                    foldername = os.path.basename(os.path.dirname(imgpath))
                    maskname = filename.replace('leftImg8bit','gtFine_labelIds')
                    maskpath = os.path.join(mask_folder, foldername, maskname)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        print('cannot find the mask or image:', imgpath, maskpath)
        print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths

    if split == 'train' or split == 'val' or split == 'test':
        img_folder = os.path.join(folder, 'leftImg8bit/' + split)
        mask_folder = os.path.join(folder, 'gtFine/'+ split)
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        return img_paths, mask_paths
    else:
        assert split == 'trainval'
        print('trainval set')
        train_img_folder = os.path.join(folder, 'leftImg8bit/train')
        train_mask_folder = os.path.join(folder, 'gtFine/train')
        val_img_folder = os.path.join(folder, 'leftImg8bit/val')
        val_mask_folder = os.path.join(folder, 'gtFine/val')
        train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
        val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
        img_paths = train_img_paths + val_img_paths
        mask_paths = train_mask_paths + val_mask_paths
    return img_paths, mask_paths