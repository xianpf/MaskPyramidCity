import os
import numpy as np
import scipy.misc as m
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms
from dataloaders import city_transforms as tr

class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/path/to/datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/home/xianr/TurboRuns/cityscapes'     # foler that contains leftImg8bit/
        elif dataset == 'panoptic_cityscapes':
            return '/home/xianr/TurboRuns/cityscapes/newdivide'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/home/xianr/TurboRuns/coco'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError

class CityscapesSegmentation(data.Dataset):
    NUM_CLASSES = 19
    def __init__(self, cfg, root=Path.db_root_dir('cityscapes'), split="train"):
        self.root = root
        self.split = split
        self.cfg = cfg
        self.files = {}
        self.base_size = cfg.DATALOADER.BASE_SIZE
        self.crop_size = cfg.DATALOADER.CROP_SIZE

        self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)
        self.annotations_base = os.path.join(self.root, 'gtFine_trainvaltest', 'gtFine', self.split)

        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.png')
        # self.files[split] = self.files[split][100:100+10]

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-2],
                                os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png')

        _img = Image.open(img_path).convert('RGB')
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        _tmp = self.encode_segmap(_tmp)
        _target = Image.fromarray(_tmp)

        sample = {'image': _img, 'label': _target}

        if self.split == 'train':
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_ts(sample)

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.base_size, crop_size=self.crop_size, fill=255),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_ts(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

import matplotlib.pyplot as plt
def show_img(img_show):
    plt.figure(figsize=(6,4))
    img_show = (img_show - img_show.min())/(img_show.max() - img_show.min())
    plt.imshow(img_show)


class CityInstanceSegm(data.Dataset):
    def __init__(self, cfg, root=Path.db_root_dir('cityscapes'), split="train"):
        self.root = root
        self.split = split
        self.cfg = cfg
        self.files = {}
        self.base_size = cfg.DATALOADER.BASE_SIZE
        self.crop_size = cfg.DATALOADER.CROP_SIZE

        self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)
        self.annotations_base = os.path.join(self.root, 'gtFine_trainvaltest', 'gtFine', self.split)

        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.png')

        self.ignore_index = 255
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']
        self.class_map = dict(zip(self.valid_classes, range(cfg.DATALOADER.NUM_CLASSES)))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(self.annotations_base, img_path.split(os.sep)[-2],
                                os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png')
        ins_path = os.path.join(self.annotations_base, img_path.split(os.sep)[-2],
                                os.path.basename(img_path)[:-15] + 'gtFine_instanceIds.png')

        _img = Image.open(img_path).convert('RGB')
        # _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        _lbl = np.array(Image.open(lbl_path), dtype=np.uint8)
        _ins = np.array(Image.open(ins_path), dtype=np.uint8)
        _lbl = self.encode_segmap(_lbl)
        _tmp_cat = np.stack((_lbl, _ins), axis=-1)
        _target = Image.fromarray(_tmp_cat)

        sample = {'image': _img, 'label': _lbl, 'instance': _ins}

        if self.split == 'train':
            transed = self.transform_tr(sample)
        elif self.split == 'val':
            transed = self.transform_val(sample)
        elif self.split == 'test':
            transed = self.transform_ts(sample)

        # res = {}
        # target_sematic = transed['label'][..., 0]
        # ins_mask = transed['label'][..., 1]
        # target_masks = torch.cat([ins_mask[None] == lbl for lbl in ins_mask.unique()])
        # target_labels = torch.Tensor([(mask.float()*target_sematic).max() for mask in target_masks]).view(-1,1)

        # sample_res = {'image': _img, 'target_sematic': target_sematic, 
        #         'target_masks': target_masks, 'label': target_labels}

        # return sample_res

        target_sematic = transed['label']
        # import pdb; pdb.set_trace()
        # print('xxxxxxxxxxxxxx', np.unique(_lbl), target_sematic.unique(), '\n', (np.unique(_lbl) == target_sematic.unique().numpy()).all())
        
        return transed


    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
            
        return mask

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.ImageWraper(),
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.base_size, crop_size=self.crop_size, fill=self.ignore_index),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.ImageWraper(),
            tr.FixScaleCrop(crop_size=self.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_ts(self, sample):

        composed_transforms = transforms.Compose([
            tr.ImageWraper(),
            tr.FixedResize(size=self.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)


def cityIns_collate(data):
    import pdb; pdb.set_trace()