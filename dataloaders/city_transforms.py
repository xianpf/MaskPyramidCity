import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        # import pdb; pdb.set_trace()
        # img = sample['image']
        # mask = sample['label']
        # img = np.array(img).astype(np.float32)
        # mask = np.array(mask).astype(np.float32)
        # img /= 255.0
        # img -= self.mean
        # img /= self.std

        # return {'image': img,
        #         'label': mask}

        ret = sample.copy()
        if random.random() < 0.5:
            for k, v in ret.items():
                v1 = np.array(v).astype(np.float32)
                ret[k] = (v1 / 255.0 - self.mean) / self.std if k in ['image'] else v1


        return ret

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # import pdb; pdb.set_trace()
        # # swap color axis because
        # # numpy image: H x W x C
        # # torch image: C X H X W
        # img = sample['image']
        # mask = sample['label']
        # img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        # mask = np.array(mask).astype(np.float32)

        # img = torch.from_numpy(img).float()
        # mask = torch.from_numpy(mask).float()

        # return {'image': img,
        #         'label': mask}

        ret = sample.copy()
        for k, v in sample.items():
            if k in ['image']:
                v1 = np.array(v).astype(np.float32).transpose((2, 0, 1))
            elif k in ['label', 'instance']:
                v1 = np.array(v).astype(np.float32)
            else:
                import pdb; pdb.set_trace()
            ret[k] = torch.from_numpy(v1).float()
        
        return ret

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        # import pdb; pdb.set_trace()
        # img = sample['image']
        # mask = sample['label']
        # if random.random() < 0.5:
        #     img = img.transpose(Image.FLIP_LEFT_RIGHT)
        #     mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # return {'image': img,
        #         'label': mask}

        ret = sample.copy()
        if random.random() < 0.5:
            for k, v in sample.items():
                ret[k] = v.transpose(Image.FLIP_LEFT_RIGHT)

        return ret

class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        import pdb; pdb.set_trace()
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}

class RandomGaussianBlur(object):
    def __call__(self, sample):
        # import pdb; pdb.set_trace()
        # img = sample['image']
        # mask = sample['label']
        # if random.random() < 0.5:
        #     img = img.filter(ImageFilter.GaussianBlur(
        #         radius=random.random()))

        # return {'image': img,
        #         'label': mask}

        ret = sample.copy()
        if random.random() < 0.5:
            for k, v in ret.items():
                if k in ['image']:
                    ret[k] = v.transpose(Image.FLIP_LEFT_RIGHT)

        return ret

class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        # import pdb; pdb.set_trace()
        img = sample['image']
        # mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)

        # img = img.resize((ow, oh), Image.BILINEAR)
        # mask = mask.resize((ow, oh), Image.NEAREST)
        ret = dict()
        for k, v in sample.items():
            if k in ['image',]:
                ret[k] = v.resize((ow, oh), Image.BILINEAR)
            elif k in ['label', 'instance']:
                ret[k] = v.resize((ow, oh), Image.NEAREST)
            else:
                import pdb; pdb.set_trace()

        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            for k, v in ret.items():
                if k in ['image',]:
                    ret[k] = ImageOps.expand(v, border=(0, 0, padw, padh), fill=0)
                elif k in ['label', 'instance']:
                    ret[k] = ImageOps.expand(v, border=(0, 0, padw, padh), fill=self.fill)
                else:
                    import pdb; pdb.set_trace()
            # img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            # mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        # img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        # mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        for k, v in ret.items():
            if k in ['image',]:
                ret[k] = v.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
            elif k in ['label', 'instance']:
                ret[k] = v.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
            else:
                import pdb; pdb.set_trace()

        # target_sematic = np.array(ret['label'])
        # if target_sematic.max() > 18:
        #     # print('ttttttttttt', np.unique(target_sematic), 'lllllllll', np.unique(np.array(sample['label'])))
        #     import pdb; pdb.set_trace()

        # return {'image': img,
        #         'label': mask}
        return ret

class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        # import pdb; pdb.set_trace()
        img = sample['image']
        # mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        # img = img.resize((ow, oh), Image.BILINEAR)
        # mask = mask.resize((ow, oh), Image.NEAREST)
        ret = dict()
        for k, v in sample.items():
            if k in ['image',]:
                ret[k] = v.resize((ow, oh), Image.BILINEAR)
            elif k in ['label', 'instance']:
                ret[k] = v.resize((ow, oh), Image.NEAREST)
            else:
                import pdb; pdb.set_trace()

        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        # img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        # mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        # return {'image': img,
        #         'label': mask}
        for k, v in ret.items():
            if k in ['image',]:
                ret[k] = v.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
            elif k in ['label', 'instance']:
                ret[k] = v.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
            else:
                import pdb; pdb.set_trace()

        return ret


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        import pdb; pdb.set_trace()
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask}

class ImageWraper(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        ret = dict()
        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                ret[k] = Image.fromarray(v)
            elif isinstance(v, Image.Image):
                ret[k] = v
            else:
                import pdb; pdb.set_trace()

        return ret
