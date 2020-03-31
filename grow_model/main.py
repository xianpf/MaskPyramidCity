import torch, cv2, json, random
import matplotlib.pyplot as plt
from cityscapesscripts.helpers.annotation import Annotation
from cityscapesscripts.helpers.labels     import labels, name2label
from PIL import Image, ImageOps, ImageFilter
import PIL.ImageDraw as ImageDraw
import numpy as np
import torchvision.transforms as transform
from torch import nn

img_path = '/home/xianr/TurboRuns/cityscapes/leftImg8bit/train/aachen/aachen_000029_000019_leftImg8bit.png'
json_path = '/home/xianr/TurboRuns/cityscapes/gtFine/train/aachen/aachen_000029_000019_gtFine_polygons.json'
labled_img_path = '/home/xianr/TurboRuns/cityscapes/gtFine/train/aachen/aachen_000029_000019_gtFine_all_instances.png'
def get_sample(img_path, labled_img_path):
    img = Image.open(img_path).convert('RGB')
    inst = Image.open(labled_img_path)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        inst = inst.transpose(Image.FLIP_LEFT_RIGHT)
    crop_size = 513
    base_size = 2048

    w, h = img.size
    long_size = random.randint(int(base_size*0.5), int(base_size*2.0))
    if h > w:
        oh = long_size
        ow = int(1.0 * w * long_size / h + 0.5)
        short_size = ow
    else:
        ow = long_size
        oh = int(1.0 * h * long_size / w + 0.5)
        short_size = oh
    img = img.resize((ow, oh), Image.BILINEAR)
    inst = inst.resize((ow, oh), Image.NEAREST)

    # pad crop
    if short_size < crop_size:
        padh = crop_size - oh if oh < crop_size else 0
        padw = crop_size - ow if ow < crop_size else 0
        img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=-1)
        inst = ImageOps.expand(inst, border=(0, 0, padw, padh), fill=-1)
    w, h = img.size
    x1 = random.randint(0, w - crop_size)
    y1 = random.randint(0, h - crop_size)
    img = img.crop((x1, y1, x1+crop_size, y1+crop_size))
    inst = np.array(inst.crop((x1, y1, x1+crop_size, y1+crop_size)))

    res_inst = inst.copy()
    # assert the values
    values = sorted(np.unique(inst))
    for i, v in enumerate(values):
        res_inst[inst==v] = i
    inst = torch.from_numpy(res_inst).long()
    img__transform = transform.Compose([transform.ToTensor(),
                transform.Normalize([.485, .456, .406], [.229, .224, .225])])
    img =img__transform(img)
    return img, inst
def rend_on_image(image_np, masks_np, labels):
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
        # import pdb; pdb.set_trace()
        if label >= 19:
            label = label%19
        colors[label+1]+=1
        color = colors[label+1]/255.0
        masked[mask_idx[0], mask_idx[1], :] += alpha * color
        # import pdb; pdb.set_trace()
        contours, hierarchy = cv2.findContours(
            (masks_np == i).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        # masked = cv2.drawContours(masked, contours, -1, (1.,1.,1.), -1)
        masked = cv2.drawContours(masked, contours, -1, (1.,1.,1.), 1)

    return masked

# import pdb; pdb.set_trace()
for k in range(500):
    image, instance = get_sample(img_path, labled_img_path)
    image, instance = get_sample(img_path, labled_img_path)
    image_np = image.permute(1,2,0).detach().cpu().numpy()
    image_np = ((image_np - image_np.min()) / (image_np.max() - image_np.min()))[...,::-1]
    instance_np = instance.detach().cpu().numpy()
    masked_instance = rend_on_image(image_np, instance_np, range(len(np.unique(instance_np))))
    masked_show = np.hstack((image_np, masked_instance))
    cv2.imshow('one image', masked_show)
    cv2.waitKey(500)




class PickNode():
    def __init__(self, in_chn, out_chn):
        self.obs_kernel=[]
        self.obs_kernel_fix=[]  # 25%
        self.obs_kernel_running=[]  # 75%
        self.conv = nn.Conv2d(in_chn, out_chn, 3)






