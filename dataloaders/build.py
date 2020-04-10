from dataloaders.cityscapes import CitySegmentation
from torch.utils.data import DataLoader
import torchvision.transforms as transform
import torch
import torch.distributed as dist
from dataloaders import coco_transforms as COCOT


def make_data_loader(cfg):
    if cfg.DATALOADER.DATASET == 'pascal':
        # train_set = pascal.VOCSegmentation(args, split='train')
        # val_set = pascal.VOCSegmentation(args, split='val')
        # if args.use_sbd:
        #     sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
        #     train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        # num_class = train_set.NUM_CLASSES
        # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        # val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        # test_loader = None

        # return train_loader, val_loader, test_loader, num_class
        import pdb; pdb.set_trace()

    elif cfg.DATALOADER.DATASET == 'cityscapes':
        input_transform = transform.Compose([transform.ToTensor(),
            transform.Normalize([.485, .456, .406], [.229, .224, .225])])
        train_set = CitySegmentation(cfg, split='train', transform=input_transform)
        val_set = CitySegmentation(cfg, split='val', transform=input_transform)
        test_set = CitySegmentation(cfg, split='test', transform=input_transform)
        num_class = cfg.DATALOADER.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=cfg.DATALOADER.BATCH_SIZE_TRAIN, shuffle=True, 
            num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=cfg.DATALOADER.PIN_MEMORY, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=cfg.DATALOADER.BATCH_SIZE_VAL, shuffle=False, 
            num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=cfg.DATALOADER.PIN_MEMORY, drop_last=True)
        test_loader = DataLoader(test_set, batch_size=cfg.DATALOADER.BATCH_SIZE_TEST, shuffle=False, 
            num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=cfg.DATALOADER.PIN_MEMORY, drop_last=True)

        return train_loader, val_loader, test_loader, num_class

    # elif cfg.DATALOADER.DATASET == 'coco':
        # train_set = coco.COCOSegmentation(args, split='train')
        # val_set = coco.COCOSegmentation(args, split='val')
        # num_class = train_set.NUM_CLASSES
        # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        # val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        # test_loader = None
        # return train_loader, val_loader, test_loader, num_class
        import pdb; pdb.set_trace()
    elif cfg.DATALOADER.DATASET == 'coco':
        from dataloaders.coco import COCODataset
        
        remove_images_without_annotations = True
        num_workers = cfg.DATALOADER.NUM_WORKERS
        num_class = cfg.DATALOADER.NUM_CLASSES
        is_distributed = False
        shuffle = False if not is_distributed else True
        aspect_grouping = [1]   # if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []
        # images_per_gpu = images_per_batch // num_gpus
        # num_iters
        # start_iter
        # data_loader = DataLoader(dataset, num_workers=num_workers, batch_sampler=batch_sampler, collate_fn=collator)
        size_divisibility = 32      # cfg.DATALOADER.SIZE_DIVISIBILITY
        collator = BatchCollator(size_divisibility)
        
        train_root = 'dataloaders/coco/val2014'
        train_set_ann_file =  "dataloaders/coco/annotations/instances_val2014.json"
        train_transforms = build_transforms(cfg, is_train=True)
        train_set = COCODataset(train_set_ann_file, train_root, remove_images_without_annotations, transforms=train_transforms)
        train_images_per_gpu = cfg.DATALOADER.BATCH_SIZE_TRAIN
        train_sampler = make_data_sampler(train_set, shuffle, is_distributed)
        train_batch_sampler = make_batch_data_sampler(train_set, train_sampler, aspect_grouping, train_images_per_gpu)
        train_loader = DataLoader(train_set, num_workers=num_workers, collate_fn=collator, batch_sampler=train_batch_sampler)

        val_root = 'dataloaders/coco/val2014'
        # val_set_ann_file =  "dataloaders/coco/annotations/instances_minival2014.json"
        # val_set_ann_file =  "dataloaders/coco/annotations/instances_val2014.json"
        val_set_ann_file =  "dataloaders/coco/annotations/instances_microval2014.json"
        val_transforms = build_transforms(cfg, is_train=False)
        val_set = COCODataset(val_set_ann_file, val_root, remove_images_without_annotations, transforms=val_transforms)
        val_images_per_gpu = cfg.DATALOADER.BATCH_SIZE_VAL
        val_sampler = make_data_sampler(val_set, shuffle, is_distributed)
        val_batch_sampler = make_batch_data_sampler(val_set, val_sampler, aspect_grouping, val_images_per_gpu)
        val_loader = DataLoader(val_set, num_workers=num_workers, collate_fn=collator, batch_sampler=val_batch_sampler)

        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError


class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors, image_sizes):
        """
        Arguments:
            tensors (tensor)
            image_sizes (list[tuple[int, int]])
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes)


def to_image_list(tensors, size_divisible=0):
    """
    tensors can be an ImageList, a torch.Tensor or
    an iterable of Tensors. It can't be a numpy array.
    When tensors is an iterable of Tensors, it pads
    the Tensors with zeros so that they have the same
    shape
    """
    if isinstance(tensors, torch.Tensor) and size_divisible > 0:
        tensors = [tensors]

    if isinstance(tensors, ImageList):
        return tensors
    elif isinstance(tensors, torch.Tensor):
        # single tensor shape can be inferred
        if tensors.dim() == 3:
            tensors = tensors[None]
        assert tensors.dim() == 4
        image_sizes = [tensor.shape[-2:] for tensor in tensors]
        return ImageList(tensors, image_sizes)
    elif isinstance(tensors, (tuple, list)):
        max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))

        # TODO Ideally, just remove this and let me model handle arbitrary
        # input sizs
        if size_divisible > 0:
            import math

            stride = size_divisible
            max_size = list(max_size)
            max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
            max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
            max_size = tuple(max_size)

        batch_shape = (len(tensors),) + max_size
        batched_imgs = tensors[0].new(*batch_shape).zero_()
        for img, pad_img in zip(tensors, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        image_sizes = [im.shape[-2:] for im in tensors]

        return ImageList(batched_imgs, image_sizes)
    else:
        raise TypeError("Unsupported type for to_image_list: {}".format(type(tensors)))


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets = transposed_batch[1]
        img_ids = transposed_batch[2]
        return images, targets, img_ids

from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import Sampler
import copy
import bisect
import itertools


class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset : offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class GroupedBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that elements from the same group should appear in groups of batch_size.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.

    Arguments:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_uneven (bool): If ``True``, the sampler will drop the batches whose
            size is less than ``batch_size``

    """

    def __init__(self, sampler, group_ids, batch_size, drop_uneven=False):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler
        self.group_ids = torch.as_tensor(group_ids)
        assert self.group_ids.dim() == 1
        self.batch_size = batch_size
        self.drop_uneven = drop_uneven

        self.groups = torch.unique(self.group_ids).sort(0)[0]

        self._can_reuse_batches = False

    def _prepare_batches(self):
        dataset_size = len(self.group_ids)
        # get the sampled indices from the sampler
        sampled_ids = torch.as_tensor(list(self.sampler))
        # potentially not all elements of the dataset were sampled
        # by the sampler (e.g., DistributedSampler).
        # construct a tensor which contains -1 if the element was
        # not sampled, and a non-negative number indicating the
        # order where the element was sampled.
        # for example. if sampled_ids = [3, 1] and dataset_size = 5,
        # the order is [-1, 1, -1, 0, -1]
        order = torch.full((dataset_size,), -1, dtype=torch.int64)
        order[sampled_ids] = torch.arange(len(sampled_ids))

        # get a mask with the elements that were sampled
        mask = order >= 0

        # find the elements that belong to each individual cluster
        clusters = [(self.group_ids == i) & mask for i in self.groups]
        # get relative order of the elements inside each cluster
        # that follows the order from the sampler
        relative_order = [order[cluster] for cluster in clusters]
        # with the relative order, find the absolute order in the
        # sampled space
        permutation_ids = [s[s.sort()[1]] for s in relative_order]
        # permute each cluster so that they follow the order from
        # the sampler
        permuted_clusters = [sampled_ids[idx] for idx in permutation_ids]

        # splits each cluster in batch_size, and merge as a list of tensors
        splits = [c.split(self.batch_size) for c in permuted_clusters]
        merged = tuple(itertools.chain.from_iterable(splits))

        # now each batch internally has the right order, but
        # they are grouped by clusters. Find the permutation between
        # different batches that brings them as close as possible to
        # the order that we have in the sampler. For that, we will consider the
        # ordering as coming from the first element of each batch, and sort
        # correspondingly
        first_element_of_batch = [t[0].item() for t in merged]
        # get and inverse mapping from sampled indices and the position where
        # they occur (as returned by the sampler)
        inv_sampled_ids_map = {v: k for k, v in enumerate(sampled_ids.tolist())}
        # from the first element in each batch, get a relative ordering
        first_index_of_batch = torch.as_tensor(
            [inv_sampled_ids_map[s] for s in first_element_of_batch]
        )

        # permute the batches so that they approximately follow the order
        # from the sampler
        permutation_order = first_index_of_batch.sort(0)[1].tolist()
        # finally, permute the batches
        batches = [merged[i].tolist() for i in permutation_order]

        if self.drop_uneven:
            kept = []
            for batch in batches:
                if len(batch) == self.batch_size:
                    kept.append(batch)
            batches = kept
        return batches

    def __iter__(self):
        if self._can_reuse_batches:
            batches = self._batches
            self._can_reuse_batches = False
        else:
            batches = self._prepare_batches()
        self._batches = batches
        return iter(batches)

    def __len__(self):
        if not hasattr(self, "_batches"):
            self._batches = self._prepare_batches()
            self._can_reuse_batches = True
        return len(self._batches)


class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        # return samplers.DistributedSampler(dataset, shuffle=shuffle)
        return DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(
    dataset, sampler, aspect_grouping, images_per_batch, num_iters=None, start_iter=0
):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        # batch_sampler = samplers.GroupedBatchSampler(
        batch_sampler = GroupedBatchSampler(
            sampler, group_ids, images_per_batch, drop_uneven=False
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=False
        )
    if num_iters is not None:
        # batch_sampler = samplers.IterationBasedBatchSampler(
        batch_sampler = IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def build_transforms_with_cfg(cfg, is_train=True):
    if is_train:
        if cfg.INPUT.MIN_SIZE_RANGE_TRAIN[0] == -1:
            # min_size = cfg.INPUT.MIN_SIZE_TRAIN
            min_size = cfg.INPUT.MIN_SIZE_TRAIN
        else:
            assert len(cfg.INPUT.MIN_SIZE_RANGE_TRAIN) == 2, \
                "MIN_SIZE_RANGE_TRAIN must have two elements (lower bound, upper bound)"
            min_size = list(range(
                cfg.INPUT.MIN_SIZE_RANGE_TRAIN[0],
                cfg.INPUT.MIN_SIZE_RANGE_TRAIN[1] + 1
            ))
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = COCOT.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )

    transform = COCOT.Compose(
        [
            COCOT.Resize(min_size, max_size),
            COCOT.RandomHorizontalFlip(flip_prob),
            COCOT.ToTensor(),
            normalize_transform,
        ]
    )
    return transform

def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = [800]
        max_size = 1333
        flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
    else:
        min_size = 800
        max_size = 1333
        flip_prob = 0

    to_bgr255 = True
    normalize_transform = COCOT.Normalize(
        mean=(102.9801,115.9465,122.7717), std=(1.0,1.0,1.0), to_bgr255=to_bgr255
    )

    transform = COCOT.Compose(
        [
            COCOT.Resize(min_size, max_size),
            COCOT.RandomHorizontalFlip(flip_prob),
            COCOT.ToTensor(),
            normalize_transform,
        ]
    )
    return transform


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()
