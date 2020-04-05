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
