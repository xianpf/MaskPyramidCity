from dataloaders.cityscapes import CitySegmentation
from torch.utils.data import DataLoader
import torchvision.transforms as transform


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

    elif cfg.DATALOADER.DATASET == 'coco':
        # train_set = coco.COCOSegmentation(args, split='train')
        # val_set = coco.COCOSegmentation(args, split='val')
        # num_class = train_set.NUM_CLASSES
        # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        # val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        # test_loader = None
        # return train_loader, val_loader, test_loader, num_class
        import pdb; pdb.set_trace()

    else:
        raise NotImplementedError

