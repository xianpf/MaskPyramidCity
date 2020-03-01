from dataloaders.cityscapes import CityscapesSegmentation, CityInstanceSegm
# from dataloaders import cityscapes, coco, combine_dbs, pascal, sbd
from torch.utils.data import DataLoader


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
        # train_set = CityscapesSegmentation(cfg, split='train')
        # val_set = CityscapesSegmentation(cfg, split='val')
        # test_set = CityscapesSegmentation(cfg, split='test')
        train_set = CityInstanceSegm(cfg, split='train')
        val_set = CityInstanceSegm(cfg, split='val')
        test_set = CityInstanceSegm(cfg, split='test')
        num_class = cfg.DATALOADER.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=cfg.DATALOADER.BATCH_SIZE_TRAIN, shuffle=True, 
            num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=cfg.DATALOADER.PIN_MEMORY)
        val_loader = DataLoader(val_set, batch_size=cfg.DATALOADER.BATCH_SIZE_VAL, shuffle=False, 
            num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=cfg.DATALOADER.PIN_MEMORY)
        test_loader = DataLoader(test_set, batch_size=cfg.DATALOADER.BATCH_SIZE_TEST, shuffle=False, 
            num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=cfg.DATALOADER.PIN_MEMORY)

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

