import sys, os
sys.path.insert(0, os.path.abspath(__file__+'/../..'))
import argparse
import numpy as np
import os
import torch
from msiip_resunet.mypath import Path
from msiip_resunet.data_loader import make_data_loader
from msiip_resunet.resunet_aspp import ResUnetAspp
from msiip_resunet.saver import Saver
from msiip_resunet.loss import SegmentationLosses
from msiip_resunet.metrics import Evaluator
from msiip_resunet.lr_scheduler import LR_Scheduler
from msiip_resunet.summaries import TensorboardSummary
from msiip_resunet.replicate import patch_replication_callback
from tqdm import tqdm
# from msiip_resunet.lookahead import Lookahead
# from msiip_resunet.lookaround import Lookaround


class Trainer(object):
    def __init__(self, args):
        self.args = args
        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
        # model = MultiScaleInitCNN(num_classes=self.nclass)
        model = ResUnetAspp(num_classes=self.nclass)

        train_params = [{'params': model.parameters(), 'lr': args.lr}]

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)
        # optimizer = torch.optim.Adam(train_params)
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
        # optimizer = Lookahead(optimizer)
        self.criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer

        self.evaluator = Evaluator(self.nclass)
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))
        # Using cuda
        if args.cuda:
            # self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            # patch_replication_callback(self.model)
            self.model = self.model.cuda()
        
        # # unet_weights = torch.load('/home/xianr/TurboRuns/Pytorch-UNet/MODEL.pth')
        # unet_weights = torch.load('/home/xianr/TurboRuns/msiip/multiscaleinit/UNETMODEL.pth')
        # # import pdb; pdb.set_trace()
        # # load unet weights
        # unet_weights['inc.conv.conv.0.weight'] = torch.cat((unet_weights['inc.conv.conv.0.weight'], \
        #     model.unet.inc.conv.conv[0].weight[:,3:,...]*1e-3), 1)
        # unet_weights['outc.conv.weight'] = model.unet.outc.conv.weight
        # unet_weights['outc.conv.bias'] = model.unet.outc.conv.bias
        # model.unet.load_state_dict(unet_weights)

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            import pdb; pdb.set_trace()
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))


    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            # import pdb; pdb.set_trace()
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            # tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            tbar.set_description('Train loss: %.3f|lr %.3f' % (train_loss / (i + 1), self.optimizer.param_groups[0]['lr']))
        
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            # import pdb; pdb.set_trace()
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                # 'state_dict': self.model.module.state_dict(),
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)
 


def main():
    if True:
        parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
        parser.add_argument('--dataset', type=str, default='cityscapes',
                            choices=['pascal', 'coco', 'cityscapes'],
                            help='dataset name (default: pascal)')
        parser.add_argument('--workers', type=int, default=4,
                            metavar='N', help='dataloader threads')
        parser.add_argument('--base-size', type=int, default=513,
                            help='base image size')
        parser.add_argument('--crop-size', type=int, default=513,
                            help='crop image size')
        parser.add_argument('--loss-type', type=str, default='ce',
                            choices=['ce', 'focal'],
                            help='loss func type (default: ce)')
        # training hyper params
        parser.add_argument('--epochs', type=int, default=None, metavar='N',
                            help='number of epochs to train (default: auto)')
        parser.add_argument('--start_epoch', type=int, default=0,
                            metavar='N', help='start epochs (default:0)')
        parser.add_argument('--batch-size', type=int, default=None,
                            metavar='N', help='input batch size for \
                                    training (default: auto)')
        # optimizer params
        parser.add_argument('--lr', type=float, default=None, metavar='LR',
                            help='learning rate (default: auto)')
        parser.add_argument('--lr-scheduler', type=str, default='poly',
                            choices=['poly', 'step', 'cos'],
                            help='lr scheduler mode: (default: poly)')
        parser.add_argument('--momentum', type=float, default=0.9,
                            metavar='M', help='momentum (default: 0.9)')
        parser.add_argument('--weight-decay', type=float, default=5e-4,
                            metavar='M', help='w-decay (default: 5e-4)')
        parser.add_argument('--nesterov', action='store_true', default=False,
                            help='whether use nesterov (default: False)')
        # cuda, seed and logging
        parser.add_argument('--no-cuda', action='store_true', default=
                            False, help='disables CUDA training')
        # checking point
        parser.add_argument('--resume', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--checkname', type=str, default=None,
                            help='set the checkpoint name')
        # evaluation option
        parser.add_argument('--eval-interval', type=int, default=1,
                            help='evaluation interval (default: 1)')
        parser.add_argument('--no-val', action='store_true', default=False,
                            help='skip validation during training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()


    if args.epochs is None:
        epoches = {
            # 'coco': 30,
            'coco': 50,
            'cityscapes': 200,
            'pascal': 50,
        }
        args.epochs = epoches[args.dataset.lower()]
    if args.batch_size is None:
        args.batch_size = 4

    # if args.test_batch_size is None:
    #     args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
        }
        args.lr = lrs[args.dataset.lower()]

    if args.checkname is None:
        args.checkname = 'resunet-aspp'
    print(args)
    # torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        # import pdb; pdb.set_trace()
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
        # if True and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)
    trainer.writer.close()


if __name__ == "__main__":
   main()