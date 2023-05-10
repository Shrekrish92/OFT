#!/Users/shreyakrishna/miniforge3/envs/oft_env/bin/python3.8

import os
import time
#import pickle5 as pickle
import yaml
import math
import numpy as np
import matplotlib
import torchmetrics
#matplotlib.use('Agg', warn=False)
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from datetime import datetime, timedelta
from argparse import ArgumentParser

from collections import defaultdict
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

from tqdm import tqdm

import oft
from oft import OftNet, KittiObjectDataset, MetricDict, masked_l1_loss, heatmap_loss, ObjectEncoder
from oft.matrix import *

current_time = datetime.now()

def train(args, dataloader, model, encoder, optimizer, summary, epoch):
    
    print('\n==> Training on {} minibatches'.format(len(dataloader)))
    model.train()
    epoch_loss = oft.MetricDict()

    t = time.time()
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
    for i, (_, image, calib, objects, grid) in enumerate(tqdm(dataloader)):
        
        # Move tensors to GPU
        if args.gpuid != "":
            model.to(f"cuda:{args.gpuid}")
            image, calib, grid = image.to(f"cuda:{args.gpuid}"), calib.to(f"cuda:{args.gpuid}"), grid.to(f"cuda:{args.gpuid}")


        # Run network forwards
        pred_encoded = model(image, calib, grid)
        
       # pred = [t[0].cpu() for t in pred_encoded]
        #detections = encoder.decode(*pred, grid.cpu())
        
        # Encode ground truth objects
        gt_encoded = encoder.encode_batch(objects, grid)
        print("TRAIN!!!")
        print(objects)
        #print(gt_encoded[2][1][1])

        # Compute losses
        loss, loss_dict = compute_loss(
            pred_encoded, gt_encoded, args.loss_weights)
        if float(loss) != float(loss):
            raise RuntimeError('Loss diverged :(')      
        epoch_loss += loss_dict

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        #lr_scheduler.step()
        
        #get IoU
        #iou = 0
        '''for i in range(min(len(detections),len(objects))):
            d = detections[i]
            o = objects[i]
            corners_3d_ground  = get_3d_box(objects[i][2], objects[i][3], objects[i][1]) 
            corners_3d_predict = get_3d_box(detections[i][2].detach().numpy(), detections[i][3].detach().numpy(), detections[i][1].detach().numpy())
            (IOU_3d, IOU_2d)=box3d_iou(corners_3d_predict,corners_3d_ground)
            iou += IOU_3d
        iou /= i'''
             
        # Print summary
        if i % args.print_iter == 0 and i != 0:
            batch_time = (time.time() - t) / (1 if i == 0 else args.print_iter)
            eta = ((args.epochs - epoch + 1) * len(dataloader) - i) * batch_time

            s = '[{:4d}/{:4d}] batch_time: {:.2f}s eta: {:s} loss: '.format(
                i, len(dataloader), batch_time, 
                str(timedelta(seconds=int(eta))))
            for k, v in loss_dict.items():
                s += '{}: {:.2e} '.format(k, v)
            print(s)
            t = time.time()
        
        # Visualize predictions
        if i % args.vis_iter == 0:

            # Visualize image
            summary.add_image('train/image', visualize_image(image), epoch)

            # Visualize scores
            summary.add_figure('train/score', 
                visualize_score(pred_encoded[0], gt_encoded[0], grid), epoch)
            
            # Decode predictions
            preds = encoder.decode_batch(*pred_encoded, grid)

            # Visualise bounding boxes
            summary.add_figure('train/bboxes',
                visualise_bboxes(image, calib, objects, preds), epoch)
        
        # TODO decode and save results        

    # Print epoch summary and save results
    print('==> Training epoch complete')
    #print(iou)
    for key, value in epoch_loss.mean.items():
        print('{:8s}: {:.4e}'.format(key, value))
        summary.add_scalar('train/loss/{}'.format(key), value, epoch)

        


def validate(args, dataloader, model, encoder, summary, epoch):
    
    print('\n==> Validating on {} minibatches\n'.format(len(dataloader)))
    
    model.eval()
    epoch_loss = MetricDict()
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    

    for i, (_, image, calib, objects, grid) in enumerate(tqdm(dataloader)):

        # Move tensors to GPU
        #if len(args.gpu) > 0:
        if args.gpuid != "":
            model.to(f"cuda:{args.gpuid}")
            image, calib, grid = image.to(f"cuda:{args.gpuid}"), calib.to(f"cuda:{args.gpuid}"), grid.to(f"cuda:{args.gpuid}")

        with torch.no_grad():

            # Run network forwards
            pred_encoded = model(image, calib, grid)

            # Encode ground truth objects
            gt_encoded = encoder.encode_batch(objects, grid)

            # Compute losses
            _, loss_dict = compute_loss(
                pred_encoded, gt_encoded, args.loss_weights)       
            epoch_loss += loss_dict
            
            # Decode predictions
            preds = encoder.decode_batch(*pred_encoded, grid)
        
        # Visualize predictions
        if i % args.vis_iter == 0:

            # Visualize image
            summary.add_image('val/image', visualize_image(image), epoch)

            # Visualize scores
            summary.add_figure('val/score', 
                visualize_score(pred_encoded[0], gt_encoded[0], grid), epoch)
            
            # Visualise bounding boxes
            summary.add_figure('val/bboxes',
                visualise_bboxes(image, calib, objects, preds), epoch)
    print(epoch_loss)    
        

    # TODO evaluate
    
    print('\n==> Validation epoch complete')
    for key, value in epoch_loss.mean.items():
        print('{:8s}: {:.4e}'.format(key, value))
        summary.add_scalar('val/loss/{}'.format(key), value, epoch)
    
    return epoch_loss['total']

def compute_loss(pred_encoded, gt_encoded, loss_weights=[1., 1., 1., 1.]):

    # Expand tuples
    score, pos_offsets, dim_offsets, ang_offsets = pred_encoded
    heatmaps, gt_pos_offsets, gt_dim_offsets, gt_ang_offsets, mask = gt_encoded
    score_weight, pos_weight, dim_weight, ang_weight = loss_weights
    #print("!!!!!MASK!!!!!")
    #print(mask.unsqueeze(2))
    # Compute losses
    score_loss = heatmap_loss(score, heatmaps)
    pos_loss = masked_l1_loss(pos_offsets, gt_pos_offsets, mask.unsqueeze(2))
    dim_loss = masked_l1_loss(dim_offsets, gt_dim_offsets, mask.unsqueeze(2))
    ang_loss = masked_l1_loss(ang_offsets, gt_ang_offsets, mask.unsqueeze(2))

    # Combine loss
    total_loss = score_loss * score_weight + pos_loss * pos_weight \
            + dim_loss * dim_weight + ang_loss * ang_weight
    
    # Store scalar losses in a dictionary
    loss_dict = {
        'score' : float(score_loss), 'position' : float(pos_loss),
        'dimension' : float(dim_loss), 'angle' : float(ang_loss),
        'total' : float(total_loss) 
    }

    return total_loss, loss_dict


def visualize_image(image):
    return image[0].cpu().detach()

def visualize_score(scores, heatmaps, grid):

    # Visualize score
    fig_score = plt.figure(num='score', figsize=(8, 6))
    fig_score.clear()

    oft.vis_score(scores[0, 0], grid[0], ax=plt.subplot(121))
    oft.vis_score(heatmaps[0, 0], grid[0], ax=plt.subplot(122))

    return fig_score

def visualise_bboxes(image, calib, objects, preds):

    fig = plt.figure(num='bbox', figsize=(8, 6))
    fig.clear()
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    oft.visualize_objects(image[0], calib[0], preds[0], ax=ax1)
    ax1.set_title('Predictions')

    oft.visualize_objects(image[0], calib[0], objects[0], ax=ax2)
    ax2.set_title('Ground truth')
    #plt.savefig("viz.png")
    return fig



def parse_args():
    parser = ArgumentParser()

    # Data options
    parser.add_argument('--root', type=str, default='oft/data/kitti',
                        help='root directory of the KITTI dataset')
    parser.add_argument('--grid-size', type=float, nargs=2, default=(80., 80.),
                        help='width and depth of validation grid, in meters')
    parser.add_argument('--train-grid-size', type=int, nargs=2, 
                        default=(120, 120),
                        help='width and depth of training grid, in pixels')
    parser.add_argument('--grid-jitter', type=float, nargs=3, 
                        default=[.25, .5, .25],
                        help='magn. of random noise applied to grid coords')
    parser.add_argument('--train-image-size', type=int, nargs=2, 
                        default=(1080, 360),
                        help='size of random image crops during training')
    parser.add_argument('--yoffset', type=float, default=1.74,
                        help='vertical offset of the grid from the camera axis')
    
    # Model options
    parser.add_argument('--grid-height', type=float, default=4.,
                        help='size of grid cells, in meters')
    parser.add_argument('-r', '--grid-res', type=float, default=0.5,
                        help='size of grid cells, in meters')
    parser.add_argument('--frontend', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34'],
                        help='name of frontend ResNet architecture')
    parser.add_argument('--topdown', type=int, default=8,
                        help='number of residual blocks in topdown network')
    
    # Optimization options
    parser.add_argument('-l', '--lr', type=float, default=1e-9,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for SGD')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--lr-decay', type=float, default=0.99,
                        help='factor to decay learning rate by every epoch')
    parser.add_argument('--loss-weights', type=float, nargs=4, 
                        default=[1., 1., 1., 1.],
                        help="loss weighting factors for score, position,"\
                            " dimension and angle loss respectively")


    # Training options
    parser.add_argument('-e', '--epochs', type=int, default=600,
                        help='number of epochs to train for')
    parser.add_argument('-b', '--batch-size', type=int, default=8,
                        help='mini-batch size for training')
    
    # Experiment options
    parser.add_argument('--name', type=str, default=f'test/{current_time}',
                        help='name of experiment')
    parser.add_argument('-s', '--savedir', type=str, 
                        default='experiments',
                        help='directory to save experiments to')
    parser.add_argument('-g', '--gpu', type=int, nargs='*', default=[0],
                        help='ids of gpus to train on. Leave empty to use cpu')

    
    
    
    parser.add_argument('-gi', '--gpuid', type=int, default=0,
                        help='ids of gpus to train on')
    
    
    
    parser.add_argument('-w', '--workers', type=int, default=12,
                        help='number of worker threads to use for data loading')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='number of epochs between validation runs')
    parser.add_argument('--print-iter', type=int, default=100,
                        help='print loss summary every N iterations')
    parser.add_argument('--vis-iter', type=int, default=50,
                        help='display visualizations every N iterations')
    return parser.parse_args()
    

def _make_experiment(args):

    print('\n' + '#' * 80)
    print(datetime.now().strftime('%A %-d %B %Y %H:%M'))
    print('Creating experiment \'{}\' in directory:\n  {}'.format(
        args.name, args.savedir))
    print('#' * 80)
    print('\nConfig:')
    for key in sorted(args.__dict__):
        print('  {:12s} {}'.format(key + ':', args.__dict__[key]))
    print('#' * 80)
    
    # Create a new directory for the experiment
    savedir = os.path.join(args.savedir, args.name)
    os.makedirs(savedir, exist_ok=True)

    # Create tensorboard summary writer
    summary = SummaryWriter(savedir)

    # Save configuration to file
    with open(os.path.join(savedir, 'config.yml'), 'w') as fp:
        yaml.safe_dump(args.__dict__, fp)
    
    # Write config as a text summary
    summary.add_text('config', '\n'.join(
        '{:12s} {}'.format(k, v) for k, v in sorted(args.__dict__.items())))
    summary.file_writer.flush()

    return summary
    
    
def save_checkpoint(args, epoch, model, optimizer, scheduler):

    model = model.module if isinstance(model, nn.DataParallel) else model
    ckpt = {
        'epoch' : epoch,
        'model' : model.state_dict(),
        'optim' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict(),
    }
    ckpt_file = os.path.join(
        args.savedir, args.name, 'checkpoint-overfit-{:04d}.pth.gz'.format(epoch))
    print('==> Saving checkpoint \'{}\''.format(ckpt_file))
    torch.save(ckpt, ckpt_file)

def min_save_checkpoint(args, epoch, model, optimizer, scheduler):

    model = model.module if isinstance(model, nn.DataParallel) else model
    ckpt = {
        'epoch' : epoch,
        'model' : model.state_dict(),
        'optim' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict(),
    }
    ckpt_file = os.path.join(
        args.savedir, args.name, 'min_checkpoint-overfit.pth.gz')
    print('==> Saving checkpoint \'{}\''.format(ckpt_file))
    torch.save(ckpt, ckpt_file)




def main():
    
    min_loss=1e+12

    # Parse command line arguments
    args = parse_args()

    # Create experiment
    summary = _make_experiment(args)

    # Create datasets
    train_data = KittiObjectDataset(
        args.root, 'train', args.grid_size, args.grid_res, args.yoffset)
    val_data = KittiObjectDataset(
        args.root, 'val', args.grid_size, args.grid_res, args.yoffset)
    
    # Apply data augmentation
    train_data = oft.AugmentedObjectDataset(
        train_data, args.train_image_size, args.train_grid_size, 
        jitter=args.grid_jitter)

    # Create dataloaders
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, 
        num_workers=args.workers, collate_fn=oft.utils.collate)
    val_loader = DataLoader(val_data, args.batch_size, shuffle=False, 
        num_workers=args.workers,collate_fn=oft.utils.collate)

    # Build model
    model = OftNet(num_classes=1, frontend=args.frontend, 
                   topdown_layers=args.topdown, grid_res=args.grid_res, 
                   grid_height=args.grid_height)
    
    #PATH_="experiments/test/Rishi10/checkpoint-overfit-0408.pth.gz"
    #chck_pt_=torch.load(PATH_)
    #model.load_state_dict(chck_pt_['model'])
    #if len(args.gpu) > 0:
    #    torch.cuda.set_device(args.gpu[0])
    #    model = nn.DataParallel(model, args.gpu).to(f"cuda:{args.gpuid}")
    
    if args.gpuid != "":
        model.to(f"cuda:{args.gpuid}")

    # Create encoder
    encoder = ObjectEncoder()

    # Setup optimizer
    optimizer = optim.SGD(
        model.parameters(), args.lr, args.momentum, args.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay)
    
    #epoch_=chck_pt_['epoch']

    for epoch in range(1, args.epochs+1):

        print('\n=== Beginning epoch {} of {} ==='.format(epoch, args.epochs))
        
        # Update and log learning rate
        scheduler.step(epoch-1)
        summary.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # Train model
        train(args, train_loader, model, encoder, optimizer, summary, epoch)

        # Run validation every N epochs
        if epoch % args.val_interval == 0:
            
            new_loss=validate(args, val_loader, model, encoder, summary, epoch)

            # Save model checkpoint
            if epoch%6==0:
                save_checkpoint(args, epoch, model, optimizer, scheduler)
                
            if new_loss<min_loss:
                min_loss=new_loss
                min_save_checkpoint(args, epoch, model, optimizer, scheduler)
                

if __name__ == '__main__':
    main()

            

