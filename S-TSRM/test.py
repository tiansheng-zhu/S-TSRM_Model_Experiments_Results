from timm.scheduler import CosineLRScheduler
from torch.utils.data import DataLoader
from dataloader_thermohaline import Thermohaline_dataset
from model import Thermohaline_Model
import numpy as np
from collections import OrderedDict
from utils import *
import os.path as osp
import torch.nn as nn
from functions_2 import train_one_epoch, valid_one_epoch,test_one_epoch
from torch.utils.data import DataLoader
from typing import Dict, List, Union
import torch
depths1 = [0., 5., 10., 20., 30., 50., 75., 100., 125.,
           150., 200., 250., 300.]
depths2 = [0., 5., 10., 20., 30., 50., 75., 100., 125.,
           150., 200., 250., 300., 400., 500., 600., 700., 800.,
           900., 1000.]
depths3 = [0., 5., 10., 20., 30., 50., 75., 100., 125.,
           150., 200., 250., 300., 400., 500., 600., 700., 800.,
           900., 1000., 1100., 1200., 1300., 1400., 1500., 1750., 2000.]

device=torch.device('cuda:0')
from train_vit import setup

def main():
    model, criterion, optimizer, lr_scheduler, train_loader, valid_loader = setup()
    model_dir, log_dir, save_dir, plot_dir, best_checkpoint, interval_checkpoint = make_dir('./tri-stage-vit-temp')
    print('>' * 35 + ' testing ' + '<' * 35)
    best_model_path = osp.join(best_checkpoint, 'checkpoint.pth')
    model.load_state_dict(torch.load(best_model_path))
    trues, preds = test_one_epoch(model, valid_loader)
    np.save(os.path.join(save_dir, 'trues.npy'), trues)
    np.save(os.path.join(save_dir, 'preds.npy'), preds)
    print('complete!')

if __name__ == '__main__':
    main()