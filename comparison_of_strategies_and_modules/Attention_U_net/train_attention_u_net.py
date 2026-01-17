from timm.scheduler import CosineLRScheduler
from collections import OrderedDict
from utils import *
import os.path as osp
import torch.nn as nn
from functions import train_one_epoch, valid_one_epoch,test_one_epoch
from torch.utils.data import DataLoader
from dataloader_thermohaline import Thermohaline_dataset
from model_2 import Thermohaline_Model
from typing import Dict, List, Union
import numpy as np
import torch
depths1 = [0., 5., 10., 20., 30., 50., 75., 100.]
depths2 = [0., 5., 10., 20., 30., 50., 75., 100., 125.,
           150., 200., 250., 300.]
depths3 = [0., 5., 10., 20., 30., 50., 75., 100., 125.,
           150., 200., 250., 300., 400., 500., 600., 700., 800.,
           900., 1000., 1100., 1200., 1300., 1400., 1500., 1750., 2000.]
def setup():

    model = Thermohaline_Model(in_shape=(10, 1, 180, 360), hid_S=64, hid_T=256, N_S=2, mlp_ratio=4, drop=0,
                               drop_path=0.1, spatio_kernel_dec=(3, 3), spatio_kernel_enc=(3, 3),
                               Depth_out1=len(depths1),Depth_out2=len(depths2),Depth_out3=len(depths3)).to(torch.device('cuda:0'))
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=0,lr=0.001)
    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=1000,
        lr_min=1e-6,
        warmup_lr_init=1e-5,
        warmup_t=5,
        t_in_epochs=True,  # update lr by_epoch
        k_decay=1.0)
    criterion = nn.MSELoss()

    train_dataset=Thermohaline_dataset('./data','thermohaline',[0,168],1)
    valid_dataset = Thermohaline_dataset('./data', 'thermohaline', [168, 180], 1)

    train_loader = DataLoader(train_dataset, batch_size=1,
                              num_workers=8, shuffle=True, pin_memory=True, drop_last=True)

    valid_loader = DataLoader(valid_dataset, batch_size=1,
                             num_workers=8, shuffle=False, pin_memory=True, drop_last=True)

    return model, criterion, optimizer,lr_scheduler, train_loader, valid_loader

def weights_to_cpu(state_dict: OrderedDict) -> OrderedDict:
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.

    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    # Keep metadata in state_dict
    state_dict_cpu._metadata = getattr(  # type: ignore
        state_dict, '_metadata', OrderedDict())
    return state_dict_cpu
def print_log(message):
    import logging
    print(message)
    logging.info(message)
class Recorder:
    def __init__(self, verbose=False, delta=0):
        self.verbose = verbose
        self.best_score = None
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score >= self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss

def save_interval_checkpoint(epoch, mdoel,optimizer,lr_scheduler,checkpoints_path,name=''):
    checkpoint = {
        'epoch': epoch + 1,
        'optimizer': optimizer.state_dict(),
        'state_dict': weights_to_cpu(mdoel.state_dict()),
        'scheduler': lr_scheduler.state_dict()}
    torch.save(checkpoint, osp.join(checkpoints_path, name + '.pth'))
def load_checkpoint(model,optimizer,lr_scheduler,checkpoints_path,name=''):
    filename = name if osp.isfile(name) else osp.join(checkpoints_path, name + '.pth')
    try:
        checkpoint = torch.load(filename)
    except:
        return
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f'No state_dict found in checkpoint file {filename}')
    model.load_state_dict(checkpoint['state_dict'])
    if checkpoint.get('epoch', None) is not None:
        epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
    print(f'resume from {epoch}th checkpoints')
def current_lr(optimizer) -> Union[List[float], Dict[str, List[float]]]:
    """Get current learning rates.

    Returns:
        list[float] | dict[str, list[float]]: Current learning rates of all
        param groups. If the runner has a dict of optimizers, this method
        will return a dict.
    """
    lr: Union[List[float], Dict[str, List[float]]]
    if isinstance(optimizer, torch.optim.Optimizer):
        lr = [group['lr'] for group in optimizer.param_groups]
    elif isinstance(optimizer, dict):
        lr = dict()
        for name, optim in optimizer.items():
            lr[name] = [group['lr'] for group in optim.param_groups]
    else:
        raise RuntimeError(
            'lr is not applicable because optimizer does not exist.')
    return lr
def main():
    set_seed(42)
    model_dir, log_dir, save_dir,plot_dir,best_checkpoint,interval_checkpoint=make_dir('./results_exp5_E_64_T_256_5_5_5_cosine_lr_300_1000')

    logger = init_logger(log_dir)
    epochs=1000
    model, criterion, optimizer, lr_scheduler,train_loader, valid_loader= setup()
    train_losses, valid_losses = [], []
    best_metric = (0, float('inf'), float('inf'))
    min_mse=np.inf
    early_stop_cnt=0
    print('>' * 35 + ' training ' + '<' * 35)
    recorder = Recorder(verbose=True)
    resume_from=False
    if resume_from:
        name=''
        load_checkpoint(model,optimizer,lr_scheduler,interval_checkpoint,name)
    for epoch in range(epochs):
        print(f'第{epoch+1}次开始')
        start_time = time.time()
        train_loss = train_one_epoch(epoch, model, train_loader, criterion, optimizer,lr_scheduler)
        train_losses.append(train_loss)
        plot_loss(train_losses, 'train', epoch, plot_dir, 1)

        cur_lr=current_lr(optimizer)
        cur_lr=sum(cur_lr)/len(cur_lr)
        with torch.no_grad():
            valid_loss, mse, ssim = valid_one_epoch( model, valid_loader, criterion)

        valid_losses.append(valid_loss)

        plot_loss(valid_losses, 'valid', epoch,plot_dir, 1)

        if mse < best_metric[1]:
            torch.save(model.state_dict(), f'{model_dir}/trained_model_state_dict')
            best_metric = (epoch, mse, ssim)

        logger.info(f'[Current Best] EP:{best_metric[0]:04d} MSE:{best_metric[1]:.4f} SSIM:{best_metric[2]:.4f}')
        print_log('Epoch: {0}, Steps: {1} | Lr: {2:.7f} | Train Loss: {3:.7f} | Vali Loss: {4:.7f}\n'.format(
            epoch + 1, len(train_loader), cur_lr, train_loss, valid_loss))
        print(f'Time usage per epoch: {time.time() - start_time:.0f}s')
        recorder(valid_loss, model, best_checkpoint)
        if min_mse>valid_loss:
            min_mse=valid_loss
            early_stop_cnt=0
        else:
            early_stop_cnt+=1

        # if epoch % 5 == 0:
        #     save_interval_checkpoint(epoch,model,optimizer,lr_scheduler,interval_checkpoint,name=f'it_is_{epoch + 1}_th_checkpoint')
        if early_stop_cnt>30:
            break
    print('>' * 35 + ' testing ' + '<' * 35)
    best_model_path = osp.join(best_checkpoint, 'checkpoint.pth')
    model.load_state_dict(torch.load(best_model_path))
    trues,preds=test_one_epoch(model,valid_loader)
    np.save(os.path.join(save_dir,'trues.npy'),trues)
    np.save(os.path.join(save_dir, 'preds.npy'), preds)
    print('complete!')
if __name__ == '__main__':
    main()
