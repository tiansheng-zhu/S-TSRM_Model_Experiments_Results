import numpy as np
import torch
from utils import compute_metrics, visualize
from tqdm import tqdm

depths1 = [0., 5., 10., 20., 30., 50., 75., 100.]
depths2 = [0., 5., 10., 20., 30., 50., 75., 100., 125.,
           150., 200., 250., 300.]
depths3 = [0., 5., 10., 20., 30., 50., 75., 100., 125.,
           150., 200., 250., 300., 400., 500., 600., 700., 800.,
           900., 1000., 1100., 1200., 1300., 1400., 1500., 1750., 2000.]
device=torch.device('cuda:2')
depths1_out = torch.tensor([i * 10 for i in depths1]).to(device)
depths2_out = torch.tensor([i * 10 for i in depths2]).to(device)
depths3_out = torch.tensor([i * 10 for i in depths3]).to(device)

weights = torch.tensor([4, 4, 4, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 3, 3, 3, 3, 3, 3,
                        2, 2, 2, 2, 1, 1, 1]).float().to(device)
weights_par = (weights / torch.sum(weights)).float().to(device)


def train_one_epoch(epoch, model, train_loader, criterion, optimizer, lr_scheduler):
    model.train()
    lr_scheduler.step(epoch)
    losses = []
    train_pbar = tqdm(train_loader)
    for batch_x, batch_y in train_pbar:
        optimizer.zero_grad()
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        # preds_tensor=[]
        # for _ in range(2):
        #     Y1, Y2, Y3 = model(batch_x, preds_tensor, depths1_out,depths2_out, depths3_out)
        #
        #     loss1 = 0
        #     for i in range(len(depths1)):
        #         loss1 += criterion(Y1[:, i, :, :, :], batch_y[:, i, :, :, :]) * weights_par[i]
        #     loss2 = 0
        #     for i in range(len(depths2)):
        #         loss2 += criterion(Y2[:, i, :, :, :], batch_y[:, i, :, :, :]) * weights_par[i]
        #
        #     loss3 = 0
        #     for i in range(len(depths3)):
        #         loss3 += criterion(Y3[:, i, :, :, :], batch_y[:, i, :, :, :]) * weights_par[i]
        #     loss = (loss1 + loss2 + loss3)*1e8
        #
        #     preds_tensor.append(batch_y[:,:len(depths2),:,:,:])
        #     loss.backward()
        #     optimizer.step()
        #     losses.append(loss.item())
        #     train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))

        Y1, Y2, Y3 = model(batch_x)

        loss1 = 0
        for i in range(len(depths1)):
            loss1 += criterion(Y1[:, i, :, :, :], batch_y[:, i, :, :, :]) * weights_par[i]
        loss2 = 0
        for i in range(len(depths2)):
            loss2 += criterion(Y2[:, i, :, :, :], batch_y[:, i, :, :, :]) * weights_par[i]

        loss3 = 0
        for i in range(len(depths3)):
            loss3 += criterion(Y3[:, i, :, :, :], batch_y[:, i, :, :, :]) * weights_par[i]
        loss = (loss1 + loss2 + loss3) * 1e8
        loss.backward()
        optimizer.step()
        pure_loss=criterion(Y3,batch_y)*1e8
        losses.append(pure_loss.item())
        train_pbar.set_description('train loss: {:.4f}'.format(pure_loss.mean().item()))
        del batch_x, batch_y, Y1, Y2, Y3

    return np.mean(losses)


def valid_one_epoch(model, vali_loader, criterion):
    model.eval()
    losses, mses, ssims = [], [], []
    vali_pbar = tqdm(vali_loader)
    for batch_x, batch_y in vali_pbar:
        with torch.no_grad():
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            Y1, Y2, Y3 = model(batch_x)

            loss = criterion(Y3,batch_y)*1e8


            losses.append(loss.item())
            vali_pbar.set_description(
                'vali loss: {:.4f}'.format(loss.mean().item()))


            mse, ssim = compute_metrics(Y3, batch_y)

            mses.append(mse)
            ssims.append(ssim)
            del batch_x, batch_y, Y1, Y2, Y3

    return np.mean(losses), np.mean(mses), np.mean(ssims)


# device = torch.device('cuda:0')
def test_one_epoch(model, test_loader):
    # depths1_out = torch.tensor([i * 10 for i in depths1]).to(torch.device('cpu'))
    # depths2_out = torch.tensor([i * 10 for i in depths2]).to(torch.device('cpu'))
    # depths3_out = torch.tensor([i * 10 for i in depths3]).to(torch.device('cpu'))
    # device = torch.device('cpu')
    model.eval()
    trues_lst, preds_lst = [], []
    test_pbar = tqdm(test_loader)
    for batch_x, batch_y in test_pbar:
        batch_x = batch_x.to(device)
        Y1,Y2,pred_y = model(batch_x)

        list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
            batch_y, pred_y], [trues_lst, preds_lst]))
        del batch_x, batch_y, pred_y

    trues, preds = map(
        lambda data: np.concatenate(data, axis=0), [trues_lst, preds_lst])
    return trues, preds