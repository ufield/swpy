import torch
import time
import pandas as pd

import sys
sys.path.append('../dataset')
sys.path.append('../utils')

from data_processes import create_inputs_with_omni_mean
from files import make_dirs

import gpytorch

def train_nn(net, dataloaders_dict, criterion, optimizer, num_epochs=50, outdir_root='.'):

    # GPU使えるか
    device = torch.device("cpu")
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス:", device)

    # ネットワークをdeviceへ
    net.to(device)

    torch.backends.cudnn.benchmark = True

    iteration = 1
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    logs = []

    for epoch in range(num_epochs + 1):
        t_epoch_start = time.time()
        t_iter_start = time.time()

        print('--------------------')
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('--------------------')

        for phase in ['train', 'val']:
            if phase == 'train':
                print('(train)')
                net.train()
            else:
                if((epoch+1)%10 == 0):
                    net.eval()
                    print('--------------------')
                    print('(val)')
                else:
                    continue

            for dst_f_b, dst_p_b, dst_diff_b, omni_b in dataloaders_dict[phase]:

                dst_f_b = dst_f_b.float().to(device)
                dst_p_b = dst_p_b.float().to(device)
                omni_b = {key: omni_b[key].to(device) for key in omni_b}

                optimizer.zero_grad()

                x = create_inputs_with_omni_mean(dst_p_b, omni_b)
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = net(x)
                    loss = criterion(outputs, dst_f_b)
                    print('loss.item: ' , loss.item())

                    if phase == 'train':
                        loss.backward()
#                         nn.utils.clip_grad_value_(net.parameters(), clip_value=2.0)
                        optimizer.step()

                        if (iteration % 10 == 0):
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            print('イテレーション {} || Loss: {:.4f} || 10iter: {:.4f} sec.'.format(iteration, loss.item(), duration))
                            t_iter_start = time.time()

                        epoch_train_loss += loss.item()
                        iteration +=1

                    else:
                        epoch_val_loss += loss.item()

        # epoch の phase ごとのloss
        t_epoch_finish = time.time()
        print('--------------------')
        print('epoch {} || Epoch_Train_Loss:{:.4f} || Epoch_Val_Loss:{:.4f}'.format(epoch+1, epoch_train_loss, epoch_val_loss))
        print('timer: {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

        # ログを保存
        log_epoch = {'epoch': epoch+1, 'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        make_dirs(outdir_root)
        df.to_csv('{}/log.csv'.format(outdir_root))

        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        # weight を保存
        if ((epoch+1) % 10 == 0):
            path = outdir_root + '/weights'
            make_dirs(path)
            torch.save(net.state_dict(), '{}/weights_{}.pth'.format(path, str(epoch + 1)))


def train_gp(train_x, train_y, model, likelihood, optimizer, num_epochs=50, outdir_root='.'):
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    logs = []
    for iter in range(num_epochs + 1):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients

        loss = -mll(output, train_y.float())
        loss.backward()

        optimizer.step()


        # -------------------------------
        lengthscale = model.covar_module.base_kernel.lengthscale.item()
        noise = model.likelihood.noise.item()

        print('Epoch %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            iter + 1, num_epochs, loss.item(),
            lengthscale,
            noise
        ))

        # ログ保存
        log_epoch = {'epoch': iter + 1, 'train_loss': loss, 'lengthscale': lengthscale, 'noise': noise}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        make_dirs(outdir_root)
        df.to_csv('{}/log_gp.csv'.format(outdir_root))

        # モデルを保存
        if ((iter+1) % 10 == 0):
            path = outdir_root + '/gp_models'
            make_dirs(path)
            torch.save(model.state_dict(), '{}/gp_model_{}.pth'.format(path, str(iter + 1)))

