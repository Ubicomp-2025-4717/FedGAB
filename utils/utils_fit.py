import os
from threading import local

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from .utils import get_lr


def fit_one_epoch(weight_r,weight_i,weight_a,model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss      = 0
    total_accuracy  = 0

    val_loss        = 0
    val_accuracy    = 0

    train_loss_r = 0
    train_loss_i = 0
    train_loss_a=0
    val_loss_r = 0
    val_loss_i = 0
    val_loss_a=0
    modelity_r = 1
    modelity_i = 1
    modelity_a = 1

    if weight_r < 0.1:
        modelity_r = 0

    if weight_i < 0.1:
        modelity_i = 0

    if weight_a < 0.1:
        modelity_a = 0


    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
        images, targets = batch
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = targets.cuda(local_rank)

                targets_t=[1,1,1,1,1,1,1,1]
                targets_t=torch.tensor(targets_t)
                targets_t=targets_t.cuda(local_rank)

                targets_v=[0,0,0,0,0,0,0,0]
                targets_v=torch.tensor(targets_v)
                targets_v=targets_v.cuda(local_rank)
                
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            # outputs,outputs_t,outputs_v     = model_train(images)
            outputs,outputs_r,outputs_i,outputs_a = model_train(images)
            #----------------------#
            #   计算损失
            #----------------------#
            # loss_value  = nn.CrossEntropyLoss()(outputs, targets)
            # loss_value = nn.CrossEntropyLoss()(outputs, targets)+0.02*(nn.CrossEntropyLoss()(outputs_t, targets_t)+nn.CrossEntropyLoss()(outputs_v, targets_v))

            loss_value = nn.CrossEntropyLoss()(outputs, targets) + weight_r * (
                nn.CrossEntropyLoss()(outputs_r, targets)) + weight_i * (nn.CrossEntropyLoss()(outputs_i, targets))+weight_a * (nn.CrossEntropyLoss()(outputs_a, targets))
            loss_value_r = nn.CrossEntropyLoss()(outputs_r, targets)
            loss_value_i = nn.CrossEntropyLoss()(outputs_i, targets)
            loss_value_a = nn.CrossEntropyLoss()(outputs_a, targets)



            loss_value.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                # outputs,outputs_t,outputs_v    = model_train(images)
                outputs,outputs_r,outputs_i,outputs_a = model_train(images)

                #----------------------#
                #   计算损失
                #----------------------#
                # loss_value  = nn.CrossEntropyLoss()(outputs, targets)
                # loss_value = nn.CrossEntropyLoss()(outputs, targets) + 0.02 * (nn.CrossEntropyLoss()(outputs_t, targets_t) + nn.CrossEntropyLoss()(outputs_v, targets_v))
                loss_value = nn.CrossEntropyLoss()(outputs, targets) + weight_r * (
                    nn.CrossEntropyLoss()(outputs_r, targets)) + weight_i * (
                                 nn.CrossEntropyLoss()(outputs_i, targets)) + weight_a * (
                                 nn.CrossEntropyLoss()(outputs_a, targets))
                loss_value_r = nn.CrossEntropyLoss()(outputs_r, targets)
                loss_value_i = nn.CrossEntropyLoss()(outputs_i, targets)
                loss_value_a = nn.CrossEntropyLoss()(outputs_a, targets)



            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()


        total_loss += loss_value.item()
        train_loss_r += loss_value_r.item()
        train_loss_i += loss_value_i.item()
        train_loss_a += loss_value_a.item()

        with torch.no_grad():
            accuracy = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
            total_accuracy += accuracy.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'accuracy'  : total_accuracy / (iteration + 1), 
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)




    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = targets.cuda(local_rank)

            optimizer.zero_grad()

            outputs, outputs_r, outputs_i,outputs_a = model_train(images)

            loss_value = nn.CrossEntropyLoss()(outputs, targets)

            loss_value_r = nn.CrossEntropyLoss()(outputs_r, targets)
            loss_value_i = nn.CrossEntropyLoss()(outputs_i, targets)
            loss_value_a = nn.CrossEntropyLoss()(outputs_a, targets)

            val_loss    += loss_value.item()
            accuracy        = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
            val_accuracy    += accuracy.item()

        val_loss += loss_value.item()
        val_loss_r += loss_value_r.item()
        val_loss_i += loss_value_i.item()
        val_loss_a += loss_value_a.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1),
                                'accuracy'  : val_accuracy / (iteration + 1), 
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
                
    if local_rank == 0:

        pbar.close()
        print('Finish Validation')

        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val,weight_r,weight_i,weight_a,modelity_r,modelity_i,modelity_a)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        # if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        #     torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        # torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
        train_r = train_loss_r / epoch_step
        train_i = train_loss_i / epoch_step
        train_a = train_loss_a / epoch_step
        val_r = val_loss_r / epoch_step_val
        val_i = val_loss_i / epoch_step_val
        val_a = val_loss_a / epoch_step_val

    return train_r, val_r, train_i, val_i, train_a, val_a

