
import torch
import os
import sys
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np
sys.path.append(os.getcwd())
from src import utils


def train(config,model,generator,optimizer,device, epoch,writer,loss_T1_f,loss_group_mi_f,loss_L2_f,loss_Dice_f):
    
    epoch_total_loss = []

    for step in range(config['experiment']['steps_per_epoch']):
        # generate inputs (and true outputs) and convert them to tensors
        y_group, y_trigger_time, y_normalization, y_seg,_ = next(generator) 
        train_set_segmentation_reference = y_seg[:, :, :, 0, :]
        y_group = torch.from_numpy(y_group).to(device).float().permute(0, 4, 1, 2, 3)
        y_group_seg = torch.from_numpy(y_seg).to(device).float().permute(0, 4, 1, 2, 3)

        #create segmentation mask based on the T1 values
        with torch.no_grad():
            y_group_mask = torch.where(torch.sum(torch.tensor(y_seg),axis=3)[:,:,:,0] > 0,1,0)

        # run inputs through the model to produce a warped image and flow field
        y_pred, warp_pre_int,M0,T1,y_pred_seg = model(y_group, y_group_seg)

        # calculate total loss
        loss_list_M0, loss_list_T1,loss_list_L2,loss_list_MI, loss_list_dice, = [],[],[],[],[]
        loss = 0
        loss_L2 = loss_L2_f(warp_pre_int) * config['training']['lambda_L2']
        loss_list_L2.append(loss_L2.item())
        loss += loss_L2
        if config['training']['lambda_MI']: 
            loss_MI = loss_group_mi_f(y_pred) * config['training']['lambda_MI']
            loss_list_MI.append(loss_MI.item())
            loss += loss_MI
        if config['training']['lambda_T1']:
            loss_T1, _ ,_= loss_T1_f(y_pred,y_trigger_time,y_normalization,y_group_mask,M0,T1) 
            loss_T1 = loss_T1 * config['training']['lambda_T1']
            loss_list_T1.append(loss_T1.item())
            loss += loss_T1
        if config['training']['lambda_Dice']:
            torch_train_set_segmentation_reference = torch.from_numpy(train_set_segmentation_reference).to(device).permute(0, 3, 1, 2)
            loss_dice,_,_ = loss_Dice_f((y_pred_seg), torch_train_set_segmentation_reference)   
            loss_dice = loss_dice * config['training']['lambda_Dice']
            loss_list_dice.append(loss_dice.item())
            loss +=loss_dice            

        epoch_total_loss.append(loss.item())

        # write the learning rate
        if writer is not None:
            writer.add_scalar("learning rate", optimizer.state_dict()['param_groups'][0]['lr'] , epoch * config['experiment']['steps_per_epoch'] + step)
        
        # backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if writer is not None:
        # train writers
        writer.add_scalar("Total Loss", np.mean(epoch_total_loss), epoch)
        writer.add_scalar("lossDeformation", np.mean(loss_list_L2), epoch)
        if config['training']['lambda_T1']:
            writer.add_scalar("lossT1", np.mean(loss_list_T1), epoch) 
            writer.add_scalar("lossM0", np.mean(loss_list_M0), epoch) 
        if config['training']['lambda_MI']:
            writer.add_scalar("lossMI", np.mean(loss_list_MI), epoch) 
        writer.add_scalar("Dice", np.mean(loss_list_dice), epoch) 

    # print epoch info train
    epoch_info = 'Epoch %d/%d' % (epoch + 1, config['experiment']['epochs'])
    losses_string =''
    for f in [('L2 loss = ',loss_list_L2),('; T1 loss = ', loss_list_T1),('; M0 loss = ',loss_list_M0),('; Mi loss =',loss_list_MI),('; Dice = ',loss_list_dice)]:
        if f[1] != []:
            losses_string = losses_string + f[0] + str(np.round(np.mean(f[1]),7))
    loss_info = 'train-loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_string)
    print(' - '.join((epoch_info, loss_info)), flush=True)

    return model, optimizer, loss.item()
    


def validate(config, model, epoch, generator_test, writer, loss_T1_f,loss_group_mi_f,loss_L2_f,loss_Dice_f, r2_flag):
    # Test loss
    with torch.no_grad():
        y_test_group, y_test_trigger_time, y_test_normalization, y_test_seg,_ = next(generator_test) 
        test_set_segmentation_reference = y_test_seg[:, :, :, 0, :]
        y_test_mask = np.where(np.sum(y_test_seg,axis=3)[:,:,:,0] > 0,1,0)
        y_test_group = torch.from_numpy(y_test_group).to(device).float().permute(0, 4, 1, 2, 3)
        y_test_seg = torch.from_numpy(y_test_seg).to(device).float().permute(0, 4, 1, 2, 3)
        test_set_segmentation_reference = torch.from_numpy(test_set_segmentation_reference).to(device).permute(0, 3, 1, 2)

        # run inputs through the model to produce a warped image and flow field
        y_pred_test, warp_pre_int_test,M0,T1,y_test_pred_seg= model(y_test_group,y_test_seg)

        # calculate total loss
        loss_list_M0_test, loss_list_T1_test,loss_list_L2_test,loss_list_MI_test = [],[],[],[]
        loss_list_dice_test = []
        loss_test = 0
        loss_L2_test = loss_L2_f(warp_pre_int_test) * config['training']['lambda_L2']
        loss_list_L2_test.append(loss_L2_test.item())
        loss_test += loss_L2_test
        if config['training']['lambda_MI']: 
            loss_MI_test = loss_group_mi_f(y_pred_test) * config['training']['lambda_MI']
            loss_list_MI_test.append(loss_MI_test.item())
            loss_test += loss_MI_test
        if config['training']['lambda_T1']:

            loss_T1_test, y_synt_test, y_pred_test_renormalize = loss_T1_f(y_pred_test,y_test_trigger_time,y_test_normalization,y_test_mask,M0,T1) 
            loss_T1_test = loss_T1_test * config['training']['lambda_T1']
            loss_list_T1_test.append(loss_T1_test.item())
            loss_test += loss_T1_test

        # Dice score
        loss_dice_test,loss_dice_test_info,_ = loss_Dice_f((y_test_pred_seg), test_set_segmentation_reference)   
        loss_list_dice_test.append(loss_dice_test.item())
        if config['training']['lambda_Dice']:
            loss_test += loss_dice_test

        # print epoch info test
        epoch_info = 'Epoch %d/%d' % (epoch + 1, config['experiment']['epochs'])

        losses_string =''
        for f in [('L2 loss = ',loss_list_L2_test),('; T1 loss = ', loss_list_T1_test),('; M0 loss = ',loss_list_M0_test),('; Mi loss =',loss_list_MI_test),('; Dice = ',loss_list_dice_test)]:
            if f[1] != []:
                losses_string = losses_string + f[0] + str(np.round(np.mean(f[1]),7))
        loss_info = 'test- loss: %.4e  (%s)' % (np.mean(loss_test.item()), losses_string)
        print(' - '.join((epoch_info, loss_info)), flush=True)

        # check the R2
        
        if r2_flag:
            registered_test_set_numpy = y_pred_test.permute(0, 2, 3, 1).detach().cpu().numpy()[:,:,:,:,np.newaxis]
            test_set_segmentation_reference_numpy = test_set_segmentation_reference.permute(0, 2, 3, 1).detach().cpu().numpy()[:,:,:,:,np.newaxis]
            registered_median_r2, registered_median_T1,_ = utils.calculate_R_square_full_data_set(data=registered_test_set_numpy,
                                                                        reference_masks=test_set_segmentation_reference_numpy,
                                                                        trigger_time=y_test_trigger_time,norm_matrix=y_test_normalization, plot_flag=False)
            print(f'{registered_median_r2 = }')
            # print(f'{registered_median_T1 = }') 
            writer.add_scalar("median r2", registered_median_r2[0] , epoch)
        else:
            registered_median_r2 = -10
        print(f'{loss_dice_test_info = }')
    return loss_test.item(), registered_median_r2, loss_dice_test_info





def original_analyze(generator_test,writer,loss_Dice_f):
    # analyze TEST data
    y_test_group, y_test_trigger_time, y_test_normalization, y_test_seg,_ = next(generator_test) 
    test_set_segmentation_reference = y_test_seg[:, :, :, 0, :]
    original_median_r2, original_median_T1,_ = utils.calculate_R_square_full_data_set(data=y_test_group,
                                                                reference_masks=test_set_segmentation_reference,
                                                                trigger_time=y_test_trigger_time,norm_matrix=y_test_normalization,
                                                                plot_flag=False,filename='temp')
    print('*********** original *************')
    writer.add_scalar("median r2", original_median_r2[0] , -1)
    writer.add_scalar("median T1", original_median_T1[0], -1)
    print('original r2 {}'.format(original_median_r2))
    registered_median_r2 = original_median_r2[0]

    y_test_group = torch.from_numpy(y_test_group).to(device).float().permute(0, 4, 1, 2, 3)
    y_test_seg = torch.from_numpy(y_test_seg).to(device).float().permute(0, 4, 1, 2, 3)
    test_set_segmentation_reference = torch.from_numpy(test_set_segmentation_reference).to(device).permute(0, 3, 1, 2)
    original_loss_dice_test,_,_ = loss_Dice_f((y_test_seg.permute(0,4,2,3,1)[:,:,:,:,0]), test_set_segmentation_reference)   
    writer.add_scalar("Dice", original_loss_dice_test, -2)
    original_loss_dice_test = -original_loss_dice_test
    print('original_loss_dice_test = ' + str(original_loss_dice_test))