import os
import numpy as np
from pathlib import Path
import torch
import sys
import time
from torch.utils.tensorboard import SummaryWriter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the CUDA_VISIBLE_DEVICES environment variable
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Set a random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Determine the project root directory
sys.path.append(os.getcwd())

# Import modules from the src directory
from src import config_loader
from src import data_preprocessing
from src import generators
from src import networks
from src import losses
from src import train

# # Function to force CuDNN initialization
# def force_cudnn_initialization():
#     s = 32
#     dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

# # Call the function to force CuDNN initialization
# force_cudnn_initialization()

print(f'***** pid ******   {os.getpid()}')



"""***********************************************************************************
############ Config Load ############
***********************************************************************************"""

# Define the path to the miccai.yaml configuration file
config = config_loader.update_config_with_args(config_loader.parse_args())

print(config)

# data organization parameters
model_saving_name = 'model_'+str(config['experiment']['date_ver'])+'_l2_' + str(config['training']['lambda_L2']) + '_lr_' + str(config['optimizer']['lr']) + '_MI_' + str(config['training']['lambda_MI']) + "_T1_" + str(config['training']['lambda_T1']) + "_DSC_" + str(config['training']['lambda_Dice']) + "_KFOLD_" + str(config['experiment']['fold'])
print(f"{model_saving_name=} ")

# init writer
log_dir = os.path.join(config['path']['models_dir'],'logs')
writer = SummaryWriter(log_dir=log_dir)


"""***********************************************************************************
############ Load Data and init generators ############
***********************************************************************************"""

data_set = data_preprocessing.loading_test_train_set(config)

generator =  generators.mri_data_generator_scan_to_scan_train_groupReg(data_set['train_set_data'], data_set['train_set_trigger_time'],data_set['train_set_norm'],data_set['train_set_segmentation'] ,\
                                                                       batch_size = config['experiment']['train_batch_size'],mode='train')
generator_validate = generators.mri_data_generator_scan_to_scan_train_groupReg(data_set['validate_set_data'], data_set['validate_set_trigger_time'],data_set['validate_set_norm'], \
                                                                               data_set['validate_set_segmentation'],batch_size = config['experiment']['test_batch_size'],mode = 'test')


"""***********************************************************************************
############ Load or init Model ############
***********************************************************************************"""

inshape = next(generator)[0].shape[1:-1] 
if config['path']['load_model'] != 'None':
    # load initial model (if specified)
    model = networks.PCMC.load(config['path']['load_model'], device)
else:
    print('configure new model')
    model = networks.PCMC(
        inshape=inshape[0:2],
        nb_unet_features=[config['arch']['enc_nf'], config['arch']['dec_nf']],
        bidir=False,
        int_steps=7,
        int_downsize=2,
        src_feats=inshape[2], 
        trg_feats=0)

print(model)
# prepare the model for training and send to device
model.to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr = config['optimizer']['lr'], weight_decay = config['optimizer']['weight_decay']) ; print('optimizer'); print(optimizer)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9997); print('scheduler'); print(scheduler)

# prepare deformation loss
loss_group_mi_f = losses.GroupMutualInformation()
loss_L2_f = losses.GradGroup('l2', loss_mult=1).group_loss
loss_T1_f = losses.T1GroupModelBased().loss
loss_Dice_f = losses.DiceSC().loss

best_r2 = 0
best_dice = 0
epochs_losses_array = []

"""***********************************************************************************
############ Training ############
***********************************************************************************"""

# training loops
training_start_time = time.time()

for epoch in range(config['experiment']['initial_epoch'], config['experiment']['epochs']):
    model, optimizer, _ = train.train(config,model,generator,optimizer,device, epoch,writer,loss_T1_f,loss_group_mi_f,loss_L2_f,loss_Dice_f)

    if epoch > 0:# and epoch % 5 == 0:
        loss_validate,r2_info, dice_info = train.validate(config, model, epoch, generator_validate, \
                                                             writer, loss_T1_f,loss_group_mi_f,loss_L2_f,loss_Dice_f, r2_flag=True)
        # save checkpoint
        if r2_info[1] > best_r2:
            print('**** saving checkpoint R2 ; epoch {}'.format(epoch))
            model.save(os.path.join(config['path']['models_dir'],model_saving_name + '.pt'))
            best_r2 = r2_info[1]

    scheduler.step()

training_total_time = (training_start_time - time.time())/60
print("training running time: {}".format(training_total_time))

