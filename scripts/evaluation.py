import os
import sys
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Determine the project root directory
sys.path.append(os.getcwd())

# Import modules from the src directory
from src import config_loader
from src import data_preprocessing
from src import generators
from src import networks
from src import losses
from src import train
from src import utils


config = config_loader.update_config_with_args(config_loader.parse_args())

# # ***************** preprocessing *****************
registered_test_set = np.zeros((210,160,160,11,5))
registered_seg_set = np.zeros((210,160,160,11,5))

calc_flag = True
if calc_flag:
    for k_fold in range(5):
        print(f"{k_fold = }")
        # ***************** Loading *****************
        data_set = data_preprocessing.loading_test_train_set(config, fold  = k_fold)

        # load and set up model
        model_name = config['evaluation']['models_names'][k_fold]
        model_dir_path = os.path.join(config['evaluation']['models_path'],model_name)
        model_pt = os.path.join(model_dir_path,model_name + '.pt')
        model = networks.PCMC.load(model_pt, device);
        model.to(device);
        model.eval();

        # ***************** Evaluation *****************
        patients_number = data_set['test_set_data'].shape[0]
        t_reference = 0
        t = t_reference

        for i,p in enumerate(data_set['test_ind']):
            for s in range(5):
                moving_input = data_set['test_set_data'][i, :, :, :, s, np.newaxis][np.newaxis,...]
                moving_input = torch.from_numpy(moving_input).to(device).float().permute(0, 4, 1, 2, 3)

                moving_seg = data_set['test_set_segmentation'][i, :, :, :, s, np.newaxis][np.newaxis,...]
                moving_seg = torch.from_numpy(moving_seg).to(device).float().permute(0, 4, 1, 2, 3)

                registered, warped,M_0,T_1,registered_seg = model(moving_input,moving_seg)  #registered[B,11,160,160]
                registered_test_set[p, :, :, :, s] = registered.permute(0, 2, 3, 1).detach().cpu().numpy()
                registered_seg_set[p, :, :, :, s] = registered_seg.permute(0, 2, 3, 1).detach().cpu().numpy()


    model_dir_path = '/home/eyalhan/T1GroupModelBased/T1_mapping/models/' + model_name 
    os.makedirs(model_dir_path,exist_ok=True)
    np.save(model_dir_path + '/registered_test_set_all_folds.npy',registered_test_set)
    np.save(model_dir_path + '/registered_seg_set_all_folds.npy',registered_seg_set)

else:
    model_name = config['evaluation']['models_names'][-1]
    model_dir_path = os.path.join(config['evaluation']['models_path'],model_name)
    registered_test_set = np.load(os.path.join(model_dir_path ,'registered_test_set_all_folds.npy'))
    registered_seg_set = np.load(os.path.join(model_dir_path ,'/registered_seg_set_all_folds.npy'))


data_set = data_preprocessing.loading_test_train_set(config, k_fold = 'all')

# ***************** Metrics check *****************
utils.analysis(data_set['test_set_data'], data_set['test_set_segmentation'], registered_seg_set, model_dir_path, registered_test_set, data_set['test_set_trigger_time'], data_set['test_set_norm'], t_reference=0,\
     full_r2_flag = 1, specific_case_flag = 0, full_dice_flag = 1, Hausdorff_flag = 1, save_flag = 1, show_specific_case_flag =0)
