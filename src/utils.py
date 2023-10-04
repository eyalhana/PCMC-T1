import sys
sys.path.append(os.getcwd())
from src import metrics

def analysis(test_set_data, test_set_segmentation, registered_seg_set, model_dir_path, registered_test_set, test_set_trigger_time, test_set_norm, t_reference=0,\
     full_r2_flag = 0, specific_case_flag = 0, full_dice_flag = 1, Hausdorff_flag = 1, save_flag = 0, show_specific_case_flag =0):
    """
    test_set_data             [BS,W,H,T,S] 
    test_set_segmentation     [BS,W,H,T,S]  
    registered_seg_set        [BS,W,H,T,S]
    test_set_trigger_time     [BS,T,S]
    registered_test_set       [BS,H,W,T,S]
    test_set_norm             [BS,T,2]
    """

    # np.save(model_dir_path+'/registered_test_set.npy',registered_test_set)
    # np.save(model_dir_path+'/registered_seg_set.npy',registered_seg_set)
    full_r2_mean, full_r2_median, case_r2_mean, case_r2_median, DSC_model_mean, DSC_model_median = -1,-1,-1,-1,-1,-1

    # **************** DICE Over all the data set *********************
    test_set_segmentation_reference_np = test_set_segmentation[:, :, :, t_reference, :][:,:,:,np.newaxis,:].repeat(test_set_data.shape[3],axis=3)
    
    if full_dice_flag:
        t_reference = 0
                                        
        _,DSC_original_info, dice_original_all = metrics.DSC_loss_np(test_set_segmentation, test_set_segmentation_reference_np)  
        _,DSC_model_info,  dice_model_all = metrics.DSC_loss_np(registered_seg_set, test_set_segmentation_reference_np)  

        if save_flag:
            np.save(model_dir_path+'/DSC_original.npy',DSC_original_info)
            np.save(model_dir_path+'/DSC_model.npy',DSC_model_info)
            np.save(model_dir_path+'/dice_original_all.npy',dice_original_all)
            np.save(model_dir_path+'/dice_model_all.npy',dice_model_all)

        print(f"{DSC_original_info = } ")
        print(f"{DSC_model_info = } ")

        DSC_model_mean = DSC_model_info[1]
        DSC_model_median = DSC_model_info[0]

    if Hausdorff_flag:
        Hd_original_info, Hd_original_all = metrics.Hausdorff_distance(test_set_segmentation, test_set_segmentation_reference_np)
        Hd_model_info, Hd_model_all = metrics.Hausdorff_distance(registered_seg_set, test_set_segmentation_reference_np)

        if save_flag:
            np.save(model_dir_path+'/Hd_original.npy',Hd_original_info)
            np.save(model_dir_path+'/Hd_model.npy',Hd_model_info)
            np.save(model_dir_path+'/HD_original_all.npy',Hd_original_all)
            np.save(model_dir_path+'/HD_model_all.npy',Hd_model_all)

        print(f"{Hd_original_info = } ")
        print(f"{Hd_model_info = } ")
        
    # **************** R2 Over all the data set *********************
    if full_r2_flag:
        np.seterr(all="ignore")
        r2_model_avg_info,r2_model_all_info,_,_ = r2_analyze(data=registered_test_set, reference_masks=test_set_segmentation_reference_np,trigger_time=test_set_trigger_time, norm_matrix= test_set_norm)
        print(f"{r2_model_avg_info = } ")
        print(f"{r2_model_all_info = } ")

        r2_original_avg_info,r2_original_all_info,_,_ = r2_analyze(data=test_set_data, reference_masks=test_set_segmentation_reference_np,trigger_time=test_set_trigger_time, norm_matrix= test_set_norm)
        print(f"{r2_original_avg_info = } ")
        print(f"{r2_original_all_info = } ")
        
        if save_flag:
            np.save(model_dir_path+'/r2_original.npy',r2_original_all_info)
            np.save(model_dir_path+'/r2_model.npy',r2_model_all_info)

        full_r2_mean = r2_model_all_info[1]
        full_r2_median = r2_model_all_info[0]

    # **************** Specific Case *********************
    if specific_case_flag:
        p,s = 18, 2

        print('**************** case p{} s{} **************** '.format(p,s))
        # test_set_segmentation_reference_ps = (test_set_segmentation[p,:,:,0,s])[np.newaxis,...].repeat(11,axis=0)

        r2_original_case,_,_,maps_original_case = r2_analyze(data=test_set_data[p,:,:,:,s], reference_masks=test_set_segmentation_reference_np[p,:,:,:,s],trigger_time=test_set_trigger_time[p,:,s], norm_matrix= test_set_norm[p,:,:],full_T1_map_flag=True)
        _,dice_info_original_case, _ = metrics.DSC_loss_np(test_set_segmentation[p,:,:,:,s][np.newaxis,:,:,:], test_set_segmentation_reference_np[p,:,:,:,s][np.newaxis,:,:,:])  
        Hd_original_case_info, _ = metrics.Hausdorff_distance(test_set_segmentation[p,:,:,:,s][np.newaxis,:,:,:,np.newaxis],  test_set_segmentation_reference_np[p,:,:,:,s][np.newaxis,...,np.newaxis])
        Hd_original_case_95_info, _ = metrics.Hausdorff_distance_95(test_set_segmentation[p,:,:,:,s][np.newaxis,:,:,:,np.newaxis],  test_set_segmentation_reference_np[p,:,:,:,s][np.newaxis,...,np.newaxis])

        print('r2_original_case = {}'.format(np.round(r2_original_case,4)))
        print('DSC_original_case = {}'.format(np.round(dice_info_original_case,4)))
        print('HD_original_case = {}'.format(np.round(Hd_original_case_info,4)))
        print('HD_original_case_95 = {}'.format(np.round(Hd_original_case_95_info,4)))

        if show_specific_case_flag:
            plt.imshow(maps_original_case[0,0,:,:,1],cmap='jet',vmin=400,vmax=1600);plt.colorbar()
            plt.title('r2 mean = {}; DSC mean = {}; HD mean'.format(round(r2_original_case[1],2) ,  round(dice_info_original_case[1],2),round(Hd_original_case_info[1],2)) ) 
            plt.savefig(model_dir_path+'/p_{}_s{}_original.png'.format(p,s))
            plt.close('all')

        r2_model_case,_,_,maps_model_case = r2_analyze(data=registered_test_set[p,:,:,:,s], reference_masks=test_set_segmentation_reference_np[p,:,:,:,s],trigger_time=test_set_trigger_time[p,:,s], norm_matrix= test_set_norm[p,:,:],full_T1_map_flag=True)
        _,dice_info_model_case, _ = metrics.DSC_loss_np(registered_seg_set[p,:,:,:,s][np.newaxis,:,:,:], test_set_segmentation_reference_np[p,:,:,:,s][np.newaxis,:,:,:])  
        Hd_model_case_info, _ = metrics.Hausdorff_distance(registered_seg_set[p,:,:,:,s][np.newaxis,:,:,:,np.newaxis],  test_set_segmentation_reference_np[p,:,:,:,s][np.newaxis,...,np.newaxis])
        Hd_model_case_95_info, _ = metrics.Hausdorff_distance_95(registered_seg_set[p,:,:,:,s][np.newaxis,:,:,:,np.newaxis],  test_set_segmentation_reference_np[p,:,:,:,s][np.newaxis,...,np.newaxis])

        print('r2_model_case = {}'.format(np.round(r2_model_case,4)))
        print('DSC_model_case = {}'.format(np.round(dice_info_model_case,4)))
        print('HD_model_case = {}'.format(np.round(Hd_model_case_info,4)))
        print('HD_model_95_case = {}'.format(np.round(Hd_model_case_95_info,4)))

        if show_specific_case_flag:
            plt.imshow(maps_model_case[0,0,:,:,1],cmap='jet',vmin=400,vmax=1600);plt.colorbar()
            plt.title('r2 mean = {}; DSC mean = {}; HD mean'.format(round(r2_model_case[1],2) ,  round(dice_info_model_case[1],2),round(Hd_model_case_info[1],2)) ) 
            plt.savefig(model_dir_path+'/p_{}_s{}_model.png'.format(p,s))

        case_r2_mean = r2_model_case[1]
        case_r2_median = r2_model_case[0]


    return full_r2_mean, full_r2_median, case_r2_mean, case_r2_median, DSC_model_mean, DSC_model_median


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy import optimize
import os

# -----Calculate T1 Map Full Image For Given Slice-----##
def create_T1_map(img, reference_mask, trigger_time):
    lenX = img.shape[0]
    lenY = img.shape[1]
    T1map = np.zeros((lenX, lenY))-1    
    Rsq_map = np.zeros((lenX, lenY))
    numPixels = 0
    RsquareSum = 0
    relevant_pixels = np.stack(np.where(reference_mask != 0), axis=1)
    for pixel_idx in range(relevant_pixels.shape[0]):
        row = relevant_pixels[pixel_idx,0]
        col = relevant_pixels[pixel_idx,1]
        T1map[row, col], Rsq_map[row, col], _ = CalcT1_CurvFit(img[row, col,:], trigger_time, PlotFit=False)
    return T1map, Rsq_map


# -----Calculate T1 Parameter For Given Pixel - Curve Fitting-----##
def T1_func(x, M0, T1):
    #return M0 * (1 - np.exp(-(1 / T1) * x))
    return M0 * (1 - 2*np.exp(-(1 / T1) * x))
    #return np.abs(M0 * (1 - 2*np.exp(-(1 / T1) * x)))

def T1_func_wo_M0(x, T1):
    return 0.115 * (1 - np.exp(-(1 / T1) * x))


def CalcT1_CurvFit(pixel_sequence, TriggerTime, PlotFit=False):
    S = pixel_sequence.squeeze()
    t = TriggerTime

    # check if all the pixels in x,y cor through the time are equal
    if len(set(S)) == 1:
        T1, r_squared, Pearson_matrix = 0, 0, 0
        M0 = 0
        PlotFit = False
    else:
        params, params_covariance = optimize.curve_fit(T1_func, t, S, p0=[150, 1200], maxfev=300000000)
        M0 = params[0] 
        T1 = params[1]
        S_fit = T1_func(t, M0, T1)

        ## Calculate Regression Performance
        Pearson_matrix = np.corrcoef(S, S_fit)
        correlation_xy = Pearson_matrix[0, 1]
        r_squared = correlation_xy ** 2
        none_flag = 0

        if np.isnan(r_squared):
            T1, r_squared, Pearson_matrix = 0, 0, 0
    

    # if (PlotFit):  ## Plot Data with Curve Fit
        # plt.figure()
        # plt.scatter(t, S, label='Data')
        # plt.plot(t, S_fit, c='r', label='Fitted T1 Function')
        # plt.legend(loc='best')
        # plt.grid()
        # plt.ylabel('Image Level')
        # plt.xlabel('time[ms]')
        # plt.show()
    if PlotFit:
        return T1, r_squared, M0,t,S,S_fit
    return T1, r_squared, M0

def show_T1_mapping(T1map, r_squared):
    plt.figure()
    plt.imshow(T1map, cmap='jet', vmin=200, vmax=2000)
    plt.title('T1Mapping')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('R Square = {:.2f}'.format(r_squared))


def calculate_R_square_full_data_set(data, reference_masks, trigger_time,t_reference=0, norm_matrix= None ,plot_flag=False):
    """
    :param  data: data set from shape of (P,H,W,T,S)
            epi_endo_masks: (P,H,W,T,S)
            trigger_time: triggers time of each slice (P,T,S).

    :return: mean R_square and array of all R_square
    """
    # back to the original intesities using the equation: I*(Max-min)+min
    if np.array([norm_matrix]).any() != None:
        m = norm_matrix[:,:,1]
        m =m[:,np.newaxis,np.newaxis,...,np.newaxis]
        M = norm_matrix[:,:,0]
        M =M[:,np.newaxis,np.newaxis,...,np.newaxis]
        data = data*(M-m)+m
    data[:,:,:,:2,:] = -1 * data[:,:,:,:2,:] 

    if len(trigger_time.shape) == 2:
        trigger_time = trigger_time[:,:,np.newaxis]
        
    patients_number = data.shape[0]
    slices_number = data.shape[4]
    r_square_list = []
    good_T1_list = []
    T1R2_array = []
    # print(patients_number)
    for p in range(0,patients_number):
        #print('patient '+ str(p))
        img = data[p,:,:,:,:] # (H,W,T,S)

        for s in range(0,slices_number):

            # T1Map and R2 calc
            reference_patient_mask = reference_masks[p,:,:,s]
            T1_map, Rsquere_map = create_T1_map(img[:,:,:,s], reference_mask=reference_patient_mask, trigger_time=trigger_time[p, :, s])
            # r_square_list.append(mean_r_square)
            
            #create tuples of R2 and T1
            T1_map_seg_ind = np.argwhere(T1_map.flatten()!=-1)
            T1_map_seg_val = T1_map.flatten()[T1_map_seg_ind]
            R2_map_seg_val = Rsquere_map.flatten()[T1_map_seg_ind]
            T1R2_itter = [(T1_map_seg_val[i][0],R2_map_seg_val[i][0]) for i in range(len(R2_map_seg_val))]
            T1R2_array.extend(T1R2_itter)

            if p==0 and s==30 and plot_flag:
                plt.figure(figsize=(25,25))
                fig,axes = plt.subplots(1,3)
                plt.subplots_adjust(wspace=1)

                plt.suptitle("patient=" +  str(p)+", slice=" + str(s) + ', $R^2 =$ ' + str(mean_r_square))
                im0 = axes[0].imshow(T1_map, vmin=0, vmax=2000,cmap='turbo')
                fig.colorbar(im0,ax=axes[0],fraction=0.046, pad=0.04)
                axes[0].set_title("T1 Mapping")
                axes[0].format_coord = Formatter(im0)
                axes[0].imshow(reference_patient_mask, cmap='jet', alpha=0.1)

                im1 = axes[1].imshow(Rsquere_map, cmap='gray')
                fig.colorbar(im1, ax=axes[1],fraction=0.046, pad=0.04)
                axes[1].set_title("R Squared")
                axes[1].imshow(reference_patient_mask, cmap='jet', alpha=0.1)
                axes[1].format_coord = Formatter(im1)

                im2 = axes[2].imshow(img[:,:,t_reference,s], cmap='gray')
                fig.colorbar(im2, ax=axes[2],fraction=0.046, pad=0.04)
                axes[2].set_title("t" + str(t_reference))
                axes[2].imshow(reference_patient_mask, cmap='jet', alpha=0.1)
                axes[2].format_coord = Formatter(im2)
                plt.show()
                
                plt.figure()
                plt.imshow(T1_map, vmin=0, vmax=2000,cmap='turbo')
                plt.colorbar()

                if np.sum(T1_map==0)>0:
                    plt.figure()
                    plt.imshow(reference_patient_mask,cmap='gray')
                    T1_map_ind_zeros_x = list(zip(*np.argwhere(T1_map == 0)))[0]
                    T1_map_ind_zeros_y = list(zip(*np.argwhere(T1_map == 0)))[1]
                    plt.scatter(T1_map_ind_zeros_y,T1_map_ind_zeros_x,s=1)
                    zero_precent = 100*np.sum(T1_map==0)/np.sum(reference_patient_mask)
                    plt.suptitle("patient " + str(p) + ", slice " + str(s) + ' ,$R^2=$' + \
                        str(round(mean_r_square,2)))
                    plt.title(str(round(zero_precent,2))+'% of T1=0')
                    plt.show()
                

    # show the T1R2 plot
    T,R= zip(*T1R2_array)
    
    mean_r_square = round(np.mean(R),6)
    median_r_square = round(np.median(R),6)
    std_r_square = round(np.std(R),4)
    # print('median r2 = ' + str(median_r_square))
    #if plot_flag:
    # print('mean r2 = ' + str(mean_r_square))
    # print('std r2 = ' + str(std_r_square))
    r2_info = np.array([median_r_square,mean_r_square,std_r_square])

    mean_T1 = round(np.mean(T),4)
    median_T1 = round(np.median(T),4)
    std_T1 = round(np.std(T),4)
    # print('median T1 = ' + str(median_T1))
    #plot_flag:
    # print('mean T1 = ' + str(mean_T1))
    # print('std T1 = ' + str(std_T1))
    T1_info = np.array([median_T1, mean_T1, std_T1])

    if plot_flag == True and np.median(reference_patient_mask)!=1:
        fig=plt.figure()
        ax=plt.axes()
        x_space = np.array([0,0.0001,0.01,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000])
        y_space = np.array([0,0.02,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
        cax=ax.hist2d(T,R,bins=[x_space,y_space],cmap="jet",norm=colors.LogNorm())
        ax.set_xlim(0,2000)
        ax.set_ylim(0,1)
        cb=fig.colorbar(cax[3])
        cb.ax.set_label("counts in bin")
        ax.set_xlabel('T1')
        ax.set_ylabel('R2')
        ax.set_title('$hist(T_1,R^2)$' + '; ' + '$median(R^2)= $'+ str(median_r_square) + '; '+ '$median(T_1)= $' + str(median_T1) + '$[ms]$')  
        plt.show()

        fig=plt.figure()
        ax=plt.axes()
        x_space = np.array([0,0.0001,0.01,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000])
        y_space = np.array([0,0.02,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
        cax=ax.hist2d(T,R,bins=[x_space,y_space],cmap="jet")
        ax.set_xlim(0,2000)
        ax.set_ylim(0,1)
        cb=fig.colorbar(cax[3])
        cb.ax.set_label("counts in bin")
        ax.set_xlabel('T1')
        ax.set_ylabel('R2')
        ax.set_title('$hist(T_1,R^2)$' + '; ' + '$mean(R^2)= $'+ str(mean_r_square) + '; '+ '$median(T_1)= $' + str(median_T1) + '$[ms]$')  
        plt.show()

    

    return r2_info, T1_info ,T1R2_array

def r2_analyze(data, reference_masks, trigger_time, norm_matrix= None, full_T1_map_flag = False, t_ref =0): 
    """
    data [P,H,W,T,S]
    reference_masks
    trigger_time
    norm_matrix
    """

    # if there is only one patient (and one slice), the dim will be 3 [W,H,T]. we should convert it to [P,W,H,T,S]
    if len(data.shape) < 5:
        data = data[np.newaxis,...,np.newaxis]
        reference_masks = reference_masks[np.newaxis,...,np.newaxis]
        trigger_time = trigger_time[np.newaxis,...,np.newaxis]
        norm_matrix = norm_matrix[np.newaxis,...]

    # back to the original intesities using the equation: I*(Max-min)+min
    if np.array([norm_matrix]).any() != None:

        m = norm_matrix[:,:,1]
        m =m[:,np.newaxis,np.newaxis,...,np.newaxis]
        M = norm_matrix[:,:,0]
        M =M[:,np.newaxis,np.newaxis,...,np.newaxis]
        data = data*(M-m)+m
    data[:,:,:,:2,:] = -1 * data[:,:,:,:2,:]

    patients_number = data.shape[0]
    slices_number = data.shape[4]
    r_square_matrix = np.empty((patients_number,slices_number, 3))
    maps = np.empty((patients_number,slices_number, data.shape[1],data.shape[2],2))
    for p in range(0,patients_number):
        for s in range(0,slices_number):

            # T1Map and R2 calc
            reference_patient_mask = reference_masks[p,:,:,t_ref,s]
            if full_T1_map_flag:
                reference_patient_mask_for_fit = np.ones_like(reference_patient_mask)
            else:
                reference_patient_mask_for_fit = reference_patient_mask
            T1_map, Rsquere_map = create_T1_map(data[p,:,:,:,s], reference_mask=reference_patient_mask_for_fit, trigger_time=trigger_time[p, :, s])
            
            Rsquere_map_with_seg = Rsquere_map[reference_patient_mask[:,:]!=0].flatten()
            if p==0 and s==0:
                Rsquere_map_with_seg_all = Rsquere_map_with_seg
            else:
                Rsquere_map_with_seg_all = np.append(Rsquere_map_with_seg_all,Rsquere_map_with_seg)
            r_square_matrix[p,s,2] = np.std(Rsquere_map_with_seg)
            r_square_matrix[p,s,1] = np.mean(Rsquere_map_with_seg)
            r_square_matrix[p,s,0] = np.median(Rsquere_map_with_seg)
            maps[p,s,:,:,0] = Rsquere_map
            maps[p,s,:,:,1] = T1_map
    
    # median/mean/std of the avg r2
    r_square_avg_info = np.array([np.median(r_square_matrix[p,s,1]), np.mean(r_square_matrix[p,s,1]),np.std(r_square_matrix[p,s,1])])
    # median/mean/std of all r2
    r_square_all_info = np.array([np.median(Rsquere_map_with_seg_all),np.mean(Rsquere_map_with_seg_all),np.std(Rsquere_map_with_seg_all)])
    return r_square_avg_info, r_square_all_info, r_square_matrix, maps         


class Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)


def main():
    # ***************** Loading *****************
    train_set_data, train_set_trigger_time, train_set_segmentation, train_set_norm, \
        test_set_data, test_set_trigger_time, test_set_segmentation, test_set_norm \
            = LoadingImages.loading_test_train_set(load_mode='np_processed',K=4)



if __name__ == '__main__':
    main()



