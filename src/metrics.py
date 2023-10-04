

def DSC_loss_np(y_pred,y_true,printflag=False):
    """
    y_pred [P,H,W,T]
    y_true [P,H,W,T]
    """
    vol_axes = (1,2)
    top = 2 * np.sum(y_true * y_pred,axis=vol_axes)
    bottom = np.clip(np.sum(y_true + y_pred,axis=vol_axes), 1e-5,None)
    dice_all = (top / bottom)
    dice = np.mean(dice_all)
    if printflag:
        print('dice mean = {}'.format(np.mean(top / bottom)))
        print('dice median = {}'.format(np.median(dice_all)))
        print('dice std = {}'.format(np.std(dice_all)))
    dice_info = [np.median(top / bottom), np.mean(top / bottom), np.std(top / bottom)]
    return -dice, dice_info, dice_all


def Hausdorff_distance(seg,ref):
    """
    seg [P,H,W,T,S]
    reg [P,H,W,T,S]
    """
    hd_all = np.zeros_like(seg)
    # hd_95_all = np.zeros_like(seg)
    for p in range(seg.shape[0]):
        for t in range(seg.shape[3]):
            for s in range(seg.shape[4]):
                hd_all[p,:,:,t,s] = metrics.hausdorff_distance(seg[p,:,:,t,s],ref[p,:,:,t,s],voxel_spacing=2.1)
                # hd_95_all[p,:,:,t,s] = metrics.hausdorff_distance_95(seg[p,:,:,t,s],ref[p,:,:,t,s],voxel_spacing=2.1)
    hd_info = [np.median(hd_all), np.mean(hd_all), np.std(hd_all)]
    # hd_95_info = [np.median(hd_95_all), np.mean(hd_95_all), np.std(hd_95_all)]
    return hd_info, hd_all#, hd_95_info,hd_95_all

def Hausdorff_distance_95(seg,ref):
    """
    seg [P,H,W,T,S]
    reg [P,H,W,T,S]
    """
    # hd_all = np.zeros_like(seg)
    hd_95_all = np.zeros_like(seg)
    for p in range(seg.shape[0]):
        for t in range(seg.shape[3]):
            for s in range(seg.shape[4]):
                # hd_all[p,:,:,t,s] = metrics.hausdorff_distance(seg[p,:,:,t,s],ref[p,:,:,t,s],voxel_spacing=2.1)
                hd_95_all[p,:,:,t,s] = metrics.hausdorff_distance_95(seg[p,:,:,t,s],ref[p,:,:,t,s],voxel_spacing=2.1)
    # hd_info = [np.median(hd_all), np.mean(hd_all), np.std(hd_all)]
    hd_95_info = [np.median(hd_95_all), np.mean(hd_95_all), np.std(hd_95_all)]
    return hd_95_info,hd_95_all
