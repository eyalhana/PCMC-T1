import numpy as np
from skimage.transform import resize
import os
from sklearn.model_selection import KFold
import mat73

def loading_test_train_set(config, fold = None):
    if fold is not None:
        config['experiment']['fold']  = fold
        
    if not os.path.exists(os.path.join(config['path']['data_dir_path'], 'full_set_data.npy')):

        # --------------Save Original Images from MAT File-------------#
        T1Dataset210_dict = mat73.loadmat(os.path.join(config['path']['data_dir_path'],'T1Dataset210.mat'))
        Dataset_dict = T1Dataset210_dict['Dataset']
        Images = Dataset_dict['Img']
        Images = np.stack(Images, axis=0)[:, 0, :, :, :, :]
        EndoContours = Dataset_dict['EndoContours']
        EndoContours = np.stack(EndoContours, axis=0)[:, 0, :, :, :, :]
        EpiContours = Dataset_dict['EpiContours']
        EpiContours = np.stack(EpiContours, axis=0)[:, 0, :, :, :, :]
        # --------------------Save Trigger Time-----------------------#
        TriggerTimes = Dataset_dict['TriggerTime']
        TriggerTimes = np.stack(TriggerTimes, axis=0)[:, 0, :, :]
        # ------------------------Crop Images and contours-------------------------#
        segmentation = preprocess_contours(EpiContours,EndoContours, [Images.shape[1],Images.shape[2]])
        del Dataset_dict; del T1Dataset210_dict
        Images, norm_matrix = normImages(Images)
        seg_mask_all = np.where(np.sum(np.sum(np.sum(segmentation[:,:,:,:,:],axis=0),axis=2),axis=2) > 1 , 1 , 0)
        segmentation, _ = crop_images(segmentation, seg_mask_all=seg_mask_all)
        DownSampleImages, numPatients = crop_images(Images, seg_mask_all=seg_mask_all)
        del Images;  del seg_mask_all

        np.save(os.path.join(config['path']['data_dir_path'], 'full_set_data.npy'),DownSampleImages)
        np.save(os.path.join(config['path']['data_dir_path'], 'full_set_trigger_time.npy'), TriggerTimes)
        np.save(os.path.join(config['path']['data_dir_path'], 'full_set_segmentation.npy'), segmentation)
        np.save(os.path.join(config['path']['data_dir_path'], 'full_norm_matrix.npy'), norm_matrix)


    data_set = np.load(os.path.join(config['path']['data_dir_path'], 'full_set_data.npy'))
    trigger_set = np.load(os.path.join(config['path']['data_dir_path'], 'full_set_trigger_time.npy'))
    segmentation_set= np.load(os.path.join(config['path']['data_dir_path'], 'full_set_segmentation.npy'))
    norm_matrix_set = np.load(os.path.join(config['path']['data_dir_path'], 'full_norm_matrix.npy'))

    data = {}
    if config['experiment']['fold'] =='all':
        data['test_set_data'] = data_set
        data['test_set_trigger_time'] = trigger_set
        data['test_set_segmentation'] = segmentation_set
        data['test_set_norm_matrix'] = norm_matrix_set
        return data

    K_f  = KFold(n_splits=5)      
    for i,(train_ind, test_ind) in enumerate(K_f.split(np.arange(data_set.shape[0]))):
        if i==config['experiment']['fold']:
            break
    
    val_len = round(0.1 * len(train_ind))
    val_ind = train_ind[val_len:]
    train_ind = train_ind[:val_len]

    data['train_set_data'] = data_set[train_ind,...]
    data['train_set_trigger_time'] = trigger_set[train_ind,...]
    data['train_set_segmentation'] = segmentation_set[train_ind,...]
    data['train_set_norm']= norm_matrix_set[train_ind,...]

    data['validate_set_data'] = data_set[val_ind,...]
    data['validate_set_trigger_time'] = trigger_set[val_ind,...]
    data['validate_set_segmentation'] = segmentation_set[val_ind,...]
    data['validate_set_norm'] = norm_matrix_set[val_ind,...]

    data['test_set_data'] = data_set[test_ind,...]
    data['test_set_trigger_time'] = trigger_set[test_ind,...]
    data['test_set_segmentation'] = segmentation_set[test_ind,...]
    data['test_set_norm'] = norm_matrix_set[test_ind,...]

    data['train_ind'] = train_ind
    data['val_ind'] = val_ind
    data['test_ind'] = test_ind 

    return data



def down_sample_image(Img, xDownSampFactor, yDownSampFactor):
    return resize(Img, (Img.shape[0] // yDownSampFactor, Img.shape[1] // xDownSampFactor,
                        Img.shape[2], Img.shape[3]), anti_aliasing=True)


def crop_images(images, xDownSampFactor=2, yDownSampFactor=2, seg_mask_all=None):
    # ------------------------Crop Images-------------------------#
    # --Set Cropping Limits(ROI)--#
    if seg_mask_all is not None:
        coordinates = np.argwhere(seg_mask_all>0)
        pyStart = np.min(coordinates[:,1])
        pyStop = np.max(coordinates[:,1])+1
        pxStart = np.min(coordinates[:,0])
        pxStop = np.max(coordinates[:,0])+1
        width = np.max([pyStop-pyStart, pxStop-pxStart])
        
        # dilated bounding box
        width = width + 5 # safety margin
        width_options = 16*np.arange(8,15)
        #width_options = 256
        extra_width = np.min(np.where(width_options-width >0,width_options-width,900))
        new_width = width+extra_width

        py_middle = pyStart+int((pyStop - pyStart)/2)
        px_middle = pxStart+int((pxStop - pxStart)/2)

        pyStart = int(py_middle-np.floor(new_width/2))
        pyStop = int(py_middle+np.ceil(new_width/2))
        pxStart = int(px_middle-np.floor(new_width/2))
        pxStop = int(px_middle+np.ceil(new_width/2))
        
        cropped_img = images[:,pxStart:pxStop,pyStart:pyStop,:,:]

        numPatients = cropped_img.shape[0]
    else:
        cropped_img = images
        numPatients = cropped_img.shape[0]
    return cropped_img, numPatients

def normImages(Images, Normilze_per_t=True):
        normalizaed_images = np.zeros_like(Images)
        # (P,T,(Max,Min))
        norm_matrix =np.zeros((Images.shape[0],Images.shape[3],2))
        for i in range(Images.shape[0]):
            normalizaed_images[i,:], norm_array = normImage(Images[i, ...])
            norm_matrix[i,:] = norm_array

        return normalizaed_images, norm_matrix

def normImage(Img):

    norm_ver = 1
    norm_array = np.zeros((Img.shape[2],2))
    if norm_ver == 1:
        normImg = np.zeros_like(Img)
        for t in range(Img.shape[2]):
            normImg[:, :, t, :] = (Img[:, :, t, :] - np.min(Img[:, :, t, :])) / (
                        np.max(Img[:, :, t, :]) - np.min(Img[:, :, t, :]))
            norm_array[t,0] = np.max(Img[:, :, t, :])
            norm_array[t,1] = np.min(Img[:, :, t, :])

    elif norm_ver == 2:
        normImg = np.zeros_like(Img)
        for t in range(Img.shape[2]):
            normImg[:, :, t, :] = 2*(Img[:, :, t, :] - np.min(Img[:, :, t, :])) / (
                        np.max(Img[:, :, t, :]) - np.min(Img[:, :, t, :]))-1  
            norm_array[t,0] = np.max(Img[:, :, t, :])
            norm_array[t,1] = np.min(Img[:, :, t, :])

    return normImg, norm_array

def preprocess_contours(epi, endo, img_shape):
    """
    find the segmentation between the epi contour and the endo contour
    """
    epi_images = creating_contours_matrix(epi, img_shape)
    endo_images = creating_contours_matrix(endo, img_shape)
    seg_images = epi_images-endo_images
    seg_images = np.where(seg_images>0,1,0)
    return seg_images

def creating_contours_matrix(contours, img_shape):
    """
    create 2D images of the contours
    :param contours: shape [p,60,2,t,s]
    :param img_shape: shape[h,w]
    :return: contours_matrix: shape [p,h/2,w/2,t,s]
    """
    # running over all the slices/times

    contours_matrix = np.zeros((contours.shape[0],img_shape[0],img_shape[1],contours.shape[3],contours.shape[4]))
    for p in range(contours.shape[0]):
        for s in range(contours.shape[4]):
            for t in range(contours.shape[3]):
                contour = contours[p,:,:,t,s]
                inside_coordinates = get_contour_inside_pixel(img_shape, contour)
                fill_contour_mat = np.zeros(img_shape)
                fill_contour_mat[inside_coordinates[:, 0], inside_coordinates[:, 1]] = 255
                contours_matrix[p, :, :, t, s] = fill_contour_mat
    return contours_matrix

def get_contour_inside_pixel(img_shape, contour):
    import cv2
    img = np.zeros(img_shape)
    contour_cv2 = contour.astype(np.int32)
    cv2.drawContours(img, [contour_cv2], -1, color= 255, thickness=cv2.FILLED)
    pixels = np.where(img == 255)
    pixels = np.stack(pixels, axis=1)
    return pixels

