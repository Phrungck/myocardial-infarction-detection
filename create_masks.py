import cv2 as cv
import numpy as np
import pandas as pd
import os
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

#File path for specific folders and files
mask_dir = os.path.join(os.getcwd(),"LV Ground-truth Segmentation Masks")
echo_dir = os.path.join(os.getcwd(),"HMC-QU Echos\\HMC-QU Echos")

#Otsu augmentation function
def augmentotsu(dset):

    arr = []

    for i in range(dset.shape[0]):
        x = dset[i]
        gray = cv.cvtColor(x,cv.COLOR_BGR2GRAY)

        ret2,th2 = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

        new = cv.cvtColor(th2,cv.COLOR_GRAY2BGR)
        arr.append(new)
        
    final = np.array(arr)
    
    return final

#Gray level slicing function
def augmentgray(dset):

    arr = []

    for i in range(dset.shape[0]):
        x = dset[i]
        gray = cv.cvtColor(x,cv.COLOR_BGR2GRAY)

        #For Gray-level slicing
        min = np.mean(gray)/1.5
        max = np.amax(gray)

        row, col = gray.shape

        gray_slice = np.zeros((row,col),dtype='uint8')

        for i in range(row):
            for j in range(col):
                if gray[i,j]>min and gray[i,j]<max:
                    gray_slice[i,j]=gray[i,j]
                else:
                    gray_slice[i,j]=0

        new = cv.cvtColor(gray_slice,cv.COLOR_GRAY2BGR)
        arr.append(new)
        
    final = np.array(arr)
    
    return final

#Gaussian blurring and gray level slicing function
def augmentgrayblur(dset):

    arr = []

    for i in range(dset.shape[0]):
        x = dset[i]
        gray = cv.cvtColor(x,cv.COLOR_BGR2GRAY)

        blur = cv.GaussianBlur(gray,(3,3),0)

        #For Gray-level slicing
        min = np.mean(blur)/1.5 #minimum is set to value which is mean divided by 1.5
        max = np.amax(blur)

        row, col = blur.shape

        gray_slice = np.zeros((row,col),dtype='uint8')

        for i in range(row):
            for j in range(col):
                if blur[i,j]>min and blur[i,j]<max:
                    gray_slice[i,j]=blur[i,j]
                else:
                    gray_slice[i,j]=0

        new = cv.cvtColor(gray_slice,cv.COLOR_GRAY2BGR)
        arr.append(new)
        
    final = np.array(arr)
    
    return final

#blurring function
def augmentblur(dset):

    arr = []

    for i in range(dset.shape[0]):
        x = dset[i]
        gray = cv.cvtColor(x,cv.COLOR_BGR2GRAY)

        blur = cv.GaussianBlur(gray,(3,3),0)

        new = cv.cvtColor(blur,cv.COLOR_GRAY2BGR)
        arr.append(new)
        
    final = np.array(arr)
    
    return final

#get the frames of echo videos
def get_frames(dataframe,data_arr):

    #resize dimensions to 224 x 224
    w = 224
    h = 224

    #Arrays for storing
    frame_arr = []
    mat_arr = np.zeros((1,w,h))

    x = 0

    for i in range(data_arr.shape[0]):
        echo_path = os.path.join(echo_dir,dataframe.loc[i][0]+'.avi')

        vid = cv.VideoCapture(echo_path)

        #Get number of frames with annotations for each echo
        start = dataframe.loc[i][7] - 1
        end = dataframe.loc[i][8] - 1

        count = 0
        
        while True:
            try:
                ret, frame = vid.read()

                #Append frames into an array within the annotated frames duration
                if count >= start and count <= end:
                    dsize = (w,h)
                    crop_frame = cv.resize(frame,dsize)
                    frame_arr.append(crop_frame)
                    #cv.imwrite(str(x)+'.jpg',crop_frame)
                    x += 1
                if ret == False:
                    break

                count += 1

            except:
                break
    #This array consists of the frames of an echo
    final_image = np.array(frame_arr)

    #Getting the ground truth masks
    for i in range(data_arr.shape[0]):
        mask_path = os.path.join(mask_dir,'Mask_'+dataframe.loc[i][0]+'.mat')

        mask_mat = scipy.io.loadmat(mask_path)

        mask_mat = mask_mat['predicted']

        mat_arr = np.append(mat_arr,mask_mat)

    mat_arr = mat_arr.reshape(-1,w,h)

    #This array represents the ground truth masks for each frames in the previous array
    final_mask = mat_arr[1:]

    lab_arr = []

    for i in range(data_arr.shape[0]):
        x = data_arr[i][10:]
        x = x.astype('uint8')

        end = data_arr[i][8]
        start = data_arr[i][7]
        count = end-start+1
        
        dummy = 1

        while count >= dummy:
            lab_arr.append(x)
            dummy += 1
    
    #This stores the labels for each echo based on the number of frames
    final_labels = np.array(lab_arr)

    return final_image, final_mask, final_labels

j = 1

while j<=5:
    dtrain_name = 'Fold'+str(j)+'_echo'+'_train.npy'
    dvalid_name = 'Fold'+str(j)+'_echo'+'_valid.npy'
    dtest_name = 'Fold'+str(j)+'_echo'+'_test.npy'

    #Loading the dataframes stored as numpy files
    train = np.load(dtrain_name,allow_pickle=True)
    valid = np.load(dvalid_name,allow_pickle=True)
    test = np.load(dtest_name,allow_pickle=True)

    dftrain = pd.DataFrame(train)
    dfvalid = pd.DataFrame(valid)
    dftest = pd.DataFrame(test)

    #Function to get the frames, augmentations, etc.
    ei_train,em_train,el_train = get_frames(dftrain,train)
    ea_otsu_train = augmentotsu(ei_train)
    ea_grayslice_train = augmentgray(ei_train)
    ea_blurgray_train = augmentgrayblur(ei_train)
    ea_blur_train = augmentblur(ei_train)

    ei_valid,em_valid,el_valid = get_frames(dfvalid,valid)
    ea_otsu_valid = augmentotsu(ei_valid)
    ea_grayslice_valid = augmentgray(ei_valid)
    ea_blurgray_valid = augmentgrayblur(ei_valid)
    ea_blur_valid = augmentblur(ei_valid)

    ei_test,em_test,el_test = get_frames(dftest,test)
    ea_otsu_test = augmentotsu(ei_test)
    ea_grayslice_test = augmentgray(ei_test)
    ea_blurgray_test = augmentgrayblur(ei_test)
    ea_blur_test = augmentblur(ei_test)

    #Naming the files
    dtrain_name = 'Fold'+str(j)+'_echo_img_train'
    dvalid_name = 'Fold'+str(j)+'_echo_img_valid'
    dtest_name = 'Fold'+str(j)+'_echo_img_test'
    ltrain_name = 'Fold'+str(j)+'_echo_lab_train'
    lvalid_name = 'Fold'+str(j)+'_echo_lab_valid'
    ltest_name = 'Fold'+str(j)+'_echo_lab_test'
    mtrain_name =  'Fold'+str(j)+'_echo_msk_train'
    mvalid_name = 'Fold'+str(j)+'_echo_msk_valid'
    mtest_name = 'Fold'+str(j)+'_echo_msk_test'

    otsu_train_name =  'Fold'+str(j)+'_echo_otsu_train'
    otsu_valid_name = 'Fold'+str(j)+'_echo_otsu_valid'
    otsu_test_name = 'Fold'+str(j)+'_echo_otsu_test'

    grayslice_train_name =  'Fold'+str(j)+'_echo_grayslice_train'
    grayslice_valid_name = 'Fold'+str(j)+'_echo_grayslice_valid'
    grayslice_test_name = 'Fold'+str(j)+'_echo_grayslice_test'

    blurgray_train_name =  'Fold'+str(j)+'_echo_blurgray_train'
    blurgray_valid_name = 'Fold'+str(j)+'_echo_blurgray_valid'
    blurgray_test_name = 'Fold'+str(j)+'_echo_blurgray_test'
    
    blur_train_name =  'Fold'+str(j)+'_echo_blur_train'
    blur_valid_name = 'Fold'+str(j)+'_echo_blur_valid'
    blur_test_name = 'Fold'+str(j)+'_echo_blur_test'

    #Saving the files
    #np.save(dtrain_name,ei_train)
    #np.save(ltrain_name,el_train)
    #np.save(mtrain_name,em_train)
    #np.save(otsu_train_name,ea_otsu_train)
    #np.save(grayslice_train_name,ea_grayslice_train)
    #np.save(blurgray_train_name,ea_blurgray_train)
    #np.save(blur_train_name,ea_blur_train)

    #np.save(dvalid_name,ei_valid)
    #np.save(lvalid_name,el_valid)
    #np.save(mvalid_name,em_valid)
    #np.save(otsu_valid_name,ea_otsu_valid)
    #np.save(grayslice_valid_name,ea_grayslice_valid)
    #np.save(blurgray_valid_name,ea_blurgray_valid)
    #np.save(blur_valid_name,ea_blur_valid)

    #np.save(dtest_name,ei_test)
    #np.save(ltest_name,el_test)
    #np.save(mtest_name,em_test)
    #np.save(otsu_test_name,ea_otsu_test)
    #np.save(grayslice_test_name,ea_grayslice_test)
    #np.save(blurgray_test_name,ea_blurgray_test)
    #np.save(blur_test_name,ea_blur_test)
    #break
    j += 1

print('Done')