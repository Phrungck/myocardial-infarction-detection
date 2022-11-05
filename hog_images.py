import numpy as np
import cv2 as cv
import albumentations as albu
import torch

j = 1

while j <= 5:

    #echo augmented name
    train_name = 'Fold'+str(j)+'_echo_grayslice_train.npy'
    valid_name = 'Fold'+str(j)+'_echo_grayslice_valid.npy'
    test_name = 'Fold'+str(j)+'_echo_grayslice_test.npy'

    #Loading the augmented datasets
    train = np.load(train_name)
    valid = np.load(valid_name)
    test = np.load(test_name)

    #Load the predicted masks
    mtrain_name =  'Fold'+str(j)+'_echo_preds_train.npy'
    mvalid_name = 'Fold'+str(j)+'_echo_preds_valid.npy'
    mtest_name = 'Fold'+str(j)+'_echo_preds_test.npy'

    mask_train = np.load(mtrain_name)
    mask_valid = np.load(mvalid_name)
    mask_test = np.load(mtest_name)

    #transpose to shape n x 3 x w x h
    train_set = train.transpose(0,3,1,2).astype('float32')
    valid_set = valid.transpose(0,3,1,2).astype('float32')
    test_set = test.transpose(0,3,1,2).astype('float32')

    #create final images dataset based on segmentation predictions
    def createSet(gt,dset):
        
        new_set = dset.transpose(0,2,3,1)
        
        new_arr = []

        area_arr = []

        for i in range(new_set.shape[0]):
            q = new_set[i].astype('uint8')
            w = gt[i].astype('uint8')

            #apply mask on the augmented dataset. overlaps the two images
            masked = cv.bitwise_or(q,q,mask=w)

            #cv.imshow('a',masked)
            #cv.waitKey(0)

            #Creates an array which computes the positions of non-zero values
            positions = np.nonzero(masked)

            #get the median position where we will split the images
            pos_med = int(np.median(positions[1]))

            #get a copy of the masks and convert to BGR
            w = cv.cvtColor(w,cv.COLOR_GRAY2BGR)
            w = w*255

            #Remove Segment 4 in masked image
            L_msk = w[:,:pos_med]
            L_img = masked[:,:pos_med]
            L_half = np.nonzero(L_img)
            L_cut = L_half[0].min() + int(2.5*(L_half[0].max()-L_half[0].min())/7)

            #Remove segment 4 by setting pixel values to [0,0,0]
            L_msk[L_half[0].min():L_cut+1,:] = [0,0,0]
            check_l = L_msk.copy()

            L_img[L_half[0].min():L_cut+1,:] = [0,0,0]

            R_msk = w[:,pos_med:]
            R_img = masked[:,pos_med:]
            R_half = np.nonzero(R_img)
            R_cut = R_half[0].min() + int(2.5*(R_half[0].max()-R_half[0].min())/7)

            R_msk[R_half[0].min():R_cut+1,:] = [0,0,0]
            R_msk[:,:10] = [0,0,0]
            check = R_msk.copy()
            
            R_img[R_half[0].min():R_cut+1,:] = [0,0,0]
            R_img[:,:10] = [0,0,0]

            img = np.hstack((L_img,R_img))

            #cv.imshow('a',img)
            #cv.waitKey(0)

            new_arr.append(img)

        final = np.array(new_arr)#.transpose(0,3,1,2)

        #final_area = np.array(area_arr).reshape(-1,6)

        #features = np.concatenate((final.reshape(new_set.shape[0],-1),final_area),axis=1)
        
        return final

    final_train = createSet(mask_train,train_set)
    final_valid = createSet(mask_valid,valid_set)
    final_test = createSet(mask_test,test_set)

    new_train = 'Fold'+str(j)+'_final_class_hog_echo_train'
    new_valid = 'Fold'+str(j)+'_final_class_hog_echo_valid'
    new_test = 'Fold'+str(j)+'_final_class_hog_echo_test'

    #np.save(new_train,final_train)
    #np.save(new_valid,final_valid)
    #np.save(new_test,final_test)

    j+= 1

print('Done')