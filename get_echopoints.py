import numpy as np
import cv2 as cv
import albumentations as albu
import torch

color1L = [255,0,0]
color2L = [0,255,0]
color3L = [0,0,255]
color4L = [255,0,255]
color5L = [255,255,0]
color6L = [0,255,255]

color1LM = [175,25,100]
color2LM = [175,100,25]
color3LM = [25,175,100]
color4RM = [25,100,175]
color5RM = [100,175,25]
color6RM = [100,25,175]

color1R = [0, 128, 225]
color2R = [225,128,0]
color3R = [225,0,128]
color4R = [128,0,225]
color5R = [0,225,128]
color6R = [128,225,0]

color1LR = [0,50,200]
color2LR = [0,200,50]
color3LR = [50,0,200]
color4LR = [50,200,0]
color5LR = [200,50,0]
color6LR = [200,0,50]

color1RL = [88,100,255]
color2RL = [88,255,100]
color3RL = [100,88,255]
color4RL = [100,255,88]
color5RL = [255,88,100]
color6RL = [255,100,88]

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

            L_pos = np.nonzero(L_img)

            L_median = int(np.median(L_pos[1]))
            #get length of left portion of segmentation
            L_length = L_pos[0].max()-L_pos[0].min()

            #Compute for the length of each segments
            seg3 = int((1.5*L_length)/4.5)
            seg2 = int((2*L_length.max())/4.5)
            seg1 = int((2*L_length.max())/4.5)

            portion1 = L_pos[0].min() + seg3
            portion2 = portion1 + seg2
            
            #Gets the pixel coordinates of non-black pixels
            seg_img = L_img[:L_pos[0].min()+seg3,:]
            query_X, query_Y = np.where(np.any(seg_img != [0,0,0],axis=2))
            query_pts = np.column_stack((query_X,query_Y))
            area_arr.append(len(query_pts))

            ind = np.lexsort((query_pts[:,1],query_pts[:,0]))
            pts = query_pts[ind]

            seg_pos = np.nonzero(seg_img)
            seg_med = int(np.median(seg_pos[1]))

            #formula to get the first, middle and last pixels per segment
            seg3_img = np.zeros(seg_img.shape)
            seg3_img[pts[0,0],pts[0,1]] = color1L
            #seg3_img[L_pos[0].min()+int(seg3*0.5),seg_med] = color1LM
            seg3_img[pts[int(len(pts)/2),0],pts[int(len(pts)/2),1]] = color1LM
            seg3_img[pts[-1,0],pts[-1,1]] = color2L

            seg_img = L_img[L_pos[0].min()+seg3:portion1+seg2,:]
            query_X, query_Y = np.where(np.any(seg_img != [0,0,0],axis=2))
            query_pts = np.column_stack((query_X,query_Y))
            area_arr.append(len(query_pts))

            ind = np.lexsort((query_pts[:,1],query_pts[:,0]))
            pts = query_pts[ind]

            seg_pos = np.nonzero(seg_img)
            seg_med = int(np.median(seg_pos[1]))

            seg2_img = np.zeros(seg_img.shape)
            seg2_img[pts[0,0],pts[0,1]] = color3L
            #seg2_img[int(seg2*0.5),seg_med] = color2LM
            seg2_img[pts[int(len(pts)/2),0],pts[int(len(pts)/2),1]] = color2LM
            seg2_img[pts[-1,0],pts[-1,1]] = color4L

            seg_img = L_img[portion2:,:]
            query_X, query_Y = np.where(np.any(seg_img != [0,0,0],axis=2))
            query_pts = np.column_stack((query_X,query_Y))
            area_arr.append(len(query_pts))

            ind = np.lexsort((query_pts[:,1],query_pts[:,0]))
            pts = query_pts[ind]

            seg_pos = np.nonzero(seg_img)
            seg_med = int(np.median(seg_pos[1]))

            seg1_img = np.zeros(seg_img.shape)
            seg1_img[pts[0,0],pts[0,1]] = color5L
            #seg1_img[int(seg1*0.5),seg_med] = color3LM
            seg1_img[pts[int(len(pts)/2),0],pts[int(len(pts)/2),1]] = color3LM
            seg1_img[pts[-1,0],pts[-1,1]] = color6L

            L_msk = np.vstack((seg3_img,seg2_img,seg1_img))

            R_msk = w[:,pos_med:]
            R_img = masked[:,pos_med:]
            R_half = np.nonzero(R_img)
            R_cut = R_half[0].min() + int(2.5*(R_half[0].max()-R_half[0].min())/7)

            R_msk[R_half[0].min():R_cut+1,:] = [0,0,0]
            R_msk[:,:10] = [0,0,0]
            check = R_msk.copy()
            
            R_img[R_half[0].min():R_cut+1,:] = [0,0,0]
            R_img[:,:10] = [0,0,0]

            R_pos = np.nonzero(R_img)

            R_median = int(np.median(R_pos[1]))
            R_length = R_pos[0].max()-R_pos[0].min()

            seg5 = int((1.5*R_length)/4.5)
            seg6 = int((2*R_length.max())/4.5)
            seg7 = int((2*R_length.max())/4.5)

            portion1 = R_pos[0].min() + seg5
            portion2 = portion1 + seg6

            seg_img = R_img[:R_pos[0].min()+seg5,:]
            query_X, query_Y = np.where(np.any(seg_img != [0,0,0],axis=2))
            query_pts = np.column_stack((query_X,query_Y))
            area_arr.append(len(query_pts))

            ind = np.lexsort((query_pts[:,1],query_pts[:,0]))
            pts = query_pts[ind]

            seg_pos = np.nonzero(seg_img)
            seg_med = int(np.median(seg_pos[1]))

            seg5_img = np.zeros(seg_img.shape)
            seg5_img[pts[0,0],pts[0,1]] = color1R
            #seg5_img[R_pos[0].min()+int(seg5*0.5),seg_med] = color4RM
            seg5_img[pts[int(len(pts)/2),0],pts[int(len(pts)/2),1]] = color4RM
            seg5_img[pts[-1,0],pts[-1,1]] = color2R

            seg_img = R_img[R_pos[0].min()+seg5:portion1+seg6,:]
            query_X, query_Y = np.where(np.any(seg_img != [0,0,0],axis=2))
            query_pts = np.column_stack((query_X,query_Y))
            area_arr.append(len(query_pts))

            ind = np.lexsort((query_pts[:,1],query_pts[:,0]))
            pts = query_pts[ind]

            seg_pos = np.nonzero(seg_img)
            seg_med = int(np.median(seg_pos[1]))

            seg6_img = np.zeros(seg_img.shape)
            seg6_img[pts[0,0],pts[0,1]] = color3R
            #seg6_img[int(seg6*0.5),seg_med] = color5RM
            seg6_img[pts[int(len(pts)/2),0],pts[int(len(pts)/2),1]] = color5RM
            seg6_img[pts[-1,0],pts[-1,1]] = color4R

            seg_img = R_img[portion2:,:]
            query_X, query_Y = np.where(np.any(seg_img != [0,0,0],axis=2))
            query_pts = np.column_stack((query_X,query_Y))
            area_arr.append(len(query_pts))

            ind = np.lexsort((query_pts[:,1],query_pts[:,0]))
            pts = query_pts[ind]

            seg_pos = np.nonzero(seg_img)
            seg_med = int(np.median(seg_pos[1]))

            seg7_img = np.zeros(seg_img.shape)
            seg7_img[pts[0,0],pts[0,1]] = color5R
            #seg7_img[int(seg7*0.5),seg_med] = color6RM
            seg7_img[pts[int(len(pts)/2),0],pts[int(len(pts)/2),1]] = color6RM
            seg7_img[pts[-1,0],pts[-1,1]] = color6R

            R_msk = np.vstack((seg5_img,seg6_img,seg7_img))

            img = np.hstack((L_msk,R_msk))

            #cv.imshow('a',img)
            #cv.waitKey(0)

            new_arr.append(img)

        final = np.array(new_arr)#.transpose(0,3,1,2)

        final_area = np.array(area_arr).reshape(-1,6)

        features = np.concatenate((final.reshape(new_set.shape[0],-1),final_area),axis=1)
        print(features.shape)

        print(final.shape)
        print(final_area.shape)
        
        return features

    final_train = createSet(mask_train,train_set)
    final_valid = createSet(mask_valid,valid_set)
    final_test = createSet(mask_test,test_set)

    new_train = 'Trial'+str(j)+'_final_class_echo_train'
    new_valid = 'Trial'+str(j)+'_final_class_echo_valid'
    new_test = 'Trial'+str(j)+'_final_class_echo_test'

    #np.save(new_train,final_train)
    #np.save(new_valid,final_valid)
    #np.save(new_test,final_test)

    j+= 1