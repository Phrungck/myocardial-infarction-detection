import cv2 as cv
import numpy as np
import pandas as pd
import os
import scipy.io
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

j = 1

while j <= 5:

    #used for loading the final, segmented images
    train_name = 'Fold'+str(j)+'_class_echo_train.npy'
    valid_name = 'Fold'+str(j)+'_class_echo_valid.npy'
    test_name = 'Fold'+str(j)+'_class_echo_test.npy'

    trainset = np.load(train_name)
    validset = np.load(valid_name)
    testset = np.load(test_name)

    #print(testset.shape)

    dtrain_name = 'Fold'+str(j)+'_echo'+'_train.npy'
    dvalid_name = 'Fold'+str(j)+'_echo'+'_valid.npy'
    dtest_name = 'Fold'+str(j)+'_echo'+'_test.npy'

    dftrain = pd.DataFrame(np.load(dtrain_name,allow_pickle=True))
    dfvalid = pd.DataFrame(np.load(dvalid_name,allow_pickle=True))
    dftest = pd.DataFrame(np.load(dtest_name,allow_pickle=True))

    def get_frames(dataframe,dset):

        vid_arr = []

        i = 0
        x = 0

        for f in range(len(dataframe)):

            #get start and end frames from annotations
            start = dataframe.loc[f][7]
            end = dataframe.loc[f][8]

            total = end-start + 1

            frame_arr = []
            count = 1
            
            vid = 0

            q1 = int(0.2*total)
            q2 = int(0.3*total)
            q3 = int(0.4*total)
            q4 = int(0.5*total)
            q5 = int(0.6*total)
            q6 = int(0.7*total)
            q7 = int(0.8*total)
            q8 = int(0.9*total)
            check = [1,q1,q2,q3,q4,q5,q6,q7,q8,total]

            if total < 14:
                while count <= 10:
                    img = dset[x]
                    frame_arr.append(img)
                    vid = np.array(frame_arr)
                    vid = vid.reshape(-1)
                    #vid += img
                    count += 1
                    x += 1
            else:
                while count <= total:
                    if count in check:
                        img = dset[x]
                        frame_arr.append(img)
                        vid = np.array(frame_arr)
                        vid = vid.reshape(-1)
                        #vid += img
                    count += 1
                    x += 1
            
            vid_arr.append(vid)

        final = np.array(vid_arr)
        print(final.shape)

        return final.reshape(-1,10,224,224,3)
        #np.save('Pre-process_sift',final)

    final_train = get_frames(dftrain,trainset)
    final_valid = get_frames(dfvalid,validset)
    final_test = get_frames(dftest,testset)

    pretrain_name = 'Trial'+str(j)+'_Preprocess_echo_hog_train'
    prevalid_name = 'Trial'+str(j)+'_Preprocess_echo_hog_valid'
    pretest_name = 'Trial'+str(j)+'_Preprocess_echo_hog_test'

    np.save(pretrain_name,final_train)
    np.save(prevalid_name,final_valid)
    np.save(pretest_name,final_test)
    j += 1

print('Done')