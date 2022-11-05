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
    train_name = 'Fold'+str(j)+'_final_class_echo_train.npy'
    valid_name = 'Fold'+str(j)+'_final_class_echo_valid.npy'
    test_name = 'Fold'+str(j)+'_final_class_echo_test.npy'

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

    #Function which groups frames based on echos
    def get_frames(dataframe,dset):

        vid_arr = []

        i = 0
        x = 0

        for f in range(len(dataframe)):

            #get the start and end frames
            start = dataframe.loc[f][7]
            end = dataframe.loc[f][8]

            total = end-start + 1

            frame_arr = []
            count = 1

            while count <= total:
                #if count in check:
                #if count == 1 or count == int(total/2) or count == total:
                img = dset[x]
                frame_arr.append(img)
                vid = np.array(frame_arr)
                vid = vid.reshape(-1)
                count += 1
                x += 1
            
            vid_arr.append(vid)

        final = np.array(vid_arr,dtype='object')

        #print(final.shape)

        return final#.reshape(-1,2,224,224,3)
        #np.save('Pre-process_sift',final)

    final_train = get_frames(dftrain,trainset)
    final_valid = get_frames(dfvalid,validset)
    final_test = get_frames(dftest,testset)

    pretrain_name = 'Fold'+str(j)+'_Preprocess_echo_corner_train'
    prevalid_name = 'Fold'+str(j)+'_Preprocess_echo_corner_valid'
    pretest_name = 'Fold'+str(j)+'_Preprocess_echo_corner_test'

    np.save(pretrain_name,final_train)
    np.save(prevalid_name,final_valid)
    np.save(pretest_name,final_test)

    j += 1

#z = final_train[0].reshape(-1,224,224,3)
print(final_train[0].shape)
print(final_train.shape)

#print(q.reshape(-1,4,3,224,224).shape)