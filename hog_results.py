from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import pandas as pd

def getHog(dataset):

    #array for storing the results
    arr = []

    #loop over the folds. Type is object
    for i in range(len(dataset)):
        images = dataset[i] #Dataset consists of varying frames that are flattened into a vector
        images = images.reshape(-1,224,224,3) #converting to frames with image dimensions

        #stores hog results per echo
        #hog_arr = np.zeros(1568)
        hog_arr = []

        for j in range(len(images)):
            img = images[j]
            #img = cv.resize(img,(64,128))

            fd, hd = hog(img,orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1),visualize=True,multichannel=True)
            
            hd = cv.resize(hd,(224,224))
            cv.imshow('a',hd)
            cv.waitKey(0)
            hog_arr = np.concatenate((hog_arr,fd),axis=0)
            #hog_arr = np.vstack((hog_arr,fd))
            
        #hog_arr = hog_arr[1:]
        #hog_arr = np.mean(hog_arr,axis=0)
        #print(hog_arr.shape)

        arr.append(hog_arr)

    final = np.array(arr)
    print(final.shape)

    return final

def getLabels(dataFrame):

    data_arr = np.array(dataFrame)

    lab_arr = []

    for i in range(data_arr.shape[0]):
        x = data_arr[i][10:]
        x = x.astype('uint8')

        lab_arr.append(x)
    
    #This stores the labels for each echo based on the number of frames
    final_labels = np.array(lab_arr)

    return final_labels

j = 1
while j <= 5:

    dtrain_name = 'Fold'+str(j)+'_echo'+'_train.npy'
    dvalid_name = 'Fold'+str(j)+'_echo'+'_valid.npy'
    dtest_name = 'Fold'+str(j)+'_echo'+'_test.npy'

    dftrain = pd.DataFrame(np.load(dtrain_name,allow_pickle=True))
    dfvalid = pd.DataFrame(np.load(dvalid_name,allow_pickle=True))
    dftest = pd.DataFrame(np.load(dtest_name,allow_pickle=True))

    train_lab = getLabels(dftrain)
    valid_lab = getLabels(dfvalid)
    test_lab = getLabels(dftest)

    train_lab_name = 'Fold'+str(j)+'_echo_hog_lab_train.npy'
    valid_lab_name = 'Fold'+str(j)+'_echo_hog_lab_valid.npy'
    test_lab_name = 'Fold'+str(j)+'_echo_hog_lab_test.npy'

    train_name = 'Fold'+str(j)+'_Preprocess_echo_hog_train.npy'
    valid_name = 'Fold'+str(j)+'_Preprocess_echo_hog_valid.npy'
    test_name = 'Fold'+str(j)+'_Preprocess_echo_hog_test.npy'

    aug_train = np.load(train_name)#,allow_pickle=True)
    aug_valid = np.load(valid_name)#,allow_pickle=True)
    aug_test = np.load(test_name)#,allow_pickle=True)
    
    train = getHog(aug_train)
    #break
    valid = getHog(aug_valid)
    test = getHog(aug_test)

    fold_train = 'Fold'+str(j)+'_final_echo_hog_train'
    fold_valid = 'Fold'+str(j)+'_final_echo_hog_valid'
    fold_test = 'Fold'+str(j)+'_final_echo_hog_test'

    np.save(fold_train,train)
    np.save(fold_valid,valid)
    np.save(fold_test,test)

    np.save(train_lab_name,train_lab)
    np.save(valid_lab_name,valid_lab)
    np.save(test_lab_name,test_lab)
    j+=1

print('Done')