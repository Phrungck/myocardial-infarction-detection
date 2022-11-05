import cv2 as cv
import numpy as np
import pandas as pd
import os
import scipy.io
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

#File path for specific folders and files
mask_dir = os.path.join(os.getcwd(),"LV Ground-truth Segmentation Masks")
echo_dir = os.path.join(os.getcwd(),"HMC-QU Echos\\HMC-QU Echos")

#Loading the dataset
df = pd.read_csv('HMC_QU.csv',na_values="NaN")


#Storing dataset as numpy array
data = np.array(df)

sub_data = data[:109]

lab_arr = []

#Getting labels for stratified K Fold
for i in range(len(sub_data)):
    x = data[i][10:]
    x = x.astype('uint8')

    end = data[i][8]
    start = data[i][7]

    lab_arr.append(x)

final_labels = np.array(lab_arr)

#Label encoding for the multilabels
def get_new_labels(y):
    y_new = LabelEncoder().fit_transform([''.join(str(l)) for l in y])
    return y_new

y_new = get_new_labels(final_labels)

kf = StratifiedKFold(n_splits=5,random_state=None,shuffle=False)

j = 1
for train_index, test_index in kf.split(sub_data, y_new):
    data_train, data_test = data[train_index], data[test_index]
    data_valid, data_test = train_test_split(data_test,test_size=0.5)

    dtrain_name = 'Fold'+str(j)+'_echo'+'_train'
    dvalid_name = 'Fold'+str(j)+'_echo'+'_valid'
    dtest_name = 'Fold'+str(j)+'_echo'+'_test'
    
    np.save(dtrain_name,data_train)
    np.save(dvalid_name,data_valid)
    np.save(dtest_name,data_test)
    j += 1

print(data_train.shape,data_test.shape,data_valid.shape)