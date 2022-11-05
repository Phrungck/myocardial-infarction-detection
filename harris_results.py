import cv2 as cv
import numpy as np
import pandas as pd
import os
import scipy.io
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

red = [125,8,200]

j = 1

#Function to get labels from a particular dataframe
def getLabels(dataframe):
    arr = [] #empty array

    for i in range(len(dataframe)): #gets length of a particular dataframe
        lab = dataframe.loc[i][10:] #gets the labels of a particular echo
        arr.append(lab)
        
    final = np.array(arr)

    return final.astype('uint8') #outputs the set of labels for a particular set

#Function to convert keypoints
def keyConvert(points):
    arr = []

    for i in points:
        x, y = i
        x, y = float(x), float(y)

        kp = cv.KeyPoint(y,x,10)
        arr.append(kp)
    
    return arr

#Function to get features based on patch sizes
def getFeatures(gray_img, points,half,row,col):

    arr = []

    for i in points:
        x, y = i
        x, y = int(x), int(y)

        #Checking if outside image range
        if x-half < 0:
            beginX = 0
            endX = beginX + size
        elif x+half > row:
            endX = row
            beginX = endX - size
        else:
            beginX = x-half
            endX = x+half

        if y-half < 0:
            beginY = 0
            endY = beginY + size
        elif y+half > col:
            endY = col
            beginY = endY - size
        else:
            beginY = y-half
            endY = y+half

        patch = gray_img[beginX:endX,beginY:endY]
        patch = patch.reshape(-1)
        arr = np.append(arr,patch)

    return arr

#This is a function to retrieve matching sift points between two images
def getSift(img1,img2):
    row, col,_ = img1.shape

    _X, _Y = np.where(np.any(img1 != [0,0,0],axis=2))
    
    base_ref = np.column_stack((_Y,_X))

    #Computes for edges using Harris Corner Detector up to subpixel accuracy
    #Query image
    img1_copy = img1.copy()
    g_query_img = cv.cvtColor(img1_copy,cv.COLOR_BGR2GRAY)
    query_dst = cv.cornerHarris(g_query_img, 2, 3, 0.01)
    #query_dst = cv.dilate(query_dst,None)
    ret, query_dst = cv.threshold(query_dst,0.0001*query_dst.max(),255,0)
    query_dst = np.uint8(query_dst)

    #subpixel accuracy codes
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(query_dst)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.01)
    corners = cv.cornerSubPix(g_query_img,np.float32(centroids),(1,1),(-1,-1),criteria)
    base_res = np.int0(corners)

    img1_copy[base_res[1:,1],base_res[1:,0]] = red
    query_X, query_Y = np.where(np.any(img1_copy==red,axis=2))
    query_pts = np.column_stack((query_X,query_Y)) #Get all points with red values
    query_pts = np.float32(query_pts)

    #computes for Train image detectors
    g_train_img = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    train_dst = cv.cornerHarris(g_train_img, 2, 3, 0.01)
    #train_dst = cv.dilate(train_dst,None)
    ret, train_dst = cv.threshold(train_dst,0.0001*train_dst.max(),255,0)
    train_dst = np.uint8(train_dst)

    ret, labels, stats, centroids = cv.connectedComponentsWithStats(train_dst)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.01)
    corners = cv.cornerSubPix(g_train_img,np.float32(centroids),(1,1),(-1,-1),criteria)
    res = np.int0(corners)

    #subpixel accuracy codes
    img2[res[1:,1],res[1:,0]] = red
    train_X, train_Y = np.where(np.any(img2==red,axis=2))
    train_pts = np.column_stack((train_X,train_Y))
    train_pts = np.float32(train_pts)

    #keypoint conversion
    kpsTrain = []
    kpsQuery = []

    kpsTrain = keyConvert(train_pts)
    kpsQuery = keyConvert(query_pts)

    #patch size
    size = 2
    half = int(size/2)

    row0,_ = query_pts.shape  
    row1,_ = train_pts.shape

    #descriptor arrays
    query_ft = []
    train_ft = []

    query_ft = getFeatures(g_query_img, query_pts,half,row,col)
    train_ft = getFeatures(g_train_img, train_pts,half,row,col)

    #reshaping the vectorized patches
    query_ft = query_ft.reshape(row0,-1)
    train_ft = train_ft.reshape(row1,-1)

    query_ft = np.float32(query_ft)
    train_ft = np.float32(train_ft)

    #computing the Euclidean distance
    bf = cv.BFMatcher(cv.NORM_L2,crossCheck=True)
    matches = bf.match(train_ft,query_ft)
    matches = sorted(matches, key = lambda x:x.distance)

    #stores keypoints
    kpsT = np.float32([kp.pt for kp in kpsTrain])
    kpsQ = np.float32([kp.pt for kp in kpsQuery])

    #get initial matched points
    ptsA = np.float32([kpsT[m.queryIdx] for m in matches])
    ptsB = np.float32([kpsQ[m.trainIdx] for m in matches])

    #enchance image matches using find homography
    h_mat, h_mask = cv.findHomography(ptsA, ptsB,cv.RANSAC,4)
    h_maskMatches = h_mask.ravel().tolist()

    new_matches = []

    for i in range(len(h_maskMatches)):
        check = h_maskMatches[i]

        if check > 0:
            new_matches.append(matches[i])

    ptsA = np.float32([kpsT[m.queryIdx] for m in new_matches])
    ptsB = np.float32([kpsQ[m.trainIdx] for m in new_matches])

    img3 = cv.drawMatches(img2,kpsTrain,img1,kpsQuery,new_matches,
                        None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


    #cv.imshow('a',img3)
    #cv.waitKey(0)

    return ptsA, ptsB, base_ref

#Function to calculate euclidean distances between two matched points
def siftDistances(dset):

    arr = []

    #Nested array which first takes the length of a dataset
    #Each dataset consists of 3 channels with 5 frames with shape of (Length,5,3,224,224)
    r = 0
    for i in range(len(dset)):

        datapoints = 18
        max_arr = np.ones(datapoints)

        #224 x 224 x 3 + 6
        data = dset[i].reshape(-1,224*224*3+6)

        base = data[0]
        base_area = base[-6:]

        #max_arr = np.append(max_arr,base_area)

        base_image = base[:-6]
        base_image = base_image.reshape(224,224,3).astype('uint8')

        #base frame
        j = 0

        #Distances between pairs of images. Total of 4 sets of 5 or 20 features
        while j < len(data) - 1:

            compare = data[j+1]
            compare_area = compare[-6:]

            compare_img = compare[:-6]
            compare_img = compare_img.reshape(224,224,3).astype('uint8')

            ptsA, ptsB, ref = getSift(base_image,compare_img)

            #function which rearranges the points to match ref
            ref = ref[ref[:,0].argsort()]
            l_ref = ref[:int(datapoints/2)]
            l_ref = l_ref[l_ref[:,1].argsort()]
            r_ref = ref[int(datapoints/2):]
            r_ref = r_ref[r_ref[:,1].argsort()]
            ref = np.vstack((l_ref,r_ref))

            #L2-norm
            d = np.linalg.norm(ptsA-ptsB,axis=-1)

            #Determine the location of the matched point in an array
            for idx in range(len(ref)):
                index = np.where(np.all((ptsB==ref[idx]),axis=1))[0]

                #Stores the best value
                if index.size > 0 and max_arr[idx] < d[index]:
                    max_arr[idx] = d[index]

            j += 1
        
        arr.append(max_arr)

    features = np.array(arr)

    return features

while j <= 5:

    #Loads the numpy files with columns showing echo names, segments, cycles, etc.
    #Already split through 5 fold cross validation
    dtrain_name = 'Fold'+str(j)+'_echo'+'_train.npy'
    dvalid_name = 'Fold'+str(j)+'_echo'+'_valid.npy'
    dtest_name = 'Fold'+str(j)+'_echo'+'_test.npy'

    #Store as dataframe
    dftrain = pd.DataFrame(np.load(dtrain_name,allow_pickle=True)) #dataframe for training set
    dfvalid = pd.DataFrame(np.load(dvalid_name,allow_pickle=True)) #dataframe for validation set
    dftest = pd.DataFrame(np.load(dtest_name,allow_pickle=True)) #dataframe fro testing set

    #Loading names numpy array containing particular frames from an echo
    train_name = 'Fold'+str(j)+'_Preprocess_echo_corner_train.npy'
    valid_name = 'Fold'+str(j)+'_Preprocess_echo_corner_valid.npy'
    test_name = 'Fold'+str(j)+'_Preprocess_echo_corner_test.npy'

    #Loading the numpy files
    trainset = np.load(train_name,allow_pickle=True)
    validset = np.load(valid_name,allow_pickle=True)
    testset = np.load(test_name,allow_pickle=True)

    #print(trainset.shape,validset.shape)

    train_labels_name = 'Fold'+str(j)+'_echo_corner_lab_train'
    valid_labels_name = 'Fold'+str(j)+'_echo_corner_lab_valid'
    test_labels_name = 'Fold'+str(j)+'_echo_corner_lab_test'

    train_lab = getLabels(dftrain)
    valid_lab = getLabels(dfvalid)
    test_lab = getLabels(dftest)

    #saves labels
    np.save(train_labels_name,train_lab)
    np.save(valid_labels_name,valid_lab)
    np.save(test_labels_name,test_lab)

    train_data_name = 'Fold'+str(j)+'_final_echo_corner_train'
    valid_data_name = 'Fold'+str(j)+'_final_echo_corner_valid'
    test_data_name = 'Fold'+str(j)+'_final_echo_corner_test'

    train_data = siftDistances(trainset)
    valid_data = siftDistances(validset)
    test_data = siftDistances(testset)

    #saves the distances
    np.save(train_data_name,train_data)
    np.save(valid_data_name,valid_data)
    np.save(test_data_name,test_data)
    #break
    j += 1
