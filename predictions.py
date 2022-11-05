import numpy as np
import cv2 as cv
import albumentations as albu
import torch

j = 1

while j <= 5:

    #echo augmentedname
    train_name = 'Fold'+str(j)+'_echo_grayslice_train.npy'
    valid_name = 'Fold'+str(j)+'_echo_grayslice_valid.npy'
    test_name = 'Fold'+str(j)+'_echo_grayslice_test.npy'

    #Loading the created datasets
    train = np.load(train_name)
    valid = np.load(valid_name)
    test = np.load(test_name)

    #loading the best model
    best_model = torch.load('./best_model.pth')

    #transpose
    train_set = train.transpose(0,3,1,2).astype('float32')
    valid_set = valid.transpose(0,3,1,2).astype('float32')
    test_set = test.transpose(0,3,1,2).astype('float32')

    #predictions function
    def maskedSet(dset):
        
        arr = []
        
        #Prediction function
        for i in range(dset.shape[0]):
            img_tensor = torch.from_numpy(dset[i]).to('cuda').unsqueeze(0)
            pr_mask = best_model.predict(img_tensor)
            pr_mask = pr_mask.squeeze().cpu().numpy().round()
            arr.append(pr_mask)
            
        return np.array(arr)

    #create final images dataset based on segmentation predictions
    def createSet(preds,dset):
        
        predictions = preds(dset)
        
        new_set = dset.transpose(0,2,3,1)

        new_arr = []

        for i in range(new_set.shape[0]):
            q = new_set[i].astype('uint8')
            w = predictions[i].astype('uint8')

            masked = cv.bitwise_or(q,q,mask=w) #overlapping the mask with the original image

            new_arr.append(masked)

        final = np.array(new_arr)#.transpose(0,3,1,2)
        
        return final,predictions

    final_train,train_preds = createSet(maskedSet,train_set)
    final_valid,valid_preds = createSet(maskedSet,valid_set)
    final_test,test_preds = createSet(maskedSet,test_set)

    new_train = 'Fold'+str(j)+'_class_echo_train'
    new_valid = 'Fold'+str(j)+'_class_echo_valid'
    new_test = 'Fold'+str(j)+'_class_echo_test'

    new_train_preds = 'Fold'+str(j)+'_echo_preds_train'
    new_valid_preds = 'Fold'+str(j)+'_echo_preds_valid'
    new_test_preds = 'Fold'+str(j)+'_echo_preds_test'

    #Stores the overlapped masks and augmented images
    np.save(new_train,final_train)
    np.save(new_valid,final_valid)
    np.save(new_test,final_test)

    #Stores the predicted segmentation masks
    np.save(new_train_preds,train_preds)
    np.save(new_valid_preds,valid_preds)
    np.save(new_test_preds,test_preds)
    j+= 1

print(final_train.shape)
print(final_valid.shape)
print(final_test.shape)