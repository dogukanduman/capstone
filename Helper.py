
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import os
from urllib.request import urlretrieve
import tarfile
import zipfile
import sys
from cifar_data import get_data_set 
from sklearn.model_selection import train_test_split
from keras.preprocessing import image                  
from tqdm import tqdm


import matplotlib.pyplot as plt
from skimage.transform import resize
def prepareCIFAR():
    dataSets={}
    print('Preparing traing, validation and test set...')
    train_instances, train_label, categories = get_data_set(cifar=10)
    train_instances, validation_instances, train_label, validation_label = train_test_split(train_instances, train_label, test_size=0.2, random_state=42)   
    test_instances, test_label, categories = get_data_set(name="test", cifar=10)
    

    dataSets["train"]={'instances': train_instances,'labels': train_label}
    dataSets["test"]={'instances': test_instances,'labels': test_label}
    dataSets["validation"]={'instances': validation_instances,'labels': validation_label}
    dataSets["categories"]=categories
    return dataSets



def reshapeInstance(instance):
        reshaped_instance=np.expand_dims(instance.reshape(32, 32, 3) , axis=0)
        return reshaped_instance


def reshapeInstances(instances):
    print('Resphaping to (1, 32, 32, 3)')
    list_of_tensors = [reshapeInstance(instance) for instance in tqdm(instances)]
    return np.vstack(list_of_tensors)

def visualize(history):
    # Plotting
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
    
    
def load_reshaped_cifar(filename):
    try:
        with open(filename, 'rb') as handle:
            return (pickle.load(handle))  
    except (FileNotFoundError, EOFError, TypeError, ValueError):
            return np.empty(shape=[0, 0])   

        
        
def updateFile(newdataset,filename):   
    
    
    saved_dataset =load_reshaped_cifar(filename)
    print('saved_dataset :{} '.format(len(saved_dataset)))
    saved_dataset=np.append(saved_dataset,newdataset)
    pickle.dump(saved_dataset, open(filename, 'wb'))   
    print('updateFile checkpoint size:{} '.format(len(saved_dataset)))
    
    avc =load_reshaped_cifar(filename)
    for image in newdataset:
        plt.imshow(image)
        plt.show()
    
    
def resizeCIFAR(dataset,filename):
    
    
    chunk_size=10
    result =load_reshaped_cifar(filename);
    newdataset=[]
    numberOfFile=1
    if len(result) >0:
        print('Resized dataset will be read from file')
        newdataset = pickle.load(open(filename, 'rb'))  
        return newdataset
    else:  
        dataset_size=20 #len(dataset)
        newdataset= [[] for i in range(chunk_size)]
        print('Dataset will be resized and write file to {}'.format(filename))
        counter=1;
    
        for image in dataset:
            new_image=resize(image, (200, 200, 3)).astype('float32')
            newdataset[counter-1] = new_image
         
            print('Percantage of processes images {}%'.format((numberOfFile-1)/dataset_size))
       
            
            print('chunksize {}'.format(numberOfFile % chunk_size))
            if numberOfFile % chunk_size == 0:
                updateFile(newdataset,filename)
                print('Reshaped instances will be saved with {} new images, counter {}'.format(len(newdataset),(numberOfFile-1))) 
                newdataset= [[] for i in range(chunk_size)]
                counter =0

            numberOfFile=numberOfFile+1            
            counter=counter+1 
            
            if numberOfFile==20:
                break  
         

      
    newdataset = pickle.load(open(filename, 'rb'))  
    return newdataset
  
    
 

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




