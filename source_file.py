
# coding: utf-8

# In[ ]:


"""
Mount google drive to facilitate importing of data 

"""

# from google.colab import drive
# drive.mount('/content/gdrive')


# In[2]:


# importing dependencies
import os
import glob
import pandas as pd
import numpy as np
import pickle
from PIL import Image, ImageEnhance
import cv2

# Keras functionalities
from keras.utils import to_categorical
import keras
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D,AveragePooling2D 
from keras import regularizers
from keras.layers.normalization import BatchNormalization


# In[ ]:


#To unzip the training images folder
# !unzip 'gdrive/My Drive/grid2019/trainImage.zip'


# In[3]:


# Loading train.csv and test.csv
train = pd.read_csv("training.csv")
test = pd.read_csv("test.csv")


# In[4]:


path_train= "images_train"
path_test= "images_test"
size=128


# # ** Generating Augmented Data**

# In[5]:


def numfy(train,path,size):   
  """
  Parameters: train = train.csv
              path = the path to the folder containing training images
  Returns: list of images in the form of numpy array along with their bounding box co-ordinates
  """
  train_names=train.iloc[:,0] 
  train_coordinates = train.drop(['image_name'],axis= 1) 
  train_images=[] 

  for i,name in enumerate(train_names):
      for image in glob.glob(os.path.join(path, name)):
          img = cv2.imread(image)
          img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
          train_images.append(img) 

  print("done")
  
  return (train_images),(train_coordinates)


# In[6]:


def flip(image_list,train_df):
    """
    Parameters: image_list = list containing numpy array version of the images,
                train_df = train co-ordinates
    Returns: flipped image list, flipped bounding-box co-ordinates

    """    
    i = train_df.shape[0]
    for a in range(i):
        h,w = image_list[a].shape[0],image_list[a].shape[1]
        elem = np.random.choice([ 90, 180, 270, 1423, 1234])
        x1,x2,y1,y2 = train_df.values[a]
        if elem % 10 == 0:
            x = x1 - w / 2
            y = y1 - h / 2

            x1 = w / 2 + x * np.cos(np.deg2rad(elem)) - y * np.sin(np.deg2rad(elem))
            y1 = h / 2 + x * np.sin(np.deg2rad(elem)) + y * np.cos(np.deg2rad(elem))

            x = x2 - w / 2
            y = y2 - h / 2

            x2 = w / 2 + x * np.cos(np.deg2rad(elem)) - y * np.sin(np.deg2rad(elem))
            y2 = h / 2 + x * np.sin(np.deg2rad(elem)) + y * np.cos(np.deg2rad(elem))

            image_list[a] = Image.fromarray(np.uint8(image_list[a]))
            image_list[a] = image_list[a].rotate(-elem)
            image_list[a] = np.asarray(image_list[a])
        else:
            if elem == 1423:
                image_list[a] = Image.fromarray(np.uint8(image_list[a]))
                image_list[a] = image_list[a].transpose(Image.FLIP_TOP_BOTTOM)
                image_list[a] = np.asarray(image_list[a])
                y1 = h - y1
                y2 = h - y2

            elif elem == 1234:
                image_list[a] = Image.fromarray(np.uint8(image_list[a]))
                image_list[a] = image_list[a].transpose(Image.FLIP_LEFT_RIGHT)
                image_list[a] = np.asarray(image_list[a])
                x1 = w - x1
                x2 = w - x2

        tmp = x1
        x1 = min(x1, x2)
        x2 = max(tmp, x2)

        tmp = y1
        y1 = min(y1, y2)
        y2 = max(tmp, y2)

        x1 = max(x1, 0)
        y1 = max(y1, 0)

        y1 = min(y1, h)
        x1 = min(x1, w)
        y2 = min(y2, h)
        x2 = min(x2, w)
        train_df.values[a] = x1,x2,y1,y2       
    return image_list,train_df
  


# In[9]:



def resize(image_list,train_df,size):
    """
      Parameters: train = train.csv 
                  path = the path to the folder containing training images
                  size = the size to which images are to be resized for the model
      Returns: resized image list, resized bounding-box co-ordinates
    """
    i = train_df.shape[0]
    for a in range(i):
        h,w = image_list[a].shape[0],image_list[a].shape[1]
        image_list[a] = cv2.resize(image_list[a], (size, size)) 
        x1,x2,y1,y2 = train_df.values[a]
        x1 = int(x1*size/w)
        y1 = int(y1*size/h)
        x2 = int(x2*size/w)
        y2 = int(y2*size/h)
        train_df.values[a] = x1,x2,y1,y2

    return image_list,train_df


# In[10]:


def normalization(image_list,train_df):
    """
    Parameters: image_list = list containing numpy array version of the images,
                train_df = train co-ordinates
    Returns: normalized image list, normalized bounding-box co-ordinates
    
    """
    i = train_df.shape[0]
    for a in range(i):
        image_list[a] = cv2.normalize(image_list[a], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
    return image_list, train_df


# In[11]:


def crop(image_list,train_df):
    """
    Parameters: image_list = list containing numpy array version of the images,
                train_df = train co-ordinates
    Returns: cropped image list, cropped bounding-box co-ordinates
    """
    
    i = train_df.shape[0]
    for a in range(i):
        h,w = image_list[a].shape[0],image_list[a].shape[1]
        start_x = np.random.randint(0, high=np.floor(0.15 * w))
        stop_x = w - np.random.randint(0, high=np.floor(0.15 * w))
        start_y = np.random.randint(0, high=np.floor(0.15 * h))
        stop_y = h - np.random.randint(0, high=np.floor(0.15 * h))
        
        image_list[a] = Image.fromarray(np.uint8(image_list[a]))
        image_list[a] = image_list[a].crop((start_x, start_y, stop_x, stop_y))
        image_list[a] = np.asarray(image_list[a])
        x1,x2,y1,y2 = train_df.values[a]
        x1 = max(x1 - start_x, 0)
        y1 = max(y1 - start_y, 0)
        x2 = min(x2 - start_x, w)
        y2 = min(y2 - start_y, h)

        if np.abs(x2 - x1) < 5 or np.abs(y2 - y1) < 5: 
            print("\nWarning: cropped too much (obj width {}, obj height {}, img width {}, img height {})\n".format(x2 - x1, y2 - y1, w, h))
            
        tmp = x1
        x1 = min(x1, x2)
        x2 = max(tmp, x2)

        tmp = y1
        y1 = min(y1, y2)
        y2 = max(tmp, y2)

        x1 = max(x1, 0)
        y1 = max(y1, 0)

        y1 = min(y1, h)
        x1 = min(x1, w)
        y2 = min(y2, h)
        x2 = min(x2, w)

        train_df.values[a] = x1,x2,y1,y2
        
    return image_list,train_df


# In[12]:


def augmentation_crop(train,path,size):
  """
  Parameters: train = train.csv 
              path = the path to the folder containing training images
              size = the size to which images are to be resized for the model

  Returns: Processed crop images, processed cropped bounding box coordinates
  """
  X_numfy,y_numfy=numfy(train,path,size)
#   print(X_numfy[356])
#   print(y_numfy.iloc[34])
  X_crop,y_crop=crop(X_numfy,y_numfy)
#   print(X_crop[356])
#   print(y_crop.iloc[34])
  X_crop_resize,y_crop_resize=resize(X_crop,y_crop,size)
#   print(X_crop_resize[356])
#   print(y_crop_resize.iloc[34])
  X_aug_crop,y_aug_crop=normalization(X_crop_resize,y_crop_resize)
#   print(X_aug_crop[356])
#   print(y_aug_crop.iloc[34])
  return X_aug_crop,y_aug_crop

size=128 
X_crop,y_crop=augmentation_crop(train,path_train,size)


# In[13]:


def augmentation_flip(train,path,size):
  """
  Parameters: train = train.csv 
              path = the path to the folder containing training images
              size = the size to which images are to be resized for the model

  Returns: Processed flipped images, processed flipped bounding box coordinates
  """
  X_numfy,y_numfy=numfy(train,path,size)
#   print(X_numfy[356])
#   print(y_numfy.iloc[34])
  X_flip,y_flip=flip(X_numfy,y_numfy)
#   print(X_flip[356])
#   print(y_flip.iloc[34])
  X_flip_resize,y_flip_resize=resize(X_flip,y_flip,size)
#   print(X_flip_resize[356])
#   print(y_flip_resize.iloc[34])
  X_aug_norm,y_aug_norm=normalization(X_flip_resize,y_flip_resize)
#   print(X_aug_norm[356])
#   print(y_aug_norm.iloc[34])
  return X_aug_norm,y_aug_norm

size=128
X_aug,y_aug=augmentation_flip(train,path_train,size)


# In[14]:


# Reshaping the augmented (flipped & cropped) images 
X_aug=np.reshape(X_aug,(14000,128,128,1))
X_crop=np.reshape(X_crop,(14000,128,128,1))


# In[15]:


def preprocessing(train,path,size):
  """
  Get X and y for your model after preprocessing the original training images
  X = list of 3D Numpy arrays of training images
  y = list of bounding box co-ordinates
  
  -----------------------
  Parameters: 
  1. train (training.csv) dataframe
  2. path to the actual training images
  3. size = dimensions to which you want to resize your images of the 
  ------------------------
  Return:
  X and y as discussed above
  
  """
   
  train_names=train.iloc[:,0] 
  train_coordinates = train.drop(['image_name'],axis= 1) 
  train_images=[] 

  for i,name in enumerate(train_names):
      for image in glob.glob(os.path.join(path, name)):
          img = cv2.imread(image)
          img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
          w,h = img.shape
          x1,x2,y1,y2 = train_coordinates.values[i]
          x1 = int(x1*size/h)
          y1 = int(y1*size/w)
          x2 = int(x2*size/h)
          y2 = int(y2*size/w)
          train_coordinates.values[i] = x1,x2,y1,y2
          img = cv2.resize(img, (size,size))
          img= cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
          train_images.append(img) 

  
  return train_images, train_coordinates


# In[16]:


X,y=preprocessing(train,path_train,size)
X=np.reshape(X,(14000,128,128,1))


# In[17]:


# concatenating original training images, cropped training images and
# flipped training images into a new list
Xc=np.concatenate((X, X_aug), axis=0)
X_new=np.concatenate((Xc,X_crop),axis=0)


# In[18]:


# concatenating arrays of bounding box coordinates of original training images, cropped training images and
# flipped training images into a new array
y_aug_arr=np.array(y_aug)
y_arr=np.array(y)
yc_arr=np.array(y_crop)
yc=np.concatenate((y_arr, y_aug), axis=0)
y_new=np.concatenate((yc, yc_arr), axis=0)


# In[19]:


def CNN_Regression():
    """
    Description: Architecture of the CNN-R model 
    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(size, size, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
  
    
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
  
    model.add(Flatten())
    model.add(Dense(2048,  activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048,  activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4))
    
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False))
#     print(model.summary())
    return model


# In[22]:



model=CNN_Regression()
model.fit(X_new, y_new, batch_size=32, epochs=110, verbose=1)


# In[ ]:


"""

Saving Models to drive

"""
# model_json = model.to_json()
# with open("gdrive/My Drive/model128/bestaug128_28k2.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("gdrive/My Drive/model128/bestaugweights128_28k2.h5")
# print("Saved model to drive")


# In[ ]:


"""
     PREDICTIONS AND OUTPUT CSV GENERATION
"""

"""
unzipping test images' folder
"""
# !unzip 'gdrive/My Drive/grid2019/testImage.zip'


# In[23]:


def preprocessing_test(test,path,size):
  """
  Get X after processing the test images
  X = list of 3D Numpy arrays of training images
  
  """
  
  test_names=test.iloc[:,0] #To get the names of every train image, indexed as per training.csv

  
  test_images=[] # will contain the 3D numpy version of all the training images 14000 x 224 x 224 x 3

  for i,name in enumerate(test_names):
#       print(i)
      for image in glob.glob(os.path.join(path, name)):
          img = cv2.imread(image)
          img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
          img = cv2.resize(img, (size,size))
          img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
          test_images.append(img) 

  
  return test_images

X_test_final=preprocessing_test(test,path_test,size)


# In[24]:


"""
In the next few lines of code, X_test_final is being reshaped so that it can be fed for
predictions using the model we just trained. The predicted bounding boxes are resized to
480 X 640 scale and then the results are saved in a csv file as asked in the submission guidelines
"""
X_test_final=np.reshape(X_test_final,(12815,128,128,1))
y_pred_final = model.predict(np.array(X_test_final))
y_pred_final[0]
w=480
h=640
for i in range(y_pred_final.shape[0]):
  x1_pred,x2_pred,y1_pred,y2_pred = y_pred_final[i]
  x1_pred = (x1_pred*h/size+2)
  y1_pred = (y1_pred*w/size+2)
  x2_pred = (x2_pred*h/size+2)
  y2_pred = (y2_pred*w/size+2)
  y_pred_final[i] = x1_pred,x2_pred,y1_pred,y2_pred

y_pred_df = pd.DataFrame({'x1':y_pred_final[:,0],'x2':y_pred_final[:,1],'y1':y_pred_final[:,2],'y2':y_pred_final[:,3]})
test_names=test.iloc[:,0]
y_pred_df.insert(loc=0, column='image_name', value=test_names)
y_pred_df.head()
y_pred_df.to_csv('Submission.csv', index=False)
# !cp bestCF128.csv gdrive/My\ Drive/
# prediction_file=pd.read_csv("/content/gdrive/My Drive/bestCF128.csv")
# prediction_file.head()

