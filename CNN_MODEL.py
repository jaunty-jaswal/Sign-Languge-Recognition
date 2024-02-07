import numpy as np
import os
import shutil
import cv2
import tensorflow as tf
# RESIZE each image to a image of 100*100
def readImage(imageName):
    img =cv2.imread(imageName)
    img = cv2.resize(img,(100,100))
    cv2.imwrite(imageName,img)
for j in range (65,91):
    for i in range (1,3001):
        readImage("dataset/"+chr(j)+"/" + chr(j) +str(i) + '.jpg')
# Split the dataset into training and test set.
# The dataset for each gesture is split into training and test set where test set consists of 25% of the dataset
rootdir= 'D:\Hand Gesture Recognition\dataset'
os.makedirs(rootdir+'/Training_Set')
os.makedirs(rootdir+'/Test_Set')
for i in range(65,91):
    os.makedirs(rootdir +'/Training_Set/' + chr(i))
    os.makedirs(rootdir +'/Test_Set/' + chr(i))
    source = rootdir + '/' + chr(i)
    allFileNames = os.listdir(source)
    np.random.shuffle(allFileNames)
    test_ratio = 0.25 #size of test set
    train_FileNames, test_FileNames = np.split(np.array(allFileNames),[int(len(allFileNames)* (1 - test_ratio))])
    train_FileNames = [source+'/'+ name for name in train_FileNames.tolist()]
    test_FileNames = [source+'/' + name for name in test_FileNames.tolist()]
    for name in train_FileNames:
      shutil.copy(name, rootdir +'/Training_Set/' + chr(i))
    for name in test_FileNames:
      shutil.copy(name, rootdir +'/Test_Set/' +chr(i))
# Preprocess each image to a greyscale image and load it for model training
#The below function reads the images within each folder of training set and test set.
#It also greyscales each image before storing it for better model training
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
def load_images_from_folder(folder,ch):
    images = []
    for filename in os.listdir(folder+ch):
        img = cv2.imread(os.path.join(folder+ch,filename))
        if img is not None:
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            class_num=labels.index(ch)
            #image_norm = cv2.normalize(gray_image, None, alpha=0,beta=1, norm_type=cv2.NORM_MINMAX)
            images.append([gray_image,class_num])
            #cv2.imwrite(folder+'GrayImages/'+ch+'/'+filename,gray_image)
    return images
trainImages = []
#os.makedirs('C:/Users/OMEN/Desktop/dataset/Training_Set/GrayImages')
#os.makedirs('C:/Users/OMEN/Desktop/dataset/Test_Set/GrayImages')
for i in range(65,91):
    #os.makedirs('C:/Users/OMEN/Desktop/dataset/Training_Set/GrayImages/'+chr(i))
    trainImages=trainImages+(load_images_from_folder('D:\Hand Gesture Recognition\dataset',chr(i)))
testImages=[]
for i in range(65,91):
    #os.makedirs('C:/Users/OMEN/Desktop/dataset/Test_Set/GrayImages/'+chr(i))
    testImages=testImages+(load_images_from_folder('D:\Hand Gesture Recognition\dataset',chr(i)))
x_train = []
y_train = []
x_test = []
y_test = []

for feature,label in trainImages:
  x_train.append(feature)
  y_train.append(label)

for feature, label in testImages:
  x_test.append(feature)
  y_test.append(label)
x_test=np.array(x_test)
x_train=np.array(x_train)
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
x_train=x_train/255.0
x_test=x_test/255.0
y_train = np.array(y_train)
y_train=y_train.reshape(-1,1)
y_test = np.array(y_test)
y_test=y_test.reshape(-1,1)
y_tt=y_test.flatten()
x_tt=[]
for i in range(0,19500):
    x_tt.append(x_test[i].flatten())
x_tt=np.array(x_tt)
y_tr=y_train.flatten()
x_tr=[]
for i in range(0,58500):
    x_tr.append(x_train[i].flatten())
x_tr=np.array(x_tr)

print(np.shape(y_test))
model_CNN = tf.keras.models.Sequential([
     tf.keras.layers.Conv1D(filters = 32, kernel_size = 5, strides = 1, activations ="relu" ,input_shape=x_train.shape[1:3]),
     tf.keras.layers.MaxPooling1D(pool_size =2),
     tf.keras.layers.Conv1D(filters = 64,kernel_size=5,strides = 1, activation= "relu"),
     tf.keras.layers.MaxPooling1D (pool_size =2),
     tf.keras.layers.Conv1D(filters = 128, kernel_size=5,strides = 1, activation= "relu"),
     tf.keras.layers.MaxPooling1D(pool_size =2),
     tf.keras.layers.Dropout(rate = 0.3),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(512,activation='relu'),
     tf.keras.layers.Dense(26, activation ='softmax')])
model_CNN.compile(loss = "sparse_categorical_crossentropy", optimzers ='adam', metrices=["accuracy"])
#50 epochs and 32 batch size

model_CNN.fit(x_tr,y_tr,batch_size=32,epochs=100)


model_CNN = tf.keras.models.Sequential([
     tf.keras.layers.Conv1D(filters = 32, kernel_size = 5, strides = 1, activations ="relu" ,input_shape=x_train.shape[1:3]),
     tf.keras.layers.MaxPooling1D(pool_size =2),
     tf.keras.layers.Conv1D(filters = 64,kernel_size=5,strides = 1, activation= "relu"),
     tf.keras.layers.MaxPooling1D(pool_size =2),
     tf.keras.layers.Conv1D(filters = 128, kernel_size=5,strides = 1, activation= "relu"),
     tf.keras.layers.MaxPooling1D(pool_size =2),
     tf.keras.layers.Dropout(rate = 0.3),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(512,activation='relu'),
     tf.keras.layers.Dense(26, activation ='softmax')])
model_CNN.compile(loss = "sparse_categorical_crossentropy", optimzers ='adam', metrices=["accuracy"])
#50 epochs and 32 batch size