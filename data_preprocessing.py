import numpy as np
import cv2
import os

TRAIN_DIR = "C:/Users/RIDZZ ZOLDYCK/Desktop/pneumonia_data/chest_xray/train"
#VAL_DIR = "C:/Users/RIDZZ ZOLDYCK/Desktop/pneumonia_data/chest_xray/val"
TEST_DIR = "C:/Users/RIDZZ ZOLDYCK/Desktop/pneumonia_data/chest_xray/test"

CATEGORIES = ["NORMAL" , "PNEUMONIA"]

img_size = 150

train_data = []
test_data = []
#val_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(TRAIN_DIR,category)
        class_num = CATEGORIES.index(category)

        for img in os.listdir(path):
            img_arr = cv2.imread(os.path.join(path,img) , cv2.IMREAD_GRAYSCALE)
            img_arr = cv2.resize(img_arr , (img_size , img_size))
            train_data.append([img_arr , class_num])
            



def create_test_data():
    for category in CATEGORIES:
        path = os.path.join(TEST_DIR,category)
        class_num = CATEGORIES.index(category)

        for img in os.listdir(path):
            img_arr = cv2.imread(os.path.join(path,img) , cv2.IMREAD_GRAYSCALE)
            img_arr = cv2.resize(img_arr , (img_size , img_size))
            test_data.append([img_arr , class_num])
            
            

create_training_data()
create_test_data()


print(" \n\nNumber of images in Training Dataset : " , len(train_data))
print(" \n\nNumber of images in Test Dataset : " , len(test_data))



import random
random.shuffle(train_data)
random.shuffle(test_data)



print(" \n\n Some labels after shuffling : " )

print("\nTrain Data : ")
for sample in train_data[:5]:
    print(sample[1])

print("\nTest Data : ")
for sample in test_data[:5]:
    print(sample[1])





x_train = []
y_train = []

x_test = []
y_test = []




for features , label in train_data :
    x_train.append(features)
    y_train.append(label)


for features , label in test_data :
    x_test.append(features)
    y_test.append(label)




x_train = np.array(x_train).reshape(-1,img_size,img_size,1)
x_test = np.array(x_test).reshape(-1,img_size,img_size,1)







import pickle

pickle_out = open("X_TRAIN.pickle" , "wb")
pickle.dump(x_train , pickle_out)
pickle_out.close()

pickle_out = open("X_TEST.pickle" , "wb")
pickle.dump(x_test , pickle_out)
pickle_out.close()





pickle_out = open("Y_TRAIN.pickle" , "wb")
pickle.dump(y_train , pickle_out)
pickle_out.close()

pickle_out = open("Y_TEST.pickle" , "wb")
pickle.dump(y_test , pickle_out)
pickle_out.close()





print(" \n\n Features and labels of all datasets have been saved ")









            
