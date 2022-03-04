# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import RandomOverSampler           #import the necessary modules 
from sklearn.model_selection import train_test_split     

# %%
data = pd.read_csv('C:/Users/vasil/Desktop/tensorflow/FER/data/icml_face_data.csv')
pixel_data = data[' pixels']
label_data = data['emotion']            #open the csv file and categorize the data to pictures and emotions

# %%
def preprocess_pixels(pixel_data):
    images=[]
    for i in range(len(pixel_data)):
        img = np.fromstring(pixel_data[i], dtype='int', sep=' ')    #define the function to iterate and reshape the images to a 48x48x1 format
        img = img.reshape(48,48,1)
        images.append(img)
    X = np.array(images)
    return X

# %%
oversampler = RandomOverSampler(sampling_strategy='auto')
X_over, Y_over = oversampler.fit_resample(pixel_data.values.reshape(-1,1), label_data)  #oversampling magic to equalize the dataset
X_over_series = pd.Series(X_over.flatten()) #flatten the dataframe of the images

# %%
X = preprocess_pixels(X_over_series)
Y = Y_over                                              #use the fuction from earlier to the oversampled images
Y = Y_over.values.reshape(Y.shape[0],1)                 #reshape the values of the emotions                                 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 45) #split the dataset to training and testing batches 

# %%
plt.imshow(X[0,:,:,0])              #plot the first picture of the dataset


