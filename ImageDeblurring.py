#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import zipfile
import imghdr

def image_types(file_path):
    mime_types_count = {} 
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.is_dir():
                continue
            with zip_ref.open(file_info) as file:
                img_type = imghdr.what(None, file.read())
                if img_type:
                    mime_types_count[img_type] = mime_types_count.get(img_type, 0) + 1
    for mime_type, count in mime_types_count.items():
        print(f"{mime_type}: {count}")


file_path = r"C:\Users\ankku\Downloads\train_sharp.zip"  
image_types(file_path)


# In[ ]:


from PIL import Image

def resize_images(file_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.is_dir():
                dir_path = os.path.join(output_dir, file_info.filename)
                os.makedirs(dir_path, exist_ok=True)
                continue
            
            filename = os.path.basename(file_info.filename)
            if filename.endswith(".png"):
                with zip_ref.open(file_info) as file:
                    img = Image.open(file)
                    img_resized = img.resize((448, 256), Image.ANTIALIAS)
                
                output_path = os.path.join(output_dir, file_info.filename)
                img_resized.save(output_path)
                
resize_images(r"C:\Users\ankku\Downloads\train_sharp.zip" , "output_directory")


# In[ ]:


import cv2
import shutil

def apply_filters(input_path, output_path):
    img = cv2.imread(input_path)
    
    gaussian_filters = [
        {"kernel_size": (3, 3), "sigma": 0.3},
        {"kernel_size": (7, 7), "sigma": 1},
        {"kernel_size": (11, 11), "sigma": 1.6}
    ]
    
    for i, params in enumerate(gaussian_filters):
        filtered_img = cv2.GaussianBlur(img, params["kernel_size"], sigmaX=params["sigma"])
        output_filename = f"{os.path.splitext(os.path.basename(output_path))[0]}_filter{i+1}.jpg"
        output_filter_path = os.path.join(os.path.dirname(output_path), output_filename)
        cv2.imwrite(output_filter_path, filtered_img)
        print(f"Gaussian filter is applied {i+1} to {input_path}")
        
def apply_gaussian_filters(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for root, dirs, files in os.walk(input_dir):
        for d in dirs:
            input_subdir = os.path.join(root, d)
            output_subdir = input_subdir.replace(input_dir, output_dir)
            os.makedirs(output_subdir, exist_ok=True)
        
        for file in files:
            if file.endswith(".png"):
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = input_path.replace(input_dir, output_dir)
                apply_filters(input_path, output_path)


apply_gaussian_filters("output_directory", "output_directory_set_B")


# In[ ]:


import numpy as np
import tensorflow as tf
from tqdm import tqdm
import random

pathA = r"C:\Users\ankku\output_directory\train\train_sharp"   #sharp image directory
pathB = r"C:\Users\ankku\output_directory_set_B\train\train_sharp"  #blur image directory

setA = []
setB = []

for folder in tqdm(sorted(os.listdir(pathA))):
    folder_path = os.path.join(pathA, folder)
    for file in tqdm(sorted(os.listdir(folder_path))):
        file_path = os.path.join(folder_path, file)
        if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
            image = tf.keras.preprocessing.image.load_img(file_path, target_size=(128, 128))
            image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255
            setA.append(image)

setA = np.array(setA)

for folder in tqdm(sorted(os.listdir(pathB))):
    folder_path = os.path.join(pathB, folder)
    files = sorted(os.listdir(folder_path))
    for i in range(0, len(files)-2, 3):
        images = []
        for j in range(3):
            file = files[i + j]
            file_path = os.path.join(folder_path, file)
            if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
                image = tf.keras.preprocessing.image.load_img(file_path, target_size=(128, 128))
                image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255
                images.append(image)
        chosen_image = random.choice(images)
        setB_frames.append(chosen_image)

setB = np.array(setB)


# In[ ]:


len(setA)


# In[ ]:


len(setB)


# In[ ]:


from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from keras import backend as K

seed = 21
random.seed = seed
np.random.seed = seed


# In[ ]:


from sklearn.model_selection import train_test_split
y = setA;
x = setB;
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[ ]:


print(x_train[0].shape)
print(y_train[0].shape)
print(x_test[0].shape)
print(y_test[0].shape)


# In[ ]:


import matplotlib.pyplot as plt
r = random.randint(0, len(setA)-1)
print(r)
fig = plt.figure()
fig.subplots_adjust(hspace=0.1, wspace=0.2)
ax = fig.add_subplot(1, 2, 1)
ax.imshow(setA[r])
ax.axis('off')
ax = fig.add_subplot(1, 2, 2)
ax.imshow(setB[r])
ax.axis('off')


# In[ ]:


input_shape = (128, 128, 3)
batch_size = 32
kernel_size = 3
latent_dim = 256

layer_filters = [64, 128, 256]

inputs = Input(shape = input_shape, name = 'encoder_input')
x = inputs

for filters in layer_filters:
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=2,
               activation='relu',
               padding='same')(x)
shape = K.int_shape(x)
x = Flatten()(x)
latent = Dense(latent_dim, name='latent_vector')(x)

encoder = Model(inputs, latent, name='encoder')
print(encoder.summary())

latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
x = Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)
for filters in layer_filters[::-1]:
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        strides=2,
                        activation='relu',
                        padding='same')(x)

outputs = Conv2DTranspose(filters=3,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(x)

decoder = Model(latent_inputs, outputs, name='decoder')
print(decoder.summary())

autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
print(autoencoder.summary())


# In[ ]:


autoencoder.compile(loss='mse', optimizer='adam',metrics=["acc"])


# In[ ]:


lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               verbose=1,
                               min_lr=0.5e-6)
c = [lr_reducer]
history = autoencoder.fit(setB,
                      setA,
                      validation_data=(setB, setA),
                      epochs=2,
                      batch_size=128,
                      c=c)


# In[ ]:


import os
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
import tensorflow as tf

input_dir_b = r"C:\Users\ankku\Downloads\mp2_test\custom_test\blur"
output_dir_a_predicted = r"C:\Users\ankku\Downloads\mp2_test\custom_test\predicted"
output_dir_a_ground_truth = r"C:\Users\ankku\Downloads\mp2_test\custom_test\sharp"

def resize_image(img_path, target_size):
    img = Image.open(img_path)
    img_resized = img.resize(target_size, resample=Image.BILINEAR)
    return img_resized
for root, dirs, files in os.walk(output_dir_a_ground_truth):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            input_path = os.path.join(root, file)
            img_resized = resize_image(input_path, (128, 128))
            img_resized.save(input_path)
            
def preprocess_image(img_path, target_size):
    img = Image.open(img_path)
    img_resized = img.resize((target_size[1], target_size[0]), resample=Image.BILINEAR)
    img_array = np.array(img_resized) / 255.0  # Normalize pixel values
    return img_array

if not os.path.exists(output_dir_a_predicted):
    os.makedirs(output_dir_a_predicted)

for root, dirs, files in os.walk(input_dir_b):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            input_path = os.path.join(root, file)
            img_b = preprocess_image(input_path, (128, 128))
            img_a_predicted = autoencoder.predict(np.expand_dims(img_b, axis=0))[0]
            output_path_a_predicted = os.path.join(output_dir_a_predicted, file)
            Image.fromarray((img_a_predicted * 255).astype(np.uint8)).save(output_path_a_predicted)

def psnr_between_folders(folder1, folder2):
    psnr_values = []
    filenames = os.listdir(folder1)
    for filename in filenames:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path1 = os.path.join(folder1, filename)
            img_path2 = os.path.join(folder2, filename)
            img1 = np.array(Image.open(img_path1)) / 255.0
            img2 = np.array(Image.open(img_path2)) / 255.0
            psnr = peak_signal_noise_ratio(img1, img2)
            psnr_values.append(psnr)
    avg_psnr = np.mean(psnr_values)
    return avg_psnr

psnr_score = psnr_between_folders(output_dir_a_predicted, output_dir_a_ground_truth)
print(f"PSNR Score: {psnr_score} dB")

