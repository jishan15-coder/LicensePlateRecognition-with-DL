# importing libraries
import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import easyocr
! pip install ultralytics
! pip install ultralytics supervision==0.2.0
! pip install -U ipywidgets
from ultralytics import YOLO
import supervision as sv

from IPython.display import clear_output
clear_output()
# setting dataset path
path = '/kaggle/input/licenseplatedataset/License Plate/'
train_dir = '/kaggle/input/licenseplatedataset/License Plate/train'
os.listdir(path)
# Data load & Augmentation
from skimage.filters import thresholding

def add_noise(img, rate=0.1):
    noise = np.random.normal(loc=0, scale=rate * 255, size=img.shape)
    noisy_img = img + noise
    noisy_img = np.clip(noisy_img, 0, 255)
    noisy_img = noisy_img.astype(np.float32)
    return noisy_img
    

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.blur(img, (2, 2))
    img = add_noise(img)
    img = img.astype('float32')
    img = np.expand_dims(img, axis= -1)
    return img


from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function= preprocess,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest',
    brightness_range= (0.75, 1.25)
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size= (640, 640),
    batch_size= 32,
    class_mode= 'categorical',
    shuffle= True
)

plt.figure(figsize= (9, 4))
for image_batch, label_batch in train_generator:
    for i in range(8):
        img = image_batch[i]
        plt.subplot(2, 4, i + 1)
        plt.imshow(img)
        plt.axis('off')
    break
# showing images of vehicles
count = 16

plt.figure(figsize= (9,7))

val_path = os.path.join(path, 'valid', 'images')
for i in range(count):
    total_car = len(os.listdir(val_path))
    j = np.random.randint(14, total_car + 1)
    try:
        img = plt.imread(val_path + f'/{j}.jpg')
        plt.subplot(4, 4, i + 1)
        plt.imshow(img)
        plt.axis('off')
        
    except:
        i -= 1
    
    
# Loading & training model
coco_model = YOLO('yolov8n.pt')
history = coco_model.train(data= path + 'data.yaml', 
                 epochs= 60,
                 show_labels= False,
                 show_conf= False,
                 device= [0],)

test_path = '/kaggle/input/licenseplatedataset/License Plate/valid/images'
X = []

for f in glob.glob(test_path + '/*.jpg'):
    img = cv2.imread(f)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    X.append(img)

X = np.array(X)
X.shape
detections = []
results = []
for x in X:
    temp = coco_model(x)[0]
    results.append(temp)
    detect = sv.Detections.from_yolov8(temp)
    detections.append(detect)

# indx = 0
# raw_image = X[indx]
# image_detected = results[indx].plot()

from skimage import exposure
from skimage.filters import thresholding

def image_processing(image):    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = np.array([
        [0, -1, 0,],
        [-1, 5, -1,],
        [0, -1, 0,]
    ])
    
#     image = cv2.filter2D(image, -1, kernel)
    image = cv2.blur(image, (2, 2), 0)
    image = cv2.equalizeHist(image)
#     image = cv2.GaussianBlur(image, (3, 3), 0)
#     image = cv2.blur(image, (3, 3), 0)
    
    return image


plt.figure(figsize= (7, 4))
plt.subplot(1, 2, 1)
plt.imshow(plate)
plt.axis('off')
plt.tight_layout()


gray_plate = image_processing(plate)
plt.subplot(1, 2, 2)
plt.imshow(gray_plate, cmap= 'gray')
plt.axis('off')
plt.tight_layout()

plt.savefig('sharpening.png')
detect = detections[indx]

x1, y1 = detect.xyxy[0][:2]
x2, y2 = detect.xyxy[0][2:]

x1, y1 = (round(x1), round(y1))
x2, y2 = (round(x2), round(y2))

plate = X[indx][y1: y2, x1: x2, :]
plt.imshow(plate)
plt.imsave('english license plate cropped.png', plate)
reader = easyocr.Reader(['bn', 'en'], gpu= False)
text = reader.readtext(gray_plate)
city_name = text[0][1]
reg_no = text[1][1]

final_res = city_name + '\n' + reg_no

plt.imshow(X[0])
plt.axis('off')


plates = detections[0]

for plate in plates:
    x1, y1 = plate[0][:2]
    x2, y2 = plate[0][2:]        
    x1, y1 = (round(x1), round(y1))
    x2, y2 = (round(x2), round(y2))

    detected_image = cv2.rectangle(X[0], 
                           (x1, y1), 
                           (x2, y2), 
                           color=[0, 0, 255], 
                           thickness=5)

detec
plt.imshow(detected_image)
detections_new = []
for k in detections.keys():
    print(len(detections[k]))
count = 16

plt.figure(figsize= (16, 9))

for i in range(count):
    x = X[i].copy()
    plates = detections[i]
    
    for plate in plates:
        x1, y1 = plate[0][:2]
        x2, y2 = plate[0][2:]        
        x1, y1 = (round(x1), round(y1))
        x2, y2 = (round(x2), round(y2))
        
        detected_image = cv2.rectangle(x, 
                               (x1, y1), 
                               (x2, y2), 
                               color=[0, 0, 255], 
                               thickness=5)
        
    plt.subplot(4, 4, i + 1)
    plt.imshow(detected_image)
    plt.axis('off')
license_plates = []

for i in range(X.shape[0]):
    x = X[i].copy()
    plates = detections[i]
    
    for plate in plates:
        x1, y1 = plate[0][:2]
        x2, y2 = plate[0][2:]        
        x1, y1 = (round(x1), round(y1))
        x2, y2 = (round(x2), round(y2))
        
        detected_image = cv2.rectangle(x, 
                               (x1, y1), 
                               (x2, y2), 
                               color=[0, 0, 255], 
                               thickness=5)
        
        plate = x[y1: y2, x1: x2, :]
        license_plates.append(plate)
        
count = 16

plt.figure(figsize= (16, 9))

for i in range(count):
    plt.subplot(4, 4, i + 1)
    plate = license_plates[i]
    plt.imshow(plate)
from skimage import exposure
from skimage.filters import thresholding

def image_processing(image):    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     image = cv2.equalizeHist(image)
#     image = cv2.GaussianBlur(image, (3, 3), 0)
    return image

preprocessed_plate = []

for plate in license_plates:
    preprocessed_plate.append(image_processing(plate))
    
    
    
count = 16
plt.figure(figsize= (16, 9))

for i in range(count):
    plt.subplot(4, 4, i + 1)
    plate = preprocessed_plate[i]
    plt.imshow(plate, cmap= 'gray')
reader = easyocr.Reader(['bn'], gpu= 'cuda:0')
for plate in preprocessed_plate:
    text = reader.readtext(plate)
    print(text)