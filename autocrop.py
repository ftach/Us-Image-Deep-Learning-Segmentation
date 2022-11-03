# -*- coding: utf-8 -*-
"""This program allows to use a localization CNN. It consists in preparing the input image, do the predictions, plot the image with the bounding box 
and crop the image following that bounding box.
"""


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
from PIL import Image
from image_processing import my_resample2, shape_to


def prepare_img(path, width=360, height=360):
    '''Prepare an image before using it in a CNN.'''
    img = Image.open(path).convert("L") # Ouvrir image et convertir en niveaux de gris
    img_array = my_resample2(np.array(img), width, height)
    img_array = shape_to(img_array, width, height)

    img = Image.fromarray(img_array).convert("RGB")


def make_predictions(img, img_array, cnn_path):
    '''Run the localization CNN. '''
    model = tf.keras.models.load_model(cnn_path, compile=False) # Chargement du modèle entraîné
    # model.summary()
    
    predictions = model.predict(np.reshape(img_array, (1, 360, 360, 1))) 

    xmin = round(predictions[(0, 0)]) 
    ymin = round(predictions[(0, 1)]) 
    xmax = round(predictions[(0, 2)]) 
    ymax = round(predictions[(0, 3)])

    x_center = xmin + (xmax-xmin)/2 - 80
    y_center = ymin + (ymax-ymin)/2 - 80
    
    
def plot_bounding_box(img, xmin, ymin, xmax, ymax):
    '''Draw the bounding box predicted by the CNN on the image. '''
    fig1, ax1 = plt.subplots()
    ax1.imshow(img)


    rect1 = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor='r', facecolor='none')
    ax1.add_patch(rect1)

    plt.show()


def plot_160_bounding_box(img, x_center, y_center):
    '''Draw a bounding box of size 160x160 on the image and centered around the bounding box predicted by the CNN. '''
    fig1, ax1 = plt.subplots()
    ax1.imshow(img)
    
    rect2 = patches.Rectangle((x_center, y_center), 160, 160, linewidth=1, edgecolor='b', facecolor='none')
    ax1.add_patch(rect2)

    plt.show()


def crop_image(img, x_center, y_center):
    '''Crop the image giving the center of the 160x160 bounding box. '''
    croped_img = img.crop((x_center, y_center, x_center+160, y_center+160))
    croped_img.show()