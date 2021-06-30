# -*- coding: utf-8 -*-
"""This program enables to perform semantic segmentation on ultrasound images of breast cancer. 
"""

from PIL import Image 
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cProfile, pstats
from image_processing import regular_sample, shape_to, save_img_from_array


def prepare_prediction(us_path, mask_path, model_path, seg_saving_path, from_image, prepared=True):    
    ''' Allow to prepare ultrasound images before running a semantic segmentation. 
    Parameters
    ----------
    us_path: str
        path leading to the ultrasound image
    mask_path: str
        path leading to the mask of the ultrasound image
    model_path: str
        path leading to the CNN used to make the predictions
    seg_saving_path: str
        path choosen to save the segmentated image
    from_image: bool
        True if the ultrasound data is provided as a image saved as a .jpg, .png or .bmp file, False if it is provided as an array saved in a .pkl file 
    prepared: bool
        True if the data already corresponds to the CNN input format, False if not.
        '''
    if from_image == True:
        original = np.array(Image.open(us_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L"))
        
    elif from_image == False: 
        with open(us_path, "rb") as pkl_file:
            original = pickle.load(pkl_file)
        with open("mask_path", "rb") as pkl_file2:
            mask = pickle.load(pkl_file2)        
            
    elif prepared == False:
        original = regular_sample(original)
        original = shape_to(original) 
    
    make_prediction(original, mask, model_path, seg_saving_path)
    

def make_prediction(img_array, mask_array, model_path, seg_saving_path):
    '''Run semantic segmentation of ultrasound images. 
        Parameters
    ----------
    us_path: str
        path leading to the ultrasound image
    mask_path: str
        path leading to the mask of the ultrasound image
    model_path: str
        path leading to the CNN used to make the predictions
    seg_saving_path: str
        path choosen to save the segmentated image
    '''

    loaded_model = tf.keras.models.load_model(model_path, compile=False)
    
    # calculer erreurs de prédiction
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(5e-4, decay_steps=1e5, decay_rate=2e-05, staircase=True)
    loaded_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.MeanIoU(num_classes=2)]) 
    loaded_model.evaluate(np.reshape(img_array, (1, 160, 160)), np.reshape(mask_array/255, (1, 160, 160)), batch_size=2)
    
    # faire la segmentation
    predictions = loaded_model.predict(np.reshape(img_array, (1, 160, 160))) 
    predictions = np.reshape(predictions, (160, 160)) # Remettre au bon format le tableau des prédictions
    # Remplacer la prédiction par la valeur de pixel qu'elle approxime 
    classified_predictions = np.ones((160, 160))
    for i in range(160):
        for j in range(160):
            if predictions[(i, j)] < 0.5:
                classified_predictions[(i, j)] = 0 # noir
                
            else: 
                classified_predictions[(i, j)] = 255 # blanc
    save_img_from_array(classified_predictions, seg_saving_path) # sauvegarder image segmentée

    
def main():
    us_path = ""
    mask_path = ""
    model_path = ""
    seg_saving_path = ""
    from_image = True
    prepared = False
    prepare_prediction(us_path, mask_path, model_path, seg_saving_path, from_image, prepared)
    
main()   