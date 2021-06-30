# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 16:50:10 2021

@author: ftachenn
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 19 16:49:01 2021

@author: ftachenn
"""
from PIL import Image 
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from os import listdir, makedirs
from os.path import isfile, join, isdir
import cProfile, pstats
import sys
from skimage import transform, color
from image_processing import *
import json



PI = np.pi 

def open_pkl_matrix(directory):
    dataset = []
    all_filenames = [f for f in listdir(directory) if isfile(join(directory, f))]
    #filename = "OASBUDdata/training_dataset/subj4_rf1.pkl"
    for filename in all_filenames:
        filename = directory + "/" + filename
        with open(filename, 'rb') as pkl_matrix:
            data = pickle.load(pkl_matrix)
            data = np.reshape(data, (160, 160, 1))
            dataset.append(data)
    dataset = np.array(dataset)
        
    return dataset
    #return tuple(dataset)
    
def create_dataset_from_images(directory):
    dataset = []
    all_filenames = [f for f in listdir(directory) if isfile(join(directory, f))]
    for filename in all_filenames:
        path = directory + "/" + filename
        img_array = np.array(Image.open(path).convert("L"))
        data = np.reshape(img_array, (160, 160, 1))
        dataset.append(data)
    dataset = np.array(dataset)
        
    return dataset

def flipping(x, y):

    x_flipped = np.fliplr(x)
    y_flipped = np.fliplr(y)
    
    return (x_flipped, y_flipped)

    
def random_rotation(x, y):

    angle = np.random.randint(8) # rotation de +- 8° maximum
    sens = np.random.randint(2)
    if sens == 0: # on introduit une rotation dans le sens anti-horaire aléatoirement
        angle *= -1
   
    x_rotated = transform.rotate(x, angle)
    y_rotated = transform.rotate(y, angle)
    
    x_rotated = transform.resize(x_rotated, (160, 160))
    y_rotated = transform.resize(y_rotated, (160, 160))
    
    x_rotated = color.rgb2gray(x_rotated)
    y_rotated = color.rgb2gray(y_rotated)
    
    #x_rotated = x_rotated*255 à utiliser que si on part de photos
    # on multiplie pas y par 255 car on veut que ca reste entre 0 et 1

    return (np.reshape(x_rotated, (160, 160, 1)), np.reshape(y_rotated, (160, 160, 1)))
    
    
def create_data_augmentation(x_training_dataset, y_training_dataset):
    new_x_dataset = []
    new_y_dataset = []
    for x, y in zip(x_training_dataset, y_training_dataset):
        
        new_x_dataset.append(x)
        new_y_dataset.append(y)
        
        x = np.reshape(x, (160, 160))
        y = np.reshape(y, (160, 160))   
        
        rotated = random_rotation(x, y)
        new_x_dataset.append(rotated[0])
        new_y_dataset.append(rotated[1])
        
        flipped = flipping(x, y)
        new_x_dataset.append(np.reshape(flipped[0], (160, 160, 1)))
        new_y_dataset.append(np.reshape(flipped[1], (160, 160, 1)))


    return (np.array(new_x_dataset), np.array(new_y_dataset))

def show_augmentated_img():
    with open("OASBUDdata/x_training_dataset/subj66_rf1.pkl", "rb") as pkl_file:
        x = pickle.load(pkl_file)
    with open("OASBUDdata/y_training_dataset/subj66.pkl", "rb") as pkl_file:
        y = pickle.load(pkl_file)
    
    (x, y) = random_rotation(x, y)
    x = Image.fromarray(np.reshape(x, (160, 160)))
    x = x.convert("L")
    x.save("OASBUDdata/x_training_dataset/subj66.jpg")
    y = Image.fromarray(np.reshape(y, (160, 160)))
    y = y.convert("L")
    y.save("OASBUDdata/y_training_dataset/subj66.jpg")


def init_model():
    #loaded_model = tf.keras.models.load_model("Trained_models/UNET_b_160_IOU.h5", compile=False) # Chargement du modèle entraîné
    loaded_model = tf.keras.models.load_model("UNET_b_160_IOU.h5", compile=False)

    encoding_layers_a = ['input_1', 'batch_normalization_1', 'conv2d_1', 'batch_normalization_2', 'conv2d_2', 'batch_normalization_3', 
                         'max_pooling2d_1', 'batch_normalization_4', 'conv2d_3', 'batch_normalization_5', 'conv2d_4', 'batch_normalization_6', 
                         'max_pooling2d_2', 'batch_normalization_7', 'conv2d_5', 'batch_normalization_8', 'conv2d_6', 'batch_normalization_9', 
                         'max_pooling2d_3', 'batch_normalization_10', 'conv2d_7', 'batch_normalization_11', 'conv2d_8', 'batch_normalization_12', 
                         'dropout_1', 'batch_normalization_13', 'max_pooling2d_4', 'batch_normalization_14', 'conv2d_9', 'batch_normalization_15', 
                         'conv2d_10', 'batch_normalization_16', 'dropout_2', 'batch_normalization_17', 'up_sampling2d_1', 'conv2d_11']
    decoding_layers_a = ['concatenate_1', 'batch_normalization_18', 'conv2d_12', 'batch_normalization_19', 'conv2d_13', 'batch_normalization_20', 
                         'up_sampling2d_2', 'conv2d_14', 'concatenate_2', 'batch_normalization_21', 'conv2d_15', 'batch_normalization_22', 
                         'conv2d_16', 'batch_normalization_23', 'up_sampling2d_3', 'conv2d_17', 'concatenate_3', 'batch_normalization_24', 
                         'conv2d_18', 'batch_normalization_25', 'conv2d_19', 'batch_normalization_26', 'up_sampling2d_4', 'conv2d_20', 
                         'concatenate_4', 'batch_normalization_27', 'conv2d_21', 'batch_normalization_28', 'conv2d_22', 'batch_normalization_29', 
                         'conv2d_23', 'batch_normalization_30', 'conv2d_24']
    
    encoding_layers_b = ['input_2', 'conv2d_25', 'batch_normalization_21', 'activation_21', 'max_pooling2d_5', 'dropout_9', 'conv2d_27', 
                         'batch_normalization_23', 'activation_23', 'max_pooling2d_6', 'dropout_10', 'conv2d_29', 'batch_normalization_25', 
                         'activation_25', 'max_pooling2d_7', 'dropout_11', 'conv2d_31', 'batch_normalization_27', 'activation_27', 
                         'max_pooling2d_8', 'dropout_12', 'conv2d_33', 'batch_normalization_29', 'activation_29', 'up_sampling2d_5', 'conv2d_34']
    decoding_layers_b = ['concatenate_5', 'dropout_13', 'conv2d_36', 'batch_normalization_31', 'activation_31', 'up_sampling2d_6', 
                         'conv2d_37', 'concatenate_6', 'dropout_14', 'conv2d_39', 'batch_normalization_33', 'activation_33', 'up_sampling2d_7', 
                         'conv2d_40', 'concatenate_7', 'dropout_15', 'conv2d_42', 'batch_normalization_35', 'activation_35', 'up_sampling2d_8', 
                         'conv2d_43', 'concatenate_8', 'dropout_16', 'conv2d_45', 'batch_normalization_37', 'activation_37', 'conv2d_46', 
                         'batch_normalization_38', 'activation_38']
    
    
    for layer_name in encoding_layers_b:
        loaded_model.get_layer(layer_name).trainable = False
    print("Encoder frozen.")
    
    for layer_name in decoding_layers_b:
        loaded_model.get_layer(layer_name).trainable = True
    print("Decoder unfrozen.")
    return loaded_model
    
def training_session(segmentation_results_path, saved_weights_path, EPOCHS=1, INIT_LRATE=0.001, DECAY_STEPS=1e5, BATCH_SIZE=64):
    # Hyperparameters
    print(EPOCHS, INIT_LRATE, DECAY_STEPS, BATCH_SIZE)
    DECAY_RATE = INIT_LRATE / EPOCHS 
    # GET MODEL
    loaded_model = init_model()
    print("Model loaded.")
    
    # GET TRAINING, VALIDATION AND TESTING DATA from array
    #x_training_dataset = open_pkl_matrix("whole_dataset/training_dataset/x/pkl") # à changer sur mon ordi: rajouter OASBUDdata au début
    #y_training_dataset = open_pkl_matrix("whole_dataset/training_dataset/y/pkl")/255 #• on a besoin de 0 ou de 1
    #x_validation = open_pkl_matrix("whole_dataset/validation_dataset/x/pkl")
    #y_validation = open_pkl_matrix("whole_dataset/validation_dataset/y/pkl")/255 #• on a besoin de 0 ou de 1
    #testing_dataset = open_pkl_matrix("whole_dataset/testing_dataset/x/pkl")    
    
    # GET TRAINING, VALIDATION AND TESTING DATA from images create_dataset_from_images
    x_training_dataset = create_dataset_from_images("oral_dataset/training/us")
    print("Training US OK")
    y_training_dataset = create_dataset_from_images("oral_dataset/training/mask")/255
    print("Training mask OK")
    x_validation = create_dataset_from_images("oral_dataset/validation/us")
    print("Validation US OK")
    y_validation = create_dataset_from_images("oral_dataset/validation/mask")/255
    print("Validation mask OK")
    testing_dataset = create_dataset_from_images("oral_dataset/testing/us")
    print("Testing US OK")
    
    augmentated_datasets = create_data_augmentation(x_training_dataset, y_training_dataset)  # DATA AUGMENTATION
    print("Data augmentation OK")

    testing_filenames = [f for f in listdir("oral_dataset/testing/us") if isfile(join("oral_dataset/testing/us", f))]    
    print("Datasets loaded.")
    
    # TRAINING THE MODEL
    print("Beginning the training.")
    

    # MAIN IMPLEMENTATION
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(INIT_LRATE, decay_steps=DECAY_STEPS, decay_rate=DECAY_RATE, staircase=True)
    loaded_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.MeanIoU(num_classes=2)]) 
    historique = loaded_model.fit(augmentated_datasets[0], augmentated_datasets[1], epochs=EPOCHS, validation_data=(x_validation, y_validation), batch_size=BATCH_SIZE)
    print("Training done.")     
    
    # SAVE THE NEW MODEL
    loaded_model.save(saved_weights_path)

    
    # récupérer metrics
  
    data = historique.history
    data['historique'] = str(EPOCHS)+str("/")+str(INIT_LRATE)+str("/")+str(DECAY_STEPS)+str("/")+str(BATCH_SIZE)
    

    
    with open(segmentation_results_path+'historique.txt', 'w') as outfile:
        json.dump(data, outfile)
        
    #plot_diagrams(1, segmentation_results_path+'historique.txt', segmentation_results_path+"graph.png")

    
    # TESTING THE MODEL
    file_nbr = 0
    for data in testing_dataset:
        predictions = loaded_model.predict(np.reshape(data, (1, 160, 160))) # Faire prédiction
        predictions = np.reshape(predictions, (160, 160)) # Remettre au bon format le tableau des prédictions

        # Remplacer la prédiction par la valeur de pixel qu'elle approxime 
        classified_predictions = np.ones((160, 160))
        for i in range(160):
            for j in range(160):
                if predictions[(i, j)] < 0.5:
                    classified_predictions[(i, j)] = 0 # noir
                else: 
                    classified_predictions[(i, j)] = 255 # blanc

        seg_img = Image.fromarray(classified_predictions) # Transformer le tableau np en image
        seg_img = seg_img.convert('L') # Convertir l'image en noir et blanc
        #seg_img.show() # Afficher l'image
        filename = segmentation_results_path + testing_filenames[file_nbr] + ".jpg"
        seg_img.save(filename)
        file_nbr += 1
        
        

def main():
    # segmentation_results_path, saved_weights_path, console_counter=0, EPOCHS=1, INIT_LRATE=0.001, DECAY_STEPS=1e5, BATCH_SIZE=64

    segmentation_results_path = "oral_dataset/testing/segmentation1/"
    if isdir(segmentation_results_path) == False:
        makedirs(segmentation_results_path)
    saved_weights_path = "oral_dataset/testing/segmentation1/OASBUD_unet_b_IOU.h5"
    training_session(segmentation_results_path, saved_weights_path, 50, 0.0005, 1e5, 2)
        
if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(10)