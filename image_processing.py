import numpy as np
from scipy import signal
import pickle
from PIL import Image
import matplotlib.pyplot as plt

def crop_from_array(rf, max_width, max_height):
    '''Crop an image array to a chosen shape. 
    Parameters:
    ----------
    rf: np.ndarray
        image
    '''
    
    (height, width) = np.shape(rf)
    high_cut = True # On commence par couper par le haut et à gauche et ensuite on alterne avec couper par le bas et à droite
    while width > max_width or height > max_height:
        if width > max_width and height > max_height:
            # couper selon les 2 axes
            if high_cut == True:
                rf = np.delete(rf, 0, axis=0) # cut off line 0
                rf = np.delete(rf, 0, axis=1) # cut off column 0
                high_cut = False
            else:
                rf = np.delete(rf, height-1, axis=0) # cut off last line
                rf = np.delete(rf, width-1, axis=1) # cut off last column 
                high_cut = True      
        elif width > max_width and height <= max_height: # Si la largeur n'est pas bonne
            # couper selon axis = 1 uniquement
            if high_cut == True:
                rf = np.delete(rf, 0, axis=1) # cut off 1st column
                high_cut = False
            else: 
                rf = np.delete(rf, width-1, axis=1) # cut off last column
                high_cut = True
        elif width <= max_width and height > max_height: # Si la hauteur n'est pas bonne
        # couper selon axis = 0 uniquement 
            if high_cut == True:
                rf = np.delete(rf, 0, axis=0) # cut off 1st line
                high_cut = False
            else: 
                rf = np.delete(rf, height-1, axis=0) # cut off last line
                high_cut = True
        (height, width) = np.shape(rf)

    return rf


def add_pixels(rf, max_width, max_height):
    '''Add black pixels to get an image array to shape (160, 160) 
    Parameters:
    ----------
    rf: np.ndarray
        image
    '''
    (height, width) = np.shape(rf)
    high_add = True # on commence par ajouter des pixels par le haut et à gauche et ensuite on alterne avec par le bas et à droite
    while width < max_width or height < max_height:
        if width < max_width and height < max_height:
            # ajouter selon les 2 axes
            if high_add == True:
                rf = np.insert(rf, 0, 0,  axis=0) # cut off line 0
                rf = np.insert(rf, 0, 0, axis=1) # cut off column 0
                high_add = False
            else:
                rf = np.insert(rf, height-1, 0, axis=0) # cut off last line
                rf = np.insert(rf, width-1, 0, axis=1) # cut off last column 
                high_add = True      
        elif width < max_width and height >= max_height: # Si la largeur n'est pas bonne
            # couper selon axis = 1 uniquement
            if high_add == True:
                rf = np.insert(rf, 0, 0, axis=1) # cut off 1st column
                high_add = False
            else: 
                rf = np.insert(rf, width-1, 0, axis=1) # cut off last column
                high_add = True
        elif width >= max_width and height < max_height: # Si la hauteur n'est pas bonne
        # couper selon axis = 0 uniquement 
            if high_add == True:
                rf = np.insert(rf, 0, 0, axis=0) # cut off 1st line
                high_add = False
            else: 
                rf = np.insert(rf, height-1, 0, axis=0) # cut off last line
                high_add = True
        (height, width) = np.shape(rf)
        
    return rf


def shape_to(rf, max_width=160, max_height=160):
    '''Reshape an image array by adding or cutting pixels. '''
    (height, width) = rf.shape
    if height < max_height and width < max_width:
        rf = add_pixels(rf, max_width, max_height)
    elif height > max_height and width > max_width:
        rf = crop_from_array(rf, max_width, max_height)
    else:
        rf = add_pixels(rf, max_width, max_height)
        rf = crop_from_array(rf, max_width, max_height)
        
    return rf


def my_resample2(rf, width, height):
    '''Reduce size array to approximately widthxheight size. Need to complete with shape_to function.'''
    rf = signal.resample_poly(rf, 1, round(np.shape(rf)[0]/height)) 
    rf = signal.resample_poly(rf, 1, round(np.shape(rf)[1]/width), axis=1)
    return rf


def my_resample3(rf, width, height):
    '''Reduce size array to approximately widthxheight size. Need to complete with shape_to function. '''
    rf = signal.decimate(rf, round(np.shape(rf)[0]/height), ftype='fir', axis=0)
    rf = signal.decimate(rf, round(np.shape(rf)[1]/width), ftype='fir', axis=1)
    return rf


def regular_sample(rf, max_width=160, max_height=160):
    '''Reduce size array to approximately widthxheight size. Need to complete with shape_to function. '''
    (height, width) = np.shape(rf)
    fe_x = round(width/max_width)
    fe_y = round(height/max_height)
    rf = rf[:, range(0, width, fe_x)]
    rf = rf[range(0, height, fe_y), :]
    return rf
    
    
def random_sample(rf):
    '''Supposed to reduce size array but it just creates a noisy image. '''

    import random
    (height, width) = np.shape(rf)
    aleatory_sample = np.ones((height, width))
    for i in range(160):
        for j in range(160):
            x = random.randint(0, height-1)
            y = random.randint(0, width-1)
            aleatory_sample[(i, j)] = rf[(x, y)]
    img = Image.fromarray(aleatory_sample)
    img = img.convert("L")
    return aleatory_sample


def load_pkl_file(saving_path):
    '''Load the image array from a pickle file. 
    Parameters
    ----------
    saving_path: str
        pathname of the file to be loaded
    '''
    
    with open(saving_path, 'rb') as rf_array :
        rf = pickle.load(rf_array)

    return rf


def disp_img_from_array(img_array):
    '''Display grayscale image from array. '''
    img = Image.fromarray(img_array)
    img = img.convert("L")
    
    img.show()    
    
    
def save_img_from_array(rf, path_to_save):
    '''Save grayscale image from array. '''
    img = Image.fromarray(rf)
    img = img.convert("L")

    img.save(path_to_save)
    
    
def save_pkl_from_array(rf, path_to_save):
    '''Save an array in a pickle file. '''
    with open(path_to_save, "wb") as pkl_file:
        pickle.dump(rf, pkl_file)
        
        
def create_mask(rf):
    '''Create a mask array to make it visible. Useful when mask are made of 0 and 1 values. '''
    for i in range(np.shape(rf)[0]):
        for j in range(np.shape(rf)[1]):
            if rf[(i, j)] > 0:
                rf[(i, j)] = 255
    return rf


def make_predictions(us_img_array, model):
    '''Gerate the 2D array predicted by the CNN.
    Parameters
    ----------
    us_img_array: 2D array
        ultrasound image used to make the predictions
        
    Returns 2D array with pixel values between 0 and 1, the segmentated image predicted
    '''
    predictions = model.predict(np.reshape(us_img_array, (1, 160, 160))) 
    return np.reshape(predictions, (160, 160)) # Remettre au bon format le tableau des prédictions
    

def create_img_from_predictions(predicted_array, pixal_format=True):
    '''Display the binary image following the predictions made by the CNN. 
    Parameters
    ----------
    predicted_array: 2D array
        the segmentated image predicted with pixel values between 0 and 1
    pixal_format: bool
        choose if the segmentated image is changed to Pixal format to display it in the UI
    Returns 2D array with pixel values of 0 and 255, the segmentated image predicted
    '''
    for i in range(160):
        for j in range(160):
            if predicted_array[(i, j)] < 0.5:
                predicted_array[(i, j)] = 0 # noir
            else: 
                predicted_array[(i, j)] = 255 # blanc
    if pixal_format == True:
        return predicted_array.astype(np.uint8) 
    else:
        return predicted_array

def equa_hist(img_array):
    '''Rise contrast by equalization histogram method and display the ancient and the new histogram.  '''
    # Histogram creation and representation
    hist, bins = np.histogram(img_array, 256, [0, 256])
    hist = np.append(hist, 0)
    represent_histogram(img_array, "Avant égalisation")
    
    # Histogram equalization
    cdf = hist.cumsum()
    cdf = (255*cdf) / cdf[-1]
    equa_img_array = np.interp(img_array, bins, cdf)
    np.reshape(equa_img_array, np.shape(img_array))
    represent_histogram(equa_img_array, "Après égalisation")
    
    return equa_img_array


def represent_histogram(img_array, hist_legend):
    '''Display an histogram using a image array. '''
    hist, bins = np.histogram(img_array, 256, [0, 256])
    hist = np.append(hist, 0)
    plt.plot(bins, hist, label=hist_legend)
    plt.legend(loc='upper right')
    plt.show()