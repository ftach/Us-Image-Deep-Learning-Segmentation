# -*- coding: utf-8 -*-
"""
Created on Sun May 30 01:27:18 2021

@author: flore
"""
from scipy import signal
from PIL import Image 
import numpy as np
import tensorflow as tf
import sys, os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from image_processing import *


class SegTool(QMainWindow):
    
    def __init__(self):
        super().__init__()
            # GRANDE FENETRE
        self.initUI()
    
    
    def initUI(self):
        
        # WINDOW TITLE AND LOGO
        self.setWindowTitle("Segmentation Tool")
        
        # MAIN WIDGET
        window_layout = QVBoxLayout()
        widget = QWidget()
        widget.setLayout(window_layout)
        
        # HELP BUTTON
        help_button = QPushButton("Aide")
        help_button.setMaximumWidth(100)
        window_layout.addWidget(help_button, alignment=Qt.AlignRight)

        # MAIN UNDER WIDGETS
        parametre_box = QGroupBox("Paramètres")
        parametre_box.setMaximumHeight(200)
        window_layout.addWidget(parametre_box)
        
        images_box = QGroupBox("Images")
        window_layout.addWidget(images_box)
        
        interaction_box = QFrame()
        interaction_box.setMaximumSize(400, 100)
        window_layout.addWidget(interaction_box, alignment=Qt.AlignRight)
    
    # FRAME PARAMETRES
        choose_model = QLabel("Choisir le fichier correspondant au modèle de segmentation")
        choose_model.setMaximumWidth(200)
        self.models_box = QComboBox()
        models = ["UNET_a_160.h5", "UNET_b_160_IOU.h5", "UNET_a_160_IOU.h5"]
        self.selected_model = "models/" + models[0]
        self.models_box.addItems(models)    
        self.models_box.currentIndexChanged.connect(self.modelSelection)
        self.models_box.setMaximumWidth(200)
    
        choose_us_image = QLabel("Choisir image ultrasonore")
        choose_us_image.setMaximumWidth(200)
        
        browse_us_image = QPushButton("Parcourir")
        browse_us_image.setMaximumWidth(200)
        browse_us_image.clicked.connect(self.getUsFile)
        
        contrast_image = QPushButton("Améliorer le contraste de l'image")
        contrast_image.setIcon(QIcon("stars2.png"))
        contrast_image.setMaximumWidth(200)
        contrast_image.clicked.connect(self.contrastImage)
        
        crop_image = QPushButton("Rogner l'image")
        crop_image.setMaximumWidth(200)
        crop_image.clicked.connect(self.cropImage)
        
        choose_mask_image = QLabel("Voulez vous ajouter une masque de comparaison?")
        choose_mask_image.setMaximumWidth(400)
        

        browse_mask_image = QPushButton("Parcourir")
        browse_mask_image.setMaximumWidth(200)
        browse_mask_image.clicked.connect(self.getMaskFile)

        parametres_layout = QGridLayout(parametre_box)
        
        parametres_layout.addWidget(choose_model, 0, 0)
        parametres_layout.addWidget(self.models_box, 0, 1)
        parametres_layout.addWidget(choose_us_image, 1, 0)
        parametres_layout.addWidget(browse_us_image, 1, 1)
        parametres_layout.addWidget(contrast_image, 1, 2)
        parametres_layout.addWidget(crop_image, 1, 3)
        parametres_layout.addWidget(choose_mask_image, 2, 0)

        parametres_layout.addWidget(browse_mask_image, 2, 1)
    
    #  FRAME AFFICHAGE D'IMAGE
        # US IMAGE FRAME
        us_img_frame = QFrame()
        self.configureImageFrame(us_img_frame)

        us_img_label = QLabel("Image ultrasonore", us_img_frame)
        us_img_label.setAlignment(Qt.AlignCenter)
        self.us_img_view = QLabel(us_img_frame)
        self.us_img_view.setAlignment(Qt.AlignCenter)
        
        self.configureImageLayout(us_img_frame, us_img_label, self.us_img_view)
        
        # CONTRASTED IMAGE FRAME
        self.contrasted_img_frame = QFrame()
        self.configureImageFrame(self.contrasted_img_frame)
        
        self.contrast_activated = False
        contrasted_us_img_label = QLabel("Image contrastée")
        contrasted_us_img_label.setAlignment(Qt.AlignCenter)
        self.contrasted_us_img_view = QLabel()
        self.contrasted_us_img_view.setAlignment(Qt.AlignCenter)
        
        self.configureImageLayout(self.contrasted_img_frame, contrasted_us_img_label, self.contrasted_us_img_view)
        
        # FRAME MASK IMAGE
        self.mask_img_frame = QFrame()
        self.configureImageFrame(self.mask_img_frame)
        
        mask_img_label = QLabel("Masque")
        mask_img_label.setAlignment(Qt.AlignCenter)
        self.mask_img_view = QLabel()
        self.mask_img_view.setAlignment(Qt.AlignCenter)
        
        self.configureImageLayout(self.mask_img_frame, mask_img_label, self.mask_img_view)
        
        # SEGMENTATION IMAGE FRAME
        seg_img_frame = QFrame()
        self.configureImageFrame(seg_img_frame)
        
        seg_img_label = QLabel("Image segmentée")
        seg_img_label.setAlignment(Qt.AlignCenter)
        self.seg_img_view = QLabel()
        self.seg_img_view.setAlignment(Qt.AlignCenter)
        
        self.configureImageLayout(seg_img_frame, seg_img_label, self.seg_img_view)
                
        # ALL IMAGES LAYOUT
        image_layout = QGridLayout(images_box)
        image_layout.addWidget(us_img_frame, 0, 0)
        image_layout.addWidget(self.contrasted_img_frame, 0, 1)
        image_layout.addWidget(self.mask_img_frame, 0, 2)
        image_layout.addWidget(seg_img_frame, 0, 3)

        # FRAME A CACHER
        self.mask_img_frame.hide()
        self.contrasted_img_frame.hide()
    # FRAME INTERACTION        
        run_seg_button = QPushButton("Lancer segmentation")
        run_seg_button.setIcon(QIcon("run2.png"))
        run_seg_button.setMaximumWidth(300)
        run_seg_button.clicked.connect(self.runSegmentation)
        
        self.save_seg_button = QPushButton("Sauvegarder image segmentée")
        self.save_seg_button.setIcon(QIcon("save.ico"))
        self.save_seg_button.setMaximumWidth(300)
        self.save_seg_button.clicked.connect(self.saveSegmentation)
        
        interaction_layout = QGridLayout(interaction_box)
        
        interaction_layout.addWidget(run_seg_button, 0, 0)
        interaction_layout.addWidget(self.save_seg_button, 0, 1)

        self.setCentralWidget(widget)
        
        
    def configureImageFrame(self, frame):
        frame.setFrameStyle(QFrame.Box)
        frame.setLineWidth(1)
        frame.setFrameShadow(QFrame.Sunken)
        
        
    def configureImageLayout(self, parent, text_label, image_view):
        layout = QGridLayout(parent)
        layout.addWidget(text_label, 0, 1, 1, 1, alignment=Qt.AlignTop)
        layout.addWidget(image_view, 1, 0, 3, 3)
        return layout
        
    
    def modelSelection(self):
        self.selected_model = "models/" + self.models_box.currentText() 


    def getUsFile(self):
        self.us_filename = QFileDialog.getOpenFileName(self, "Open image")[0]
        self.us_img_array = prepare_image(self.us_filename)
        self.dispUsImage(self.us_img_array)
        
        
    def dispUsImage(self, us_img_array):
        self.us_img = QImage(us_img_array, us_img_array.shape[1], us_img_array.shape[0], us_img_array.shape[1], QImage.Format_Grayscale8)
        self.us_img_view.setPixmap(QPixmap(self.us_img))
        
        
    def contrastImage(self):
        self.contrast_image_array = equa_hist(self.us_img_array).astype(np.uint8) 
        self.contrast_image = QImage(self.contrast_image_array, self.contrast_image_array.shape[1], self.contrast_image_array.shape[0],  
                                     QImage.Format_Grayscale8)
        self.contrasted_us_img_view.setPixmap(QPixmap(self.contrast_image))
        self.contrast_activated = True
        self.contrasted_img_frame.show()
      
        
    def cropImage(self):
        cropWidget = cropTool(self.us_img_array, self.us_img)
        cropWidget.exec()
        self.us_img_array = cropWidget.sendImage()
        self.dispUsImage(self.us_img_array)
        
        
    def getMaskFile(self):
        self.mask_filename = QFileDialog.getOpenFileName(self, "Open image")[0]
        self.mask_img_view.setPixmap(QPixmap(self.mask_filename))
        self.mask_img_frame.show()
        

    def runSegmentation(self):
        
        if self.contrast_activated == False:
            self.us_img_array = resize_and_sample(self.us_img_array)
            self.seg_img_array = make_prediction(self.us_img_array, self.selected_model)
        else:
            self.contrast_image_array = resize_and_sample(self.contrast_image_array)
            self.seg_img_array = make_prediction(self.contrast_image_array, self.selected_model)
            
        q_seg_img = QImage(self.seg_img_array, self.seg_img_array.shape[0], self.seg_img_array.shape[1], QImage.Format_Grayscale8)
        self.seg_img_view.setPixmap(QPixmap(q_seg_img))
        
        
    def saveSegmentation(self):
        seg_filename = QFileDialog().getSaveFileName(self, "Save image")[0]
        save_image(self.seg_img_array, seg_filename)


class cropTool(QDialog):
    
    def __init__(self, us_img_array, q_us_img):
        super().__init__()
        self.cropWidget = QDialog()
        
        self.cropWidget.setModal(True)
        self.setWindowTitle("Crop Tool")
        self.setFixedSize(1280, 800)

        image_to_crop_view = QLabel()
        self.image_to_crop = Image.fromarray(us_img_array)
        self.q_image_to_crop = QPixmap(q_us_img)
        
        self.window_width = self.size().width()
        self.window_height = self.size().height()
        
        self.img_width = self.q_image_to_crop.size().width()
        self.img_height = self.q_image_to_crop.size().height()
        
        self.setFixedSize(self.img_width, self.img_height)
        
        crop_button = QPushButton("Rogner image")
        crop_button.clicked.connect(self.cropImage)
        
        window_layout = QVBoxLayout(self)
        self.setLayout(window_layout)
        window_layout.addWidget(crop_button, alignment=Qt.AlignBottom)
        
        self.begin, self.destination = QPoint(), QPoint()

        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QColor(255, 255, 255))
        center_point = QPoint()
        self.new_x_origin = int(round((self.window_width-self.img_width)/2))
        self.new_y_origin = int(round((self.window_height-self.img_height)/2))
        center_point.setX(0)
        center_point.setY(0)
        painter.drawPixmap(center_point, self.q_image_to_crop)    
        
        if not self.begin.isNull() and not self.destination.isNull():
            rect = QRect(self.begin, self.destination)
            painter.drawRect(rect.normalized())
            
    def mousePressEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.begin = event.pos()
            self.destination = self.begin
            self.update()
        
    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.destination = event.pos()
            self.update()
        
    def mouseReleaseEvent(self, event):
        if self.begin.manhattanLength() < self.destination.manhattanLength():
            self.saveCoordinates(self.begin.x(), self.begin.y(), self.destination.x(), self.destination.y())
            
        elif self.begin.manhattanLength() > self.destination.manhattanLength(): 
            self.saveCoordinates(self.destination.x(), self.destination.y(), self.begin.x(), self.begin.y())

        elif event.buttons() & Qt.LeftButton:
            rect = QRect(self.begin, self.destination)
            painter = QPainter(self.q_image_to_crop)
            painter.drawRect(rect.normalized())
            
            self.begin, self.destination = QPoint(), QPoint()
            self.update()
            
    def saveCoordinates(self, x_begin, y_begin, x_destination, y_destination):
        self.x_begin = x_begin
        self.y_begin = y_begin
        self.x_destination = x_destination
        self.y_destination = y_destination
        
        
    def cropImage(self):
        self.croped_img = self.image_to_crop.crop((self.x_begin, self.y_begin, 
                                                   self.x_destination, self.y_destination)) # left, top, right, bottom
        self.close()
       
    def sendImage(self):
        return np.array(self.croped_img).astype(np.uint8)

def prepare_image(img_array):
    '''Transform an PIL image to array with the correct format to display it on PyQt. 
    Parameters
    ----------
    img_array: 2D numpy array
    
    Returns 2D numpy array, uint8 encoded
    '''
    img_array = Image.open(img_array)
    img_array = img_array.convert("L")
    
    return np.array(img_array).astype(np.uint8) 
    

def resize_and_sample(img_array):
    '''Sample and crop the image. 
    Parameters
    ----------
    img_array: : 2D numpy array
        image to resize and crop
    
    Returns 2D numpy array
    '''
    img_array = regular_sample(img_array) # échantilloner image
    sized_array = shape_to(img_array) # mettre à la bonne taille l'image
    
    return sized_array


def save_image(seg_img_array, seg_filename):
    '''Save an image from a 2D array. 
    Parameters
    ----------
    seg_img_array: 2D numpy array
        segmentated image to be saved
    seg_filename: str
        path wanted to save the segmentated picture
    
    Returns None
    '''
    seg_img = Image.fromarray(seg_img_array)
    seg_img = seg_img.convert("L")
    seg_img.save(seg_filename)
        

def make_prediction(sized_array, model_path="models/UNET_a_160.h5"): # garder le 1er de la combo box en valeur par défaut
    loaded_model = tf.keras.models.load_model(model_path, compile=False) # charger modèle
    seg_img_array = create_img_from_predictions(make_predictions(sized_array, loaded_model)) # faire la prédiction forward

    return seg_img_array


def main():
    app = QApplication(sys.argv)
    ex = SegTool()
    ex.showMaximized()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()