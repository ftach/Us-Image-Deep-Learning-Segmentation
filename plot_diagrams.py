# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 11:20:55 2021

@author: ftachenn
"""
import json
import matplotlib.pyplot as plt
    # faire un graph des IOU

def plot_diagrams(quantity, historique_path, graph_saving_path):
    iou_name = 'mean_io_u'
    val_iou_name = 'val_mean_io_u'
    
    i = 0
    while i < quantity:

        new_iou_name = 'mean_io_u'
        new_val_iou_name = 'val_mean_io_u'
        if i > 0:
            new_iou_name = iou_name + "_" + str(i)
            new_val_iou_name = val_iou_name + "_" + str(i)  
            
        with open(historique_path, 'r') as saving_file:
            historique = json.load(saving_file)
            
        hyperparameters = historique["historique"]
        loss = historique["loss"]
        val_loss = historique["val_loss"]
        iou = historique["mean_io_u_1"]
        val_iou = historique["val_mean_io_u_1"]
        
        plt.figure(figsize=(8, 8))
        # IOU graph
        plt.subplot(2, 1, 1)
        plt.plot(iou, label='IOU')
        plt.plot(val_iou, label='Validation IOU')
        plt.legend(loc='lower right')
        plt.ylabel('IOU')
        plt.ylim([min(plt.ylim()),1])
        plt.title('Training and Validation IOU')
        # LOSS graph
        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0,1.0])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        
        plt.savefig(graph_saving_path)
        plt.close()
        i+=1

plot_diagrams(1, "oral_dataset/testing/segmentationhistorique.txt", "oral_dataset/testing/graphe.png")