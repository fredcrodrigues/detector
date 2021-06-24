from __future__ import print_function
from keras import optimizers
#os.environ["KERAS_BACKEND"] = "tensorflow"
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input,concatenate,Conv2D,MaxPooling2D,AveragePooling2D,Conv2DTranspose,Layer,Cropping2D
from keras.layers import Dense,Flatten, Dropout,BatchNormalization, ZeroPadding2D,Lambda
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint ,ReduceLROnPlateau
from keras.layers.core import Activation
from keras import backend as K
from keras.applications import vgg16
from numpy.core.fromnumeric import resize 
from numpy.lib.function_base import append
from scipy.ndimage.measurements import label
from scipy.spatial import distance
from skimage.transform import hough_ellipse
from skimage import measure
from tensorflow.python.keras.backend import dtype, shape
from PIL import Image

import os
import copy

import cv2 as cv
import statistics as st
import statistics
import pickle  as pk
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import PIL

from skimage import metrics
from scipy.spatial.distance import directed_hausdorff


from scipy.ndimage import morphology

path = os.getcwd()



def houg(canny_pred_iris , canny_pred_pupil  ,conv_image ,i_origin ): 

    circles_pred = cv.HoughCircles(canny_pred_iris, cv.HOUGH_GRADIENT , 1 , 20 , param1=2, param2=25, minRadius=0 , maxRadius=0)
    circles_pred_pupil= cv.HoughCircles(canny_pred_pupil , cv.HOUGH_GRADIENT , 1 , 20 , param1=2, param2=10, minRadius=0 , maxRadius=0)

    pred_image = conv_image

    ##TEST
    #cv.imwrite(path + '/segmentation_pupil/CANNY_IRIS.png' ,canny_pred_pupil)
    #cv.imwrite(path + '/segmentation_pupil/CANNY_PUPI.png', canny_pred_iris)

    if circles_pred is not None: 
        circles_pred = np.uint16(np.around(circles_pred))
        for i in circles_pred[0,:]:
           cv.circle( i_origin, (i[0] , i[1]) , i[2] ,(0,255,0), 1)
       
          
    if circles_pred_pupil is not None: 
        circles_pred_pupil = np.uint16(np.around(circles_pred_pupil))
        for x in circles_pred_pupil[0,:]:

           cv.circle( i_origin, (x[0],x[1]),x[2],(0,0,255), 1)
         
    return i_origin
    #image_pred_iris = cv.resize(image_pred_iris , dsize=(300,300) , interpolation=cv.INTER_CUBIC)
    #DIFERENÇA ENTRE OS DIAMENTROS ENCONTRADOS
    

    
def pos_processing(pred1 , pred2 ,image , i_origin):
    conv_pred_iris = pred1.astype('uint8')
    conv_pred_pupil = pred2.astype('uint8')
    conv_image = image


    #conv_image = conv_image.astype('uint8')

    
    norm_pred_iris = np.zeros(conv_pred_iris.shape)
    norm_pred_iris = cv.normalize(conv_pred_iris ,  norm_pred_iris , 0 ,255 , cv.NORM_MINMAX)

    norm_pred_pupil = np.zeros(conv_pred_pupil.shape)
    norm_pred_pupil = cv.normalize(conv_pred_pupil ,  norm_pred_pupil , 0 ,255 , cv.NORM_MINMAX)
    
    ### CANNY ###

    canny_pred_iris = cv.Canny(norm_pred_iris , 60 , 200)
    canny_pred_pupil = cv.Canny(norm_pred_pupil , 60, 200)
   

    d_image= houg(canny_pred_iris , canny_pred_pupil  ,conv_image , i_origin)

    
    return d_image


def c_remove(mask):
      
    labeli = measure.label(mask)
    regions = measure.regionprops(labeli)
    
    area_maior = regions[0].area
    area_menor = regions[1].area
    #print('Area' , area_maior)
    #print('Area 2' , area_menor)

    for props in range(len(regions)):
    #labeli[regions[prop].coords[:,0] , regions[prop].coords[:,1]] = 0
    # if prop == 1 and prop ==0:
        if area_maior > area_menor:
            labeli[regions[0].coords[:,0] , regions[0].coords[:,1]] = 1
            mask[labeli==1] = 0 
        if area_maior < area_menor:
            labeli[regions[0].coords[:,0] , regions[0].coords[:,1]] = 1
            mask[labeli==1] = 0 

        if props == 2:
            labeli[regions[2].coords[:,0] , regions[2].coords[:,1]] = 1
            mask[labeli==1] = 0 
    return mask
def c_divid(mask):

    mask = cv.cvtColor( mask ,  cv.COLOR_BGR2GRAY)
    
    lin , col   = mask.shape  

    for l  in range(0 , lin):
        for c in range(0 , col):
                valup = mask[l,c]
                if valup == 150:
                    mask[l,c] = 0
                if valup != 255:
                    mask[l,c] = 0
    return mask
def c_pixel(mask):
    lin , col  , channel  = mask.shape  
    for l in range(0 , lin):
        for c in range(0 , col):
                valup = mask[l,c]
                if valup[0] == 255  and valup[1] == 255  and valup[2] == 255:
                    black = 0
                    mask[l,c] = black
                
                if valup[0] == 0  and valup[1] == 255  and valup[2] == 0:
                    black = 0
                
                    mask[l,c] = black
                    break
    return mask
def c_limite(mask):
    lin , col  , channel  = mask.shape   

    for l  in range(0 , lin):
        for c in range(0 , col):
                valup = mask[l,c]
                if valup[0] >= 0  and valup[1] >= 0  and valup[2] >= 0 and valup[0] <= 255 and valup[1] <= 250  and valup[2] <= 255:
                    white = 255 
                    mask[l,c] = white

    return mask
def s_circle(mask):
   
    #print('Shape 1' , mask.shape)
    # AJUSTE DAS IMAGENS BINARIZADAS

    kernel = np.ones((5,5) , np.uint8)
    mask = cv.dilate(mask , kernel , iterations=2)
    
   
    # A NORMALIZAÇÃO DOS PIXEL ENTRE 0 E 255 
    mask_gray = cv.cvtColor(mask , cv.COLOR_BGR2GRAY)
    mask_gray = mask_gray.astype('uint8')
    mask_gray[np.where(mask_gray == 1)] = 255 
    

    #cv.imshow('Cinza' , mask_gray)
    #cv.waitKey(0)

    ret, thresh = cv.threshold(mask_gray, 127, 255, 0)

    
    contours , h = cv.findContours(thresh , cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) 
    ## SMALL CIRCLE
    (x,y) , radius = cv.minEnclosingCircle(contours[0])
    center  = (int(x) , int(y))
    radius = int(radius)
    circle_coturs = cv.circle(mask , center , radius , (0,255,0) , 2)
   
    ## FUNÇÃO QUE VERIFICA LIMITES DO SMAAL CIRCLE

    mask = c_limite(mask)
  
    mask = c_pixel(mask)

    mask = c_divid(mask)
   
 


    mask  = c_remove(mask)

    #AJUSTES FINAIS#
    kernel = np.ones((3,3) , np.uint8)
    clossing = cv.morphologyEx(mask , cv.MORPH_CLOSE , kernel)
    

    ##AJUSTA PARA O TAMANHO PROXIMO 
    kernel = np.ones((5,5) , np.uint8)
    mask = cv.erode(clossing , kernel , iterations=1)


    mask = mask.astype('float32')
    mask[np.where(mask == 255)] = 1 

    mask = np.stack((mask,)*3, axis=-1)


    return mask

def v_remove(pred):
    labels_pred= measure.label(pred)
    region_prop_pred= measure.regionprops(labels_pred)
    
    if len(region_prop_pred) > 1 :
        print('Entrou na condição 2' )
        kernel = np.ones((5,5) , np.uint8)
        pred = cv.morphologyEx(pred , cv.MORPH_OPEN  , kernel ,  iterations=3)

    return pred
def r_maior(b_prin):
    ## SELECIONA A PRIMEIRA AREA QUE É A MENOR IRIS
    labels= measure.label(b_prin)
    region_prop = measure.regionprops(labels)
    
    if len(region_prop)> 1:
        b_area = region_prop[0].area 
        for props in range(len(region_prop)):
           # print(region_prop[props].area)
            
            if b_area > region_prop[props].area: 
                labels[region_prop[0].coords[:,0] , region_prop[0].coords[:,1]] = 1
                b_prin[labels==1] = 0 
            if b_area < region_prop[props].area: 
                labels[region_prop[0].coords[:,0] , region_prop[0].coords[:,1]] = 1
                b_prin[labels==1] = 0 
        
    return b_prin
def r_menor(pred):

    labels_pred= measure.label(pred)
    region_prop_pred= measure.regionprops(labels_pred)
    
    if len(region_prop_pred) > 1 :
        print('Entrou na condição')
        kernel = np.ones((5,5) , np.uint8)
        pred = cv.morphologyEx(pred , cv.MORPH_OPEN  , kernel ,  iterations=2)
        pred = v_remove(pred) ##SEGUNDA REMOÇÃO SE OUVER NECESSIDADE

    return pred
def r_area(pred):
    #cv.imshow('PRED' , pred)
    #cv.waitKey(0)
    
    ### REMOVE BLOBS MENORES ## PUPIL E IRIS
    b_prin = r_menor(pred)

    labels= measure.label(b_prin)
    region_prop = measure.regionprops(labels)
    

    ## REMOVE PEQUENOS BLOBS SE POR ACASO NÃO FORAM REMOVIDOS ACIMA
    if len(region_prop) > 1 and len(region_prop) < 3:

       kernel = np.ones((3,3) , np.uint8)
       b_morf = cv.morphologyEx(b_prin , cv.MORPH_OPEN  , kernel ,  iterations=5)
       b_prin = b_morf 
   
    ## REMOVE PARTES MAIORES FORA DA REGIÃO DE INTERESSE 
    r_image = r_maior(b_prin)

    return r_image


class Unet():
    
    def __init__(self , img_rows,img_cols,img_channels=3,batch_size=5,path_weights=None):
        
        self.img_rows=img_rows
        self.img_cols=img_cols
        self.img_channels=img_channels
        self.path_weights = path_weights
        self.batchsize=batch_size
        self.build_model()

   
    def build_model(self):
        
        ###ARUQTETURA COM VGG 16 E UM OUTPUT##

        self.vgg = vgg16.VGG16(include_top=False , weights='imagenet', input_shape=(self.img_rows , self.img_cols , self.img_channels)) 
        
        VGG16M = self.vgg 
        
        enconder = self.vgg.output
      
       ##FREEZE LAYERS
        set_trainable = False

        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(VGG16M.get_layer("block5_conv3").output), VGG16M.get_layer("block4_conv3").output], axis=3)
        conv6 = Conv2D(256, (3, 3), padding='same')(up6)
        conv6 = LeakyReLU()(conv6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Dropout(0.2)(conv6)
        conv6 = Conv2D(256, (3, 3), padding='same')(conv6)
        conv6 = LeakyReLU()(conv6)
        conv6 = BatchNormalization()(conv6)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), VGG16M.get_layer("block3_conv3").output], axis=3)
        conv7 = Conv2D(128, (3, 3), padding='same')(up7)
        conv7 = LeakyReLU()(conv7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Dropout(0.2)(conv7)
        conv7 = Conv2D(128, (3, 3), padding='same')(conv7)
        conv7 = LeakyReLU()(conv7)
        conv7 = BatchNormalization()(conv7)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7),VGG16M.get_layer("block2_conv2").output], axis=3)
        conv8 = Conv2D(64, (3, 3), padding='same')(up8)
        conv8 = LeakyReLU()(conv8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Dropout(0.2)(conv8)
        conv8 = Conv2D(64, (3, 3), padding='same')(conv8)
        conv8 = LeakyReLU()(conv8)
        conv8 = BatchNormalization()(conv8)
        
        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), VGG16M.get_layer("block1_conv2").output], axis=3)
        conv9 = Conv2D(32, (3, 3), padding='same')(up9)
        conv9 = LeakyReLU()(conv9)
        conv9 = Dropout(0.2)(conv9)
        conv9 = Conv2D(32, (3, 3), padding='same')(conv9)
        conv9 = LeakyReLU()(conv9)
        
        conv10 = Conv2D(3 , (1, 1) ,padding='same')(conv9)
        conv11 = Conv2D(3 , (1, 1) ,padding='same')(conv9)

        conv12 = Activation('sigmoid' ,  name='conv12')(conv10)
        conv13 = Activation('sigmoid' , name='conv13')(conv11)
        
        self.model = Model(VGG16M.input , [conv12 , conv13])
        
  
    def load(self):
        self.model.load_weights(path + '/weigths/segmentation.h5')
        

    def test(self , image , i_origin ,n_image , origen):
        
        cv.imwrite(path + '/segmentation_pupil/image_origin.png' , i_origin)
        p_image = np.expand_dims(image, axis=0)
        
        

        pred1 , pred2  = self.model.predict(p_image)
       
        pred1[pred1>0.5] = 1.0
        pred1[pred1<0.5] = 0.0

        pred2[pred2>0.5] = 1.0
        pred2[pred2<0.5] = 0.0
    

        pred1 = pred1[0]
        pred2 = pred2[0]
        ## VERIFICA SE NA IMAGE, PREDITA EXISTE BLOB
        label_i = measure.label(pred1)
        regions_i = measure.regionprops(label_i)

        label_p = measure.label(pred2)
        regions_p = measure.regionprops(label_p)
        
        if len(regions_i) != 0:
                ## REMOÇÃO DE BLOBS 
                pred1 = r_area(pred1)
                pred1 = s_circle(pred1)

        if len(regions_p) != 0:

            ## REMOÇÃO DE BLOBS 
            pred2 = r_area(pred2)
            ## SMALL CIRCLE
            pred2 = s_circle(pred2)

            
            ##TEST
            pred1_i =  pred1.astype('uint8')
            pred2_i = pred2.astype('uint8')
            pred1_i[np.where(pred1_i== 1)] = 255
            pred2_i[np.where(pred2_i== 1)] = 255
            
            #cv.imwrite(path + '/segmentation_pupil/testes/' + n_image + 'iris_circle' + '.png' ,pred1_i)
            #cv.imwrite(path + '/segmentation_pupil/testes/' + n_image + 'pupila_circle' + '.png', pred2_i)
            
        
        d_image = pos_processing(pred1 , pred2 , image ,i_origin)
     
        return d_image 

     
        
Model = Unet(224,224,img_channels=3,batch_size=16)

def load_image(i_origin , n_image):
 
    cv.imwrite(path + '/results/' + n_image ,i_origin)
    i_origin = i_origin.astype('uint8')
    h , l , c = i_origin.shape
    
    t_crop = i_origin

    ##PRE-PROCESSAMENTO
    
    #c_image = i_origin[30:254,30:254]
    c_image = cv.resize(i_origin , (224,224) , cv.INTER_LINEAR)
    p_image =  c_image/255.0
    
    print(c_image.shape)
    
    
    #AJUSTES 
   
    
    Model.load()
  
    i_detect = Model.test(p_image ,  c_image , n_image , i_origin)
    
  
    i_resize  = cv.resize(i_detect , (l ,h) , cv.INTER_LINEAR)

    add = cv.addWeighted(i_origin ,  0.7, i_resize ,  0.6 , 0)
   #i_origin[30:254,30:254] =  i_detect

    cv.imwrite(path + '/results/' + n_image + "tes.png", add  )
    
    #return i_origin
    

 