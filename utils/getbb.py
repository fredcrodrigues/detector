import os
import cv2
import glob
import random
import numpy as np
from numpy.core.arrayprint import printoptions
from numpy.core.fromnumeric import sort
import tifffile as tff
from shutil import copyfile
from PIL import Image

import matplotlib.patches as mpatches
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import closing, square
from skimage.measure import label, regionprops

from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img

import os


cwd = os.getcwd()


path_images = cwd + '/mascara_iris/'
path_masks = r'F:/Alan/Documentos/Doutorado/Bases/polipo576/kvasir/masks'

def getBoundingBox(path, result_file):
	print("Generating bounding boxes")
	files = []
	for (dirpath, dirnames, filenames) in os.walk(path):
		files.extend(filenames)
		break  
	for name_past  in os.listdir(path):
		sorted(name_past)
		if name_past != '015':
			for name_p in os.listdir(path + name_past + '/mascara'):
				if name_p != 'DEXTRO.psd' or name_p != 'INFRA.psd' or name_p != 'DEXTRO.psd' or   name_p != 'SUPRA.psd':
					path_f = path + name_past + '/mascara/' + name_p
					print(path_f)
					mask = cv2.imread(path_f, 0)

					thresh = threshold_otsu(mask)
					bw = closing(mask > thresh, square(3))
					cleared = clear_border(bw)
					label_image = label(cleared)
					rect = mask.copy()
					f = open(str(result_file) + ".txt", "a")	
					for region in regionprops(label_image):
						#i+=1 
						if region.area >= 100:
							minr, minc, maxr, maxc = region.bbox
							
							#print(minr, minc, maxr, maxc)
							#rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
							#                         fill=False, edgecolor='red', linewidth=2)

							rect = cv2.rectangle(rect,(minc-60, minr-60),(maxc +60, maxr + 60), (255,255,255), 1)
							resize = cv2.resize(rect , (550 , 550))
							#cv2.imshow('test rect' , resize)

							#cv2.waitKey(0)	
							f.write(str(path_f ) + ',' + str(minc) + ',' + str(minr) + ',' + str(maxc) + ',' + str(maxr) + ',1\n')
					
					#cv2.imwrite('resultados/' + file, rect)
					f.close() 
					#print(name_p)
				
	'''
	print(path)
	print(len(files))
	i = 0
	for file in files:
		filepath = path + "/" + file
		if os.path.isfile(filepath):
			#print('********    file   ********', filepath)
			mask = cv2.imread(filepath, 0)
			thresh = threshold_otsu(mask)
			bw = closing(mask > thresh, square(3))
			cleared = clear_border(bw)
			label_image = label(cleared)
			rect = mask.copy()
			f = open(str(result_file) + ".txt", "a")	
			for region in regionprops(label_image):
				i+=1 
				if region.area >= 100:
					minr, minc, maxr, maxc = region.bbox
					
					#print(minr, minc, maxr, maxc)
					#rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
					#                         fill=False, edgecolor='red', linewidth=2)

					rect = cv2.rectangle(rect,(minc-60, minr-60),(maxc +60, maxr + 60), (255,255,255), 1)
					resize = cv2.resize(rect , (550 , 550))
					cv2.imshow('test rect' , resize)

					cv2.waitKey(0)	
					f.write(str(filepath) + ',' + str(minc) + ',' + str(minr) + ',' + str(maxc) + ',' + str(maxr) + ',1\n')
			
			#cv2.imwrite('resultados/' + file, rect)
			f.close() 
	print(i)  
	'''

if __name__ == '__main__':

	#print(path_images)
	#img = cv2.imread(path_images)
	#print(img)


	getBoundingBox(path_images, 'Olhos')
