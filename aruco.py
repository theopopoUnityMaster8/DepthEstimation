# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 14:16:23 2022

@author: enora
"""

from PIL import Image
im = Image.open(r"D:/Documents/Etudes/Esir/Esir_3/Projet indus/Ensemble_test/sequence_4/depth167.png") 
import cv2
from cv2 import aruco
import  numpy as np


image = cv2.imread("D:/Documents/Etudes/Esir/Esir_3/Projet indus/Ensemble_test/sequence_4/color167.png")
depth_map = cv2.imread("D:/Documents/Etudes/Esir/Esir_3/Projet indus/Ensemble_test/sequence_4/depth167.png", cv2.IMREAD_UNCHANGED)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters_create()
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
frame_markers = aruco.drawDetectedMarkers(image.copy(), corners, ids)
print (corners)
print (int(corners[0][0][0][0]))
i = int(corners[0][0][0][0])
j = int(corners[0][0][0][1])
print(depth_map.dtype)




def click_event(event, x, y, flags, params):
 
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
 
        # displaying the coordinates
        # on the Shell
        x= x
        pix_val = depth_map[y,x]
        print(x,'  ', y)
        print(depth_map[y,x])
        MAX = 4320
        MIN = 0
        max_ = 255
        min_ = 0
        if (pix_val != 0):
            normalisée = (pix_val - MIN) * (max_ - min_) / (MAX - MIN) + min_
            
            x_aruco = 520
            y_aruco = 110
            print(f'la valeur du capteur aruco est environ : {depth_map[y_aruco,x_aruco]}')
            capteur = (depth_map[y_aruco,x_aruco] - MIN) * (max_ - min_) / (MAX - MIN) + min_
            
            dist = (normalisée*2.04)/capteur
            
            print(f'ce point se trouve à une distance de {dist}m de la caméra')
            
        else:
            print ('Pas de valeur pour cette zone')
            
def click_event_midas(event, x, y, flags, params):
 
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
 
        # displaying the coordinates
        # on the Shell
        x= x
        pix_val = midas[y,x]
        pix_val = 65000-pix_val
        print(x,'  ', y)
        print(midas[y,x])
        MAX = 65000
        MIN = 0
        max_ = 25000
        min_ = 0
        if (pix_val != 0):
            normalisée = (pix_val - MIN) * (max_ - min_) / (MAX - MIN) + min_
            capteur = (40458 - MIN) * (max_ - min_) / (MAX - MIN) + min_
            print(f'normalisé : {normalisée},  capteur : {capteur} ')
            
            dist = (normalisée*2.04)/capteur
            
            print(f'ce point se trouve à une distance de {dist}m de la caméra')
            
        else:
            print ('Pas de valeur pour cette zone')
        

x_aruco = 110
y_aruco = 520

color_image = cv2.imread("D:\Documents\Etudes\Esir\Esir_3\Projet indus\Ensemble_test\MiDas\color026.png")
midas = cv2.imread("D:\Documents\Etudes\Esir\Esir_3\Projet indus\Ensemble_test\MiDas\color026_MiDas.png", cv2.IMREAD_UNCHANGED)

cv2.imshow("midas", midas)
cv2.setMouseCallback('midas', click_event_midas)
cv2.waitKey(0) 

depth_img = depth_map.astype(np.uint8)
depth_img = cv2.circle(depth_img, (y_aruco,x_aruco), radius=1, color=(255, 255, 255), thickness=5)
cv2.imshow("depth image", depth_img)
cv2.setMouseCallback('depth image', click_event)
cv2.waitKey(0) 


im_rgb = cv2.cvtColor(frame_markers, cv2.COLOR_BGR2RGB)
cv2.imshow("aruco", frame_markers)
cv2.waitKey(5000) 

markerSizeInCM = 0.184 #en mètre

mtx = []
fx=612.069
fy=612.134
cx=637.802
cy=367.899
mtx.append([fx, 0, cx])
mtx.append([0, fy, cy])
mtx.append([0, 0, 1])

mtx = np.array(mtx)

k1=0.22688
k2=-2.17356
k3=1.26934
k4=0.112075
k5=-2.00597
k6=1.20173
p1=0.000842876
p2=0.0000975816
dist = []
dist.append([k1,k2,p1,p2,k3,k4,k5,k6])
dist = np.array(dist)

rvec , tvec, _ = aruco.estimatePoseSingleMarkers(corners, markerSizeInCM, mtx, dist)
#We now only need to read out z from the tvec which will be the distance from our camera to the marker center in the same measuring unit we provided in step 3.
print(f'distance marqueur/ camera : {tvec[0][0][2]} m')