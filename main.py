# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 14:16:23 2022

@author: enora
"""

import cv2
from cv2 import aruco
import  numpy as np

import pandas as pd
import plotly.express as px
from plotly.offline import plot

import os.path
from PIL import Image
import matplotlib.pyplot as plt

from evaluation import plot_eval_seq, evaluation


# Importation des images (RGB, Vérité terrain, midas)
image = cv2.imread("D:/Documents/Etudes/Esir/Esir_3/Projet indus/Ensemble_test/sequence_7/color025.png", cv2.IMREAD_UNCHANGED)
depth_map = cv2.imread("D:/Documents/Etudes/Esir/Esir_3/Projet indus/Ensemble_test/sequence_7/depth025.png", cv2.IMREAD_UNCHANGED)
midas = cv2.imread("D:\Documents\Etudes\Esir\Esir_3\Projet indus\Ensemble_test\MiDas\color025_MiDas.png",cv2.IMREAD_ANYDEPTH)

# Variables d'affichages pour ce code 
affiche_images_ = True # True or False si on veut afficher les images exemples
points3D = True; # True or False si on veut afficher nos points 3D

# __________________________________________________________________________________________________________________________
# _________________________________________ I. Capteurs aruco et calcul de pente ___________________________________________
# __________________________________________________________________________________________________________________________ 

# ------------------------------------------------------------------------------------------------
# ----------------------------------------- Fonctions --------------------------------------------
# ------------------------------------------------------------------------------------------------    
"""
Ouvre une image de la vérité terrain, permet, pour un point cliqué, de donner la distance qui sépare ce point de la caméra
"""
def click_event(event, x, y, flags, params):
 
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
 
        # displaying the coordinates
        # on the Shell
        pix_val = depth_map[y,x]
        print('')
        print(f'Coordonnées du point cliqué : [{x} , {y}]')
        print(f'Valeur du pixel cliqué : {depth_map[y,x]}')
        
        if (pix_val != 0):
            dist = pix_val * 0.001
            print(f'Ce point se trouve à une distance de {dist}m de la caméra')
            
        else:
            print ('Pas de valeur pour cette zone')

"""
Même chose que la fonction précédente mais cette fois on le fait pour l'image calaculer pas MiDas
"""
def click_event_midas(event, x, y, flags, params):
 
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
 

        pix_val = midas[y,x]
        pix_val = 65534-pix_val
        
        print(f'Coordonnées du point cliqué : [{x} , {y}]')
        print(f'Valeur du pixel cliqué : {midas[y,x]}')
        
        
        if (pix_val != 0):
            dist = (pix_val * S) - offset
            print(f'(pix_val * S) - offset = {pix_val} * {S} - {offset} = {dist}')
            dist = dist * 0.001
            print(f'ce point se trouve à une distance de {dist}m de la caméra')
            
        else:
            print ('Pas de valeur pour cette zone')
            
"""
Permet de calculer les valeurs de la pente (S et offset)
"""
def calcul_value_capteur (img, cap1_coor, cap2_coor):
    MAX = np.amax(img)
    #MIN = np.amin(img[np.where(img > 300)])
    #print(f"pour l'image, les min et la max sont : [{MIN},{MAX}]")
    
    pix_val_ar = img[cap1_coor[0],cap1_coor[1]]
    #print(f'- Valeur du grand capteur au fond :{pix_val_ar}')
    
    pix_val_av = img[cap2_coor[0],cap2_coor[1]]
    #print(f"- Valeur du petit capteur à l'avant :{pix_val_av}")
    
    return pix_val_ar, pix_val_av, MAX

def affiche_images():
        
        # Affichage de l'image avec la détection des marquers Aruco 
        cv2.imshow("aruco", frame_markers)
        cv2.waitKey(1) 


        # Outil de calcul de la distance dans l'image vérité terrain
        depth_img = depth_map.astype(np.uint8)
        cv2.imshow("depth image", depth_img)
        cv2.setMouseCallback('depth image', click_event)
        cv2.waitKey(0) 


        # Outil de calcul de la distance dans l'image rendu par Midas
        cv2.imshow("midas", midas)
        cv2.setMouseCallback('midas', click_event_midas)
        cv2.waitKey(0) 
        


def iterate_over_folder(aruco_pos, aruco_pos_GT):
    path_GT = "sequence_7_iter/GT"
    dirs_GT = os.listdir(path_GT)

    path_MiDas = "sequence_7_iter/MiDas"
    dirs_MiDas = os.listdir(path_MiDas)

    result = zip(dirs_MiDas, dirs_GT)
    result = list(result)
    
    S_list = []
    offset_list = []
    
    # Evaluation
    RMSE_list = []
    DM_list = []
    PS_list = []
    
    print("evaluation...")
    for i in range (len(result)):
         print(f"étape {i}/{len(result)}")
         MiDas_PIL = Image.open("sequence_7_iter/MiDas/"+result[i][0])
         GT_PIL = Image.open("sequence_7_iter/GT/"+result[i][1])
         if (i>5) : break
         
         MiDas = np.array(MiDas_PIL, np.int32)
         GT = np.array(GT_PIL, np.int32)
         
         # ----------------------------------------------------------------------------
         # Calcul de l'équation reliant la GT et MiDas --------------------------------
         # ----------------------------------------------------------------------------
         
         Z1, Z2, _ = calcul_value_capteur(GT,aruco_pos_GT[1],aruco_pos_GT[0])
         Z1_, Z2_, MAX = calcul_value_capteur(MiDas,aruco_pos[1],aruco_pos[0])

         # Pour inverser l'échelle de MiDas
         Z1_ = MAX - Z1_
         Z2_ = MAX - Z2_


         S = (Z1 - Z2)/(Z1_-Z2_)
         offset = ((S* Z1_) - Z1)
         offset_list.append(offset)
         S_list.append(S)
         
         # ----------------------------------------------------------------------------
         # Evaluation -----------------------------------------------------------------
         # ----------------------------------------------------------------------------
         seuil = 200
         path_fig = "D:/Documents/Etudes/Esir/Esir_3/Projet indus/Ensemble_test/Figures/"
         seq_name = "figs" + str(6) + "_"
         
         crop_MiDas = crop(MiDas_PIL)
         scale_MiDas = scale(crop_MiDas, S, offset)
         scale_MiDas = np.array(scale_MiDas, np.int32)
         #cv2.imshow("crop", scale_MiDas)
         #cv2.waitKey(0)
         
         RMSE, DM, PS = evaluation(scale_MiDas, GT, seuil) 
         RMSE_list.append(RMSE)
         DM_list.append(DM)
         PS_list.append(PS)
         
    plot_eval_seq(RMSE_list, DM_list, PS_list, seuil, path_fig, seq_name)
    
    plt.plot(S_list)
    plt.savefig(path_fig + seq_name + "S.png")
    
    plt.plot(offset_list)
    plt.savefig(path_fig + seq_name + "offset.png")
    
    print(f"Moyenne Ps = {sum(PS_list)/len(PS_list)}")
    print(f"Distance moyenne DM = {sum(DM_list)/len(DM_list)}")
    print(f"Moyenne RMSE = {sum(RMSE_list)/len(PS_list)}")
         
            
def crop (im):
    width = im.width
    height = im.height
    
    # Crop from 16:9 to 10:9
    imCrop = im.crop( ( width/5.33333, 0, width - (width/5.3333), height ) )

    # Now resize to (640x576)
    new_image = imCrop.resize((640, 576))
    return new_image
            
def scale(im, S, offset):
    new_image = im
    
    (largeur, hauteur)= im.size
    for x in range(largeur):
        for y in range(hauteur):
            # print(new_image.getpixel((x,y)))
            new_image.putpixel((x,y), int( ((65534 - im.getpixel((x,y))) * S) - offset))
    return new_image
   
    """
    #Application du scale
    (largeur, hauteur)= im.shape
    for y in range(largeur):
        for x in range(hauteur):
            # print(new_image.getpixel((x,y)))
            new_image[y,x], int ((MAX - im[y,x]) * S) - offset      
    return new_image
    """



# ------------------------------------------------------------------------------------------------
# ---------------------------------- Taille réel des marquers ------------------------------------ 
# ------------------------------------------------------------------------------------------------        
# Séquence 1 à 5
markerSizeInCM = 0.184          #en mètre

# Séquence 6 à 8
markersSizes = []
markersSizes.append(0.084)      #Petit marquer en mètre
markersSizes.append(0.178)      #Grand marquer en mètre

# ------------------------------------------------------------------------------------------------
# ----------------------------- Paramètres intrinsects de la caméra ------------------------------
# ------------------------------------------------------------------------------------------------      
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

# ------------------------------------------------------------------------------------------------
# ------------------------- Calcul des distances capteurs aruco/ caméra --------------------------
# ------------------------------------------------------------------------------------------------  

# Détection de l'emplacement des capteurs aruco
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters_create()
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
frame_markers = aruco.drawDetectedMarkers(image.copy(), corners, ids)

aruco_pos = []
aruco_pos_GT = []

for i in range(0, len(ids)):  # Iterate in markers
        # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
        rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], markersSizes[i], mtx, dist)
        print(f'Le grand {i} est à une distance de {tvec[0][0][2]} m avec la caméra')
        
        milieuX  = 0
        milieuY = 0
        
        # Position des capteur aruco
        for value in corners[i]:
            # print (f' value = {value}') # Coordoné des 4 coin du capteur
            
            # Fait la moyenne des 4 points afin de determiner le point du milieu du capteur
            for j in range(4) :
                milieuX += int(value[j][0])
                milieuY += int(value[j][1])
            milieuX /= 4
            milieuY /= 4
        aruco_pos.append([int(milieuY),int(milieuX)])
        
        print(f"Position du capteur dans l'image MiDas : {aruco_pos[i]} ")
        
        # Conversion des coordonnée image RGB en coordonnées image vérité terrain
        if (milieuX < 320 or milieuX > 960) :
               x_aruco_D = 0
        else :
               x_aruco_D =  milieuX-320
               
        y_aruco_D =  milieuY*0.75
        aruco_pos_GT.append([int(y_aruco_D), int(x_aruco_D)])
        
        #x_aruco_D = np.array(x_aruco_D)
        #y_aruco_D = np.array(y_aruco_D)
        #cv2.circle(depth_map,[int(x_aruco_D), int(y_aruco_D)],radius=5, color=(200, 200, 200), thickness=3)
        
        
        print(f'Dans la vérité terrain ça donnera les coordonées : {aruco_pos_GT[i]} ')
aruco_pos_GT = np.array(aruco_pos_GT)

# ------------------------------------------------------------------------------------------------
# ---------------------------- Calcul de la pente entre GT et MiDas ------------------------------
# ------------------------------------------------------------------------------------------------  

iterate_over_folder(aruco_pos, aruco_pos_GT)

Z1, Z2, _ = calcul_value_capteur(depth_map,aruco_pos_GT[1],aruco_pos_GT[0])
Z1_, Z2_, MAX = calcul_value_capteur(midas,aruco_pos[1],aruco_pos[0])

# Pour inverser l'échelle de MiDas
Z1_ = MAX - Z1_
Z2_ = MAX - Z2_

print(f'Z1 = {Z1}, Z2 = {Z2}')
print(f'Z1_ = {Z1_}, Z2_ = {Z2_}')

S = (Z1 - Z2)/(Z1_-Z2_)
offset = ((S* Z1_) - Z1)
offset2 = ((S* Z2_)-Z2)
print (f"Offset 1 :{offset}, Offset 2 :{offset2}")
print (f"la facteur S calculé nous donne : {S}")

# Z1 = Z1_ * S - Offset
# Z1_ = (Z1 + Offset)/S

if (affiche_images_):
    affiche_images()


# __________________________________________________________________________________________________________________________
# _____________________________________________ II. Scale et nuage de points _______________________________________________
# __________________________________________________________________________________________________________________________ 

new_image = midas

#Save the image file (uncomment last line too)
#f, e = os.path.splitext("depth025midas.png")


        
        
#Affichage des nuages de points ?
if(points3D):
    print("Chargement de l'image en point 3D...")
    #Pour afficher points 3D if 3Dpoints = True
    imgMid = new_image
    
    # MiDas points 3D
    X = []
    Y = []
    Z = []
    H = []
    
    cpt = 0
    resize_value = 14
    seuil_3D = -65535 #Indicatif pour l'affichage
    
    (largeur, hauteur)= imgMid.shape
    for y in range(largeur):
        for x in range(hauteur):
            cpt += 1
            if(cpt == resize_value):
                cpt = 0
                #on enlève les zéros
                if(imgMid[y, x] > 0 and imgMid[y, x]> 900  and imgMid[y, x]< 2700):
                    X.append(x)
                    Y.append(y)
                    Z.append(imgMid[y, x])
                    
                    if(imgMid[y, x] > seuil_3D):
                        H.append(1)
                    else:
                        H.append(0)
            
    
    dfMid = pd.DataFrame(dict(x=X, y=Y, z=Z, h=H))
    
    figMid = px.scatter_3d(dfMid, x='z', y='x', z='y', color = H)
    figMid.update_traces(marker_size = 2)
    plot(figMid)