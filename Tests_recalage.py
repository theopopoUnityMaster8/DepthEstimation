import numpy as np
import cv2 as cv

#############################################################################
# RECHERCHE DE CORRESPONDANCE ENTRE GROUND TRUTH & MIDAS
#############################################################################

#############################################################################
# SECTION image path / name     MODIFICATION NECESSAIRE          
file_path = './Tests/'
truth_name = 'upgrade090.png' #'depth090.png'       
midas_name = 'midas090.png'         
output_name = 'fusion090.png'     
#############################################################################"

# Préparation des données
truth = cv.imread(str(file_path + truth_name), cv.IMREAD_ANYDEPTH)
midas = cv.imread(str(file_path + midas_name), cv.IMREAD_ANYDEPTH)

# Paramètres caméra 
fx=612.069
fy=612.134

cx=637.802
cy=367.899

k1=0.22688
k2=-2.17356
k3=1.26934
k4=0.112075
k5=-2.00597
k6=1.20173

K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1],], dtype = np.float32)     #Matrice intrinseque

D = np.array([[k1], [k2], [k3], [k4], [k5],], dtype = np.float32)                  #Paramètres de distorsion


# imT = cv.undistort(midas, K, D)

# cv.imshow('undistored', midas)
# cv.imshow("Disto", imT)
# cv.waitKey()
             
#Recalage translation + rotation

imT = np.copy(midas)

M = np.array([[1, 0, 30],[0, 1, -10]], dtype = float)   #Matrice de transforamtion
#t = np.shape(im2)
t = (640, 576)

imT = cv.warpAffine(imT, M, t)      #Application de la transformation à l'image


# AFFICHAGE REACALGE
img = truth.copy()
for i in range(280):
    pair = 2*i
    impair = 2*i + 1
    img[pair] = truth[pair]
    img[impair] = imT[impair]

cv.imwrite('./Evaluation/transform090.png', imT)
# cv.imshow("truth", truth)
# cv.imshow("transform", midas)

#img = img[100:400, 310:330]
cv.imshow("fusion", img)
cv.waitKey()