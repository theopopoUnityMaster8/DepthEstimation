import numpy as np
import cv2 as cv
import os.path
import time

#############################################################################
# RECALAGE MANUEL ENTRE GROUND TRUTH & MIDAS (TRANSLATION) 
# VISUALISATION SUR UNE SEQUENCE VIDEO (FUSION Truth MiDas)
#############################################################################

#############################################################################
# SECTION image path / name     MODIFICATION NECESSAIRE          
path_midas = "../seq_midas/" #"output/"
dirs_midas = os.listdir(path_midas)

path_truth = "../seq_truth/" #"output/"
dirs_truth = os.listdir(path_truth)

n = len(dirs_midas)
#############################################################################

Pause = False

t = (640, 576)
    
#Prepare video
#fourcc = cv.VideoWriter_fourcc(*'XVID')
#video = cv.VideoWriter('fusion.avi', fourcc, 20.0, (576,  640))



for n_item in range(n):
    
    # Préparation des données
    truth = cv.imread(path_truth + dirs_truth[n_item], cv.IMREAD_ANYDEPTH)
    midas = cv.imread(path_midas + dirs_midas[n_item], cv.IMREAD_ANYDEPTH)


    #Recalage translation + rotation

    imT = np.copy(midas)

    M = np.array([[1, 0, 30],[0, 1, -10]], dtype = float)   #Matrice de transforamtion
    #t = np.shape(im2)
    #t = (640, 576)

    imT = cv.warpAffine(imT, M, t)      #Application de la transformation à l'image


    # AFFICHAGE REACALGE
    img = truth.copy()
    for i in range(280):
        pair = 2*i
        impair = 2*i + 1
        img[pair] = truth[pair]
        img[impair] = imT[impair]

    #cv.imwrite('./Evaluation/transform090.png', imT)
    # cv.imshow("truth", truth)
    # cv.imshow("transform", midas)
    #video.write(np.flip(img,0))
    
    #img = img[100:400, 310:330]
    cv.imshow("fusion", midas)
    
    k = cv.waitKey(25)
    if k & 0xFF == ord('q'):
        break 
    
    if (k == 32) & (Pause == False) :
        Pause = True
        
        
    while Pause:
        time.sleep(0.1)
        if cv.waitKey(25) == 32 :
            Pause = False


cv.destroyAllWindows()
#video.release()

              
