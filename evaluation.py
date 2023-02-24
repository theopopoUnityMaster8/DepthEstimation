import numpy as np
import matplotlib.pyplot as plt
import math


##############################################################################
##      EVALUATION
##############################################################################

# Ce script sert à l'évaluation des prédictions de MiDas

# On calcul 3 métriques de corrélation :
#   RMSE : sqrt( 1/N * Somme((pred - truth)²) )
#   DM : Distance moyenne entre les images pred & truth
#   PS : proportion de bonne prédiction d'après un seuil prédéfini


# Fonction d'évaluation d'une image (pred & truth)
#   pred : image prédite (MiDas)
#   truth : ground truth
#   seuil : seuil pour les bonnes prédictions
#   return : RMSE, DN, PS(%)

def evaluation(pred, truth, seuil):

    # conversion PIL to array
    #pred_arr = np.array(pred, dtype=np.uint16)
    #truth_arr = np.array(truth, dtype=np.uint16)
    pred_arr = pred
    truth_arr = truth
    
    # préparation des données
    pred_flat = pred_arr.flatten()
    truth_flat = truth_arr.flatten()

    # nombre de pixels
    N = len(pred_flat)

    # Initialisation des métriques
    RMSE = 0
    DM = 0
    PS = 0  
    cpt= 0
    
    # parcours des images
    for k in range(N):
        # suppression des zones mortes (contours, bords de l'image)
        
        if truth_flat[k] != 0 and pred_flat[k]> 900  and pred_flat[k]< 2700:
            distance = abs(pred_flat[k] - truth_flat[k])
            # calcul métrique
            RMSE += math.sqrt(abs(math.pow(int((pred_flat[k]*255)/65535),2) - math.pow(int((truth_flat[k]*255)/65535),2)))
            DM += abs(((distance*255)/65535))
            cpt = cpt +1
            if abs(distance) < seuil:
                PS += 1  #bonne prédiction 
    RMSE = np.sqrt(RMSE/cpt)
    DM = DM/cpt
    PS = PS/cpt * 100

    return RMSE, DM, PS
    
#############################################################################
##      AFFICHAGE RESULTAT
#############################################################################

# Fonction d'affichage des résultats sur une séquence
#   RMSE : liste des RMSE par image d'une séquence
#   DM : liste des distances moyennes par image d'une séquence
#   PS : proportion de bonne prédiction par image d'une séquence
#   seuil : seuil de bonne prédiction
#   path_fig : dossier de sauvegarde des figures
#   seq_name : nom de la séquence
#   return : proportion de bonne prédiction de la séquence 

def plot_eval_seq(RMSE, DM, PS, seuil, path_fig, seq_name):
    
    # plot RMSE
    plt.figure(1)
    plt.title("courbe RMSE")
    plt.plot(RMSE, lw=2, color='navy', label='RMSE curve')
    plt.xlabel('n°image sequence')
    plt.ylabel('RMSE')
    plt.savefig(path_fig + seq_name + "rmse.png")
    #plt.show()

    # plot DM
    plt.clf()
    plt.title("courbe distance moyenne")
    plt.plot(DM, lw=2, color='red', label='Distance curve')
    plt.xlabel('n°image sequence')
    plt.ylabel('Distance moyenne')
    plt.savefig(path_fig + seq_name + "dm.png")
    #plt.show()  

    # plot PS
    plt.clf()
    plt.title("courbe précision")
    plt.plot(PS, lw=2, color='green', label='Distance curve')
    plt.xlabel('n°image sequence')
    plt.ylabel('précision, distance < ' + str(seuil/10) + ' cm')
    plt.savefig(path_fig + seq_name + "ps.png")
    #plt.show()  

    return sum(PS)/len(PS)