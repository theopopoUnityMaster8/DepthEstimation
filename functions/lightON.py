#Script de test
import cv2


#Transform depth image
#Rendre visible les cartes fournies

#############################################################################
# SECTION image path / name         MODIFICATION NECESSAIRE    
file_path = './Evaluation/'         
input_name = 'depth090.png'         
output_name = 'upgrade090.png'      
#############################################################################"

#Lire en uint16
img = cv2.imread(str(file_path + input_name), cv2.IMREAD_ANYDEPTH)

t = img.shape               #taille de l'image
output_img = img.copy()     #copie de l'image pour amélioration

# Traitement
img_list = img.flatten().tolist()       #image converti en vecteur
val = [*set(img_list)]                  #liste des valeurs uniques du vecteur

# return coef de la fonction affine appliquée à l'image 
def get_coef(val):
    max_val = max(val)
    coef = 25
    while coef*max_val >= 65535:        #val max uint16 65535
        coef -= 1
    return coef


coef = get_coef(val)        #récupération du coef
print('coef =', coef)

# Transormation de l'image
for i in range(t[0]):
    for j in range(t[1]):
        output_img[i,j] = img[i,j]*coef     #fonction affine : x -> coef * x
        
# Affichage et enregistrement
cv2.imshow('UPGRADE', output_img)
cv2.imwrite(str(file_path + output_name), output_img)

cv2.waitKey()
cv2.destroyAllWindows()