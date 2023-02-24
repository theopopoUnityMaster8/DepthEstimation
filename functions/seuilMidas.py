import cv2
import numpy as np
import matplotlib.pyplot as plt

t = (576, 640)

#inverse midas
img = cv2.imread('/depth090.png', cv2.IMREAD_ANYDEPTH)

im1 = cv2.imread('./Evaluation/midas090.png', cv2.IMREAD_ANYDEPTH)

im_inv = np.copy(im1)

for i in range(t[0]):
    for j in range(t[1]):
        im_inv[i][j] = 65535 - im_inv[i][j]  


im2 = np.copy(im1)

for i in range(t[0]):
    for j in range(t[1]):
        if im2[i][j] < 45000:
            im2[i][j] = 0

cv2.imshow('Depth inv', im_inv)
cv2.imshow('MiDas', im1)
cv2.imshow('test', im2)
cv2.waitKey()

#Tcv2.imwrite('./Evaluation/midas_inv090.png', im_inv)

###################################################################

# Recolte données fusion


img = cv2.imread('./depth090.png', cv2.IMREAD_ANYDEPTH)
val_truth = []

im1 = cv2.imread('transform090.png', cv2.IMREAD_ANYDEPTH)
val_midas = []

for i in range(576):
    val_truth.append(img[i][320])
    val_midas.append(img[i][320])

#cémoche beurk
plt.scatter(val_truth, val_midas)
plt.show()
        
        