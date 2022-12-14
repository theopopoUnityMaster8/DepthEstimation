# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 10:38:51 2022

@author: pauma
"""

from plotly.offline import plot
from PIL import Image

# On sélectionne le renderer de notre graph
# io.renderers.default='browser' # pour afficher sur le navigateur

# Nos images de depth : Humain statique
imgGT = Image.open("depth100GT.png")
imgMid = Image.open("depth100midas.png")

# Nos images de depth : Humain bras vers l'avant
# imgGT = Image.open("depth191GT.png")
# imgMid = Image.open("depth191midas.png")


# Tout avec plotly.express
import plotly.express as px
import pandas as pd

#Créer un jeu de données 3D où chaque point a trois valeurs X,Y et Z
X = []
Y = []
Z = []
H = []

cpt = 0
resize_value = 13
seuil = 1850

(largeur, hauteur)= imgGT.size
for x in range(largeur):
    for y in range(hauteur):
        cpt += 1
        if(cpt == resize_value):
            cpt = 0
            #on enlève les zéros
            if(imgGT.getpixel((x, y)) != 0 and imgGT.getpixel((x, y)) < 3000):
                X.append(x)
                Y.append(y)
                Z.append(imgGT.getpixel((x, y)))
                
                if(imgGT.getpixel((x, y)) < seuil):
                    H.append(1)
                else:
                    H.append(0)
        
        
dfGT = pd.DataFrame(dict(x=X, y=Y, z=Z, h=H))

figGT = px.scatter_3d(dfGT, x='z', y='x', z='y', color = H)
figGT.update_traces(marker_size = 2)
plot(figGT)


# MiDas points 3D
X = []
Y = []
Z = []
H = []

cpt = 0
resize_value = 14
seuil = 57000

(largeur, hauteur)= imgMid.size
for x in range(largeur):
    for y in range(hauteur):
        cpt += 1
        if(cpt == resize_value):
            cpt = 0
            #on enlève les zéros
            if(imgMid.getpixel((x, y)) != 0):
                X.append(x)
                Y.append(y)
                Z.append(imgMid.getpixel((x, y)))
                
                if(imgMid.getpixel((x, y)) > seuil):
                    H.append(1)
                else:
                    H.append(0)
        

dfMid = pd.DataFrame(dict(x=X, y=Y, z=Z, h=H))

figMid = px.scatter_3d(dfMid, x='z', y='x', z='y', color = H)
figMid.update_traces(marker_size = 2)
plot(figMid)