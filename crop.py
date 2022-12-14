# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 15:04:28 2022

@author: pauma
"""

# Crop images from 16:9 to 10:9 (640x576)

#% From : https://stackoverflow.com/questions/47785918/python-pil-crop-all-images-in-a-folder


from PIL import Image
import os.path

path = "output/"
dirs = os.listdir(path)

def crop_folder():
    for item in dirs:
        fullpath = os.path.join(path,item)
        if os.path.isfile(fullpath) and fullpath.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            im = Image.open(fullpath)
            
            width = im.width
            height = im.height
            
            # Crop from 16:9 to 10:9
            f, e = os.path.splitext(fullpath)
            imCrop = im.crop( ( width/5.33333, 0, width - (width/5.3333), height ) )

            # Now resize to (640x576)
            new_image = imCrop.resize((640, 576))
            new_image.save(f + '.png', "PNG", quality=100)
            


crop_folder()