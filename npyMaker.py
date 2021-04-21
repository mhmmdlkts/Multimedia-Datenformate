from PIL import Image
import os, sys
import cv2
import numpy as np
import pyheif

format = "AVIF" # AVIF | JXR | JPEG2000 | JPG  | Original
# Path to image directory
path = "Bilder/Training/" + format + "/"
dirs = os.listdir( path )
dirs.sort()
x_train=[]

def load_dataset():
    # Append images to a list
    for subDirName in dirs:
        if subDirName == ".DS_Store":
            continue
        subDirPath = path+subDirName+"/"
        subDirs = os.listdir( subDirPath )
        subDirs.sort()
        for item in subDirs:
            if item == ".DS_Store":
                continue
            itemPath = subDirPath + item
            print (itemPath)
            if os.path.isfile(itemPath):
                im = Image.open(itemPath).convert("RGB")
                im = np.array(im)
                x_train.append(im)

if __name__ == "__main__":
    
    load_dataset()
    
    # Convert and save the list of images in '.npy' format
    imgset=np.array(x_train)
    np.save("imgds_" + format +  ".npy",imgset)
