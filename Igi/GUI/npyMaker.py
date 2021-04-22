from PIL import Image
import os, sys
import cv2
import numpy as np
from cykooz.heif.pil import register_heif_opener

format = "AVIF"  # AVIF | JXR | JPEG2000 | JPG  | Original
# Path to image directory
path = "Bilder/Training/" + format + "/"
dirs = os.listdir(path)
dirs.sort()
x_train = []
x_target = []


def load_dataset():
    counter = 0
    # Append images to a list
    for subDirName in dirs:
        if subDirName == ".DS_Store":
            continue
        subDirPath = path + subDirName + "/"
        subDirs = os.listdir(subDirPath)
        subDirs.sort()
        targetAdded = False
        for item in subDirs:
            if item == ".DS_Store":
                continue
            itemPath = subDirPath + item
            if os.path.isfile(itemPath):

                register_heif_opener()
                im = Image.open(itemPath)

                left = 14
                top = 24
                right = 78
                bottom = 88

                im = im.crop((left, top, right, bottom))

                im = np.array(im)
                x_train.append(im)
                x_train.append(im)
                counter = counter + 1
                counter = counter + 1
                if counter == 400:
                    return
                if not targetAdded:
                    print(itemPath)
                    x_target.append(im)
                    targetAdded = True


if __name__ == "__main__":
    load_dataset()

    # Convert and save the list of images in '.npy' format
    imgset = np.array(x_train)
    np.save("imgds_" + format + ".npy", imgset)
    imgset = np.array(x_target)
    np.save("target_" + format + ".npy", imgset)
