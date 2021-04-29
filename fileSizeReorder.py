import os
import sys

OUTPUT_FOLDER_NAME = "Output"
TRAIN_FOLDER = "Bilder2.0"
MAX_FOLDER_NUMBER = 40
MAX_IMAGE_NUMBER = 10

def runBash(command):
    import subprocess
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output.decode("utf-8")
    
def copyFile(origin, destination, image):
    runBash("cp " + origin + " " + destination + image)

def getFilePath(extension_folder, rate, folder_number, image, isInput):
    if isInput:
        folder = "1-" + str(rate)
    else:
        folder = OUTPUT_FOLDER_NAME
        
    file_path = TRAIN_FOLDER + "/" + folder + "/" + extension_folder + "/s" + str(folder_number) + "/" + image
    return file_path
    
def getFileSize(extension_folder, extension, rate, folder_number, image):
    file_name = getFilePath(extension_folder, rate, folder_number, image, True)
    file_stats = os.stat(file_name)
    return file_stats.st_size
    
def initFolder(extension):
    runBash("rm -r " + TRAIN_FOLDER + "/" + OUTPUT_FOLDER_NAME)
    runBash("mkdir " + TRAIN_FOLDER + "/" + OUTPUT_FOLDER_NAME)
    runBash("mkdir " + TRAIN_FOLDER + "/" + OUTPUT_FOLDER_NAME + "/" + extension)
    for s_folder in range(1, MAX_FOLDER_NUMBER):
        file_path = getFilePath(extension_folder, None, s_folder, "", False)
        runBash("mkdir " + file_path)
        
def doIt(size, extension, extension_folder, folder_number, image_number):

    image = str(image_number) + "." + extension
    nearest_rate = 0
    nearest_size_diff = sys.maxsize
    
    for i in range(1, 11):
        file_size = getFileSize(extension_folder, extension, i, folder_number, image)
        file_size_diff = abs(file_size - size)
        
        if file_size_diff < nearest_size_diff:
            nearest_rate = i
            nearest_size_diff = file_size_diff
                        
    best_path = getFilePath(extension_folder, nearest_rate, folder_number, image, True)
    output_path = getFilePath(extension_folder, nearest_rate, folder_number, "", False)
    
    copyFile(best_path, output_path, image)
    
    print('nearest_rate:', nearest_rate)
    print('nearest_size_diff:', nearest_size_diff)

def run(size, extension, extension_folder):
    for s_folder in range(1,MAX_FOLDER_NUMBER):
        for image_number in range(MAX_IMAGE_NUMBER):
            doIt(size, extension, extension_folder, s_folder, image_number)

if len(sys.argv) < 4:
    print('Please enter the expected file size, folder and file extension')
else:
    size = int(sys.argv[1])
    extension_folder= str(sys.argv[2])
    extension = str(sys.argv[3])
    initFolder(extension_folder)
    run(size, extension, extension_folder)
