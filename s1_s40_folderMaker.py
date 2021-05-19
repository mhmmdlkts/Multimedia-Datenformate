def runBash(command):
    import subprocess
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output.decode("utf-8")

def make0in(path, folder):
    path = path + "/" + folder
    extension = getExtensionInPath(path)
    oldImgPath = path + "/1." + extension
    newImgPath = path + "/0." + extension
    runBash("cp " + oldImgPath + " " + newImgPath)
    print("new file " + newImgPath)

for i in range(1,41):
    print(i)
    # make0in("Bilder/Training/JXR", "s" + str(i))
    
