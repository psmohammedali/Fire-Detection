import cv2
import os
import numpy as np
import glob
def main(path, videoname):
    img_array = []
    for filename in glob.glob(path):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    
    out = cv2.VideoWriter("videos4/"+videoname+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

output = "output3/"
output_folders= os.listdir(output)
for folder in output_folders:
    path = output+ folder + "/detection/*.png"
    print(path)
    videoname = folder
    main(path,videoname)
    print(folder)