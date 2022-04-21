import cv2
import numpy as np 
import matplotlib.pyplot as plt
import os


#TODO: Changed the thresholds based on high accuracy
R_t = 190 
G_t = 100
B_t = 140
S_t = 60 
L1_t = 200 
L2_t = 255 
D1_t = 178 
D2_t = 200 

def R_component(r): 
    if(r > R_t):
        return 1
    else:
        return 0

def RGB_compare(b, g, r): # Comparing the red componenet values with green and blue
    if (r >= g > b) and (g>G_t) and (b<B_t):
        return 1
    else:
        return 0
    
def saturation(s, r):        
    lim = ((255.0-r)*S_t/R_t)
    if(s >= lim):
        return 1
    else:
        return 0

def is_fire_pixel(b, r, g, s):  # Chromatic analysis
    if(R_component(r) and RGB_compare(b, g, r) and saturation(s, r)):
        return 1
    else:
        return 0

def is_grey(b, r, g):
    if(abs(b-r)<=30 and abs(r-g)<=30 and abs(g-b)<=30):
        return 1
    else:
        return 0

def grey_intensity(v):      # Checking for the intensity level for smoke detection
    if(L1_t <= v <= L2_t or D1_t <= v <= D2_t):
        return 1
    else:
        return 0  

def is_smoke_pixel(b, r, g, v):     # Checking for smoke pixel
    if(is_grey(b, r, g) and grey_intensity(v)):
        return 1
    else:
        return 0

def fire_pixels_count(pixels_count,area_count):
    if pixels_count > 0.01 * area_count:
        print("pixels_count add")
        return True
    return False

def main_function(image, output_path, count,original_frame):     
    frame = cv2.imread(image)
    original_detection = cv2.imread(original_frame)
    one = np.zeros((frame.shape[0], frame.shape[1], frame.shape[2]))         
    one_cnt = 0
    two = np.ones((frame.shape[0], frame.shape[1], frame.shape[2]))
    two_cnt = 0
    x=-1 #x axis
    
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #FIXME: Single frame detection
    img_fp = np.zeros((frame.shape[0], frame.shape[1]),dtype=np.uint8) 
    img_sp = np.zeros((frame.shape[0], frame.shape[1]),dtype=np.uint8) 
    fire_pixel = np.zeros((frame.shape[0], frame.shape[1], frame.shape[2]))    
    smoke_pixel = np.zeros((frame.shape[0], frame.shape[1], frame.shape[2]))
    tre_cnt = 0
    # cv2.imwrite(output_path+"original/"+str(count)+".png",frame)
    for i in range(frame.shape[0]): #row iterate
        for j in range(frame.shape[1]): #col iterate
            bgr = frame[i][j].astype(np.float)
            hsv = img_hsv[i][j].astype(np.float)
            blue = bgr[0]
            green = bgr[1]
            red = bgr[2]
            satur = hsv[1]
            intens = hsv[2]

            #TODO: fire pixel detection
            if(is_fire_pixel(blue, red, green, satur)):
                img_fp[i][j] = 1    # Changing in Grayscale image
                fire_pixel[i][j] = bgr  # Storing the fire pixels
                tre_cnt += 1        # Count of Fire pixels
            
            if(is_smoke_pixel(blue, red, green, intens)):
                img_sp[i][j] = 1
                smoke_pixel[i][j] = bgr



    tre = fire_pixel
    tre_cnt = tre_cnt
    frame_area = frame.shape[0] * frame.shape[1]
    print("frame_area: ", frame_area)
    print("fire_area: ", tre_cnt)
    is_fire_count = fire_pixels_count(tre_cnt,frame_area)

    plt.scatter(x,tre_cnt,alpha=0.5,color='blue')   
    FD_t1 = np.absolute(np.subtract(tre, two))  
    
    FD_t = np.absolute(np.subtract(two, one))   
    FD = np.divide(np.absolute(FD_t1-FD_t), FD_t, out=np.zeros_like(np.abs(FD_t1-FD_t)), where=FD_t!=0)     # Checking with Fire disorder threshold value
    print(np.amax(FD))      
    per = float((FD>64.0).sum())/FD.size    
    print("FD>64.0sum : ", (FD>64.0).sum())
    print("FD size   : ", FD.size)
    print("percentage: ", per)
    if (per >= 0.00001) and (is_fire_count):        
        cv2.putText(original_detection,"FLAME DETECTED",(50,100),
            cv2.FONT_HERSHEY_PLAIN,2,(100,205,100),3)
        print("Real flame")
    else:  
        cv2.putText(original_detection,"NO FLAME",(50,100),
            cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),3)
        print("Fake flame")

    one = two
    two = tre
    cv2.imwrite(output_path+"detection/"+str(count)+" detection.png",original_detection)
    cv2.imwrite(output_path+"fire_pixels/"+str(count)+" fire.png",fire_pixel)
    cv2.imwrite(output_path+"smoke_pixels/"+str(count)+" smoke.png",smoke_pixel)
        
        
    '''
    cv2.imshow('frame', frame)      # Display video file
    cv2.imshow('img_fp', img_fp)    # Display Fire pixels
    cv2.imshow('img_sp', img_sp)    # Display smoke pixels
    '''
        # cv2.imshow("output"+"/"+str(vcount)+".jpg",fire_pixel)
        
        # k = cv2.waitKey(5) & 0xFF
        # if k == 27:
        #     break   
def main():
    input_folder = "../output"
    input_folder_list = os.listdir(input_folder)
    # print(input_folder_list)
    for folders in input_folder_list:
        image_path = input_folder + "/" + folders + "/and/"
        output_path = input_folder+ "/" + folders + "/"
        
        os.makedirs(output_path+"detection")
        # os.makedirs(output_path+"original")
        os.makedirs(output_path+"fire_pixels")
        os.makedirs(output_path+"smoke_pixels")
        image_list = os.listdir(image_path)
        # print(image_list)
        count = 0
        for image in image_list:
            path = image_path+image
            original_image_name = image[:-7] + "original.png"
            original_path = output_path +"/original/"+ original_image_name
            main_function(path,output_path,count,original_path)
            count = count + 1 
if  __name__ == "__main__":
    main()


