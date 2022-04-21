import numpy as np
import cv2
import os
import time


def main():
    file = open("frame_report.txt", "w+")
    total_frames = 0
    total_time = 0
    path = "../video/"
    output_path = "../output/"
    videos_list = os.listdir(path)
    output_videos_list = os.listdir(output_path)
    for i in videos_list:
        folder_name = i[:-4]
        if folder_name not in output_videos_list:
            os.makedirs(output_path+folder_name)
            folders = ["original","mog2","filter"]
            for f in folders:
                os.makedirs(output_path+folder_name+"/"+f)
        
        standard_sub = cv2.createBackgroundSubtractorMOG2()
        count = 1
        cap = cv2.VideoCapture(path+i)
        while(1):
            ret, img = cap.read()
            if ret == True:
                if count == 30:
                    break
#TODO: Start TIME
                total_frames = total_frames + 1
                start = time.time()
                fgmask = standard_sub.apply(img)
                blur = cv2.GaussianBlur(fgmask,(7,7),0)
                thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)[1]
                end = time.time()
                effective_time = end - start
                total_time = total_time + effective_time
#TODO: END TIME
                cv2.imwrite(output_path+folder_name+'/original/'+str(count)+" original.png", img)
                cv2.imwrite(output_path+folder_name+'/mog2/'+str(count)+" mog.png", fgmask)
                cv2.imwrite(output_path+folder_name+'/filter/'+str(count)+" filter.png", thresh)
                
                count = count + 1
                print(count)
                file.write("Video Name : {} \n".format(folder_name))
                file.write("Current Video Frame: {} \n".format(count))
                file.write("Frame Effective Time: {} \n".format(effective_time))
                file.write("-------\n")

                print("Total Frames: ", total_frames)
                print("Total Time: ",total_time)
            if ret == False:
                break
        print(folder_name)
        cap.release()
        cv2.destroyAllWindows()
    return[total_frames, total_time]
    

if __name__ == "__main__":
    ans = main()
    my_file = open("report.txt", "w+")
    my_file.write("Total Frames: {}".format(ans[0]))
    my_file.write("Total Time Taken : {}".format(ans[1]))
    my_file.close()
