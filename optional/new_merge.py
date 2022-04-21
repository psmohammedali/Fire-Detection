import os
import cv2

def hconcat_resize(img_list, interpolation = cv2.INTER_CUBIC):
      # take minimum hights
    h_min = min(img.shape[0] 
                for img in img_list)
      
    # image resizing 
    im_list_resize = [cv2.resize(img,
                       (int(img.shape[1] * h_min / img.shape[0]),
                        h_min), interpolation
                                 = interpolation) 
                      for img in img_list]
      
    # return final image
    return cv2.hconcat(im_list_resize)

input_path = "output/"
output_path = "new_merge/"
input_folder = os.listdir(input_path)
output_folder = os.listdir(output_path)
for folder in input_folder:
    if folder not in output_folder:
        os.mkdir(output_path+folder)

for folder in input_folder:
    detection = input_path+folder+"/"+"detection"+"/"
    and_path = input_path+folder+"/"+"and"+"/"
    length =  len(os.listdir(detection))
    
    for i in range(0,length):
        original_frame = cv2.imread(detection + str(i+1)+" detection.png")
        and_frame = cv2.imread(and_path+ str(i)+ " and.png")
        # cv2.imshow("original", cv2.imread(original_frame))
        # cv2.imshow("filter bg",cv2.imread(filter_frame))
        # cv2.imshow("aimodel ",cv2.imread(aimodel_frame))
        # cv2.imshow("and ", cv2.imread(and_frame))
        # img_h_resize = hconcat_resize([original_frame, and_frame])

        scale_percent = 50 # percent of original size
        width = int(original_frame.shape[1] * scale_percent / 100)
        height = int(original_frame.shape[0] * scale_percent / 100)
        dim = (width, height)
 
        # resized_original = cv2.resize(original_frame, dim, interpolation = cv2.INTER_AREA)
        # resized_and = cv2.resize(and_frame, dim, interpolation = cv2.INTER_AREA)
 
        # cv2.imshow("original",resized_original)
        # cv2.imshow("resized_filter",resized_filter)
        # cv2.imshow("resized_aimodel",resized_aimodel)
        # cv2.imshow("resized_and",resized_and)

        final = cv2.hconcat([original_frame, and_frame])
  
# show the output image
        # cv2.imshow('sea_image.jpg', final)
        # cv2.waitKey(0)
        cv2.imwrite(output_path+folder+"/"+str(i)+" final.png",final)
        print(i)
 
        



