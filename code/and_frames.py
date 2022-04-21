import cv2
import os
def main():
    input_path = "../output/"
    folder_list = os.listdir(input_path)
    for folder in folder_list:
        # print(folder)
        aimodel_folder = input_path+folder+"/aimodel/"
        bg_folder = input_path+folder+"/filter/"
        
        if "and" not in os.listdir(input_path+folder):
            os.makedirs(input_path+folder+"/and")
            
        aimodel_images = os.listdir(aimodel_folder)
        bg_images = os.listdir(bg_folder)
        

        length = len(aimodel_images)
        for i in range(1,length+1):
            aimodel_name = str(i)+" aimodel.png"
            bg_name = str(i)+ " filter.png"
            img1 = cv2.imread(aimodel_folder+aimodel_name)
            img2 = cv2.imread(bg_folder+bg_name)
            print(aimodel_folder+aimodel_name)
            print(bg_folder+bg_name)
            bitwiseAnd = cv2.bitwise_and(img1, img2)
            # cv2.imshow("jdlkf",bitwiseAnd)
            # cv2.waitKey(0)
            cv2.imwrite(input_path+folder+"/and/"+str(i)+" and.png", bitwiseAnd)
            # print(input_path+folder+"/out/"+str(i)+" and.png")
            # print("-----")
            print(i)
        
        
# img1 = cv2.imread("output/salil_diwali_1/aimodel/368 original.jpg")
# img2 = cv2.imread("output/salil_diwali_1/filter/368 filter.png")
# bitwiseAnd = cv2.bitwise_and(img1, img2)

# # cv2.imwrite(input_path+folder+"/out/"+str(i)+" and.png", bitwiseAnd)
# cv2.imshow("jdlkf",bitwiseAnd)
# cv2.imshow("aimodel",img1)
# cv2.imshow("filter",img2)
# cv2.waitKey(0)
if __name__ == "__main__":
    main()