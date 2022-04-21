import os
import cv2

detected_fire = "prediction1/salil_diwali_3/detected_fire"
detected_non_fire = "prediction1/salil_diwali_3/detected_non_fire"
groundtruth_fire = "prediction1/salil_diwali_3/gt_fire"
groundtruth_non_fire = "prediction1/salil_diwali_3/gt_non_fire"

def accuracy(detected_fire,detected_non_fire,groundtruth_fire,groundtruth_non_fire):
    detected_fire_images = os.listdir(detected_fire)
    gt_fire_images = os.listdir(groundtruth_fire)
    tp = 0
    for i in detected_fire_images:
        if i in gt_fire_images:
            tp = tp + 1
    fp = len(detected_fire_images) - tp
    
    tn = 0
    fn = 0
    
    gt_non_fire_images = os.listdir(groundtruth_non_fire)
    detected_non_fire_images = os.listdir(detected_non_fire)
    for i in detected_non_fire_images:
        if i in gt_non_fire_images:
            tn = tn + 1
    
    fn = len(detected_non_fire_images) - tn
    
    print("detected_fire images: ",len(detected_fire_images))
    print("detected_non_fire_images: ",len(detected_non_fire_images))
    print("gt_fire_images: ",len(gt_fire_images))
    print("gt_non_fire_images ",len(gt_non_fire_images))
    print("tp :",tp)
    print("tn :",tn)
    print("fp :",fp)
    print("fn :",fn)
    
    precision = tp / len(detected_fire_images)
    print("Precision: ",precision)

    accuracy = (tp + tn)/(len(detected_fire_images)+len(detected_non_fire_images))
    print("Accuracy: ", accuracy)
    
    
    recall = tp/len(gt_fire_images)
    print("Recall: ",recall)
    

    f1 = (2*(precision*recall))/(precision+recall)
    print("F1 Score: ",f1)

accuracy(detected_fire,detected_non_fire,groundtruth_fire,groundtruth_non_fire)

