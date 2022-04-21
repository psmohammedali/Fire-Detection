from __future__ import print_function

import logging
import time

import cv2
import numpy as np
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import os
import json


MODEL_WIDTH = 640
MODEL_HEIGHT = 640

use_ai_model = True

def _iou(box1, box2):
    """
    Computes Intersection over Union value for 2 bounding boxes

    :param box1: array of 4 values (top left and bottom right coords): [x0, y0, x1, x2]
    :param box2: same as box1
    :return: IoU
    """
    b1_x0, b1_y0, b1_x1, b1_y1 = box1
    b2_x0, b2_y0, b2_x1, b2_y1 = box2

    int_x0 = max(b1_x0, b2_x0)
    int_y0 = max(b1_y0, b2_y0)
    int_x1 = min(b1_x1, b2_x1)
    int_y1 = min(b1_y1, b2_y1)

    int_area = max(int_x1 - int_x0, 0) * max(int_y1 - int_y0, 0)

    b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
    b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)

    # we add small epsilon of 1e-05 to avoid division by 0
    iou = int_area / (b1_area + b2_area - int_area + 1e-05)
    return iou

def non_max_suppression(predictions_with_boxes, confidence_threshold=None, iou_threshold=0.4):
    """
    Applies Non-max suppression to prediction boxes.

    :param predictions_with_boxes: 3D numpy array, first 4 values in 3rd dimension are bbox attrs, 5th is confidence
    :param confidence_threshold: the threshold for deciding if prediction is valid
    :param iou_threshold: the threshold for deciding if two boxes overlap
    :return: dict: class -> [(box, score)]
    """
        
    result = {}
    for cam_id, image_pred in enumerate(predictions_with_boxes):
        result_per_image = {}

        if confidence_threshold is None:
            conf_threshold = 0.4
        else:
            conf_threshold = confidence_threshold[cam_id]   
        
        start = time.time()
        image_pred = image_pred[image_pred[:, 4] > conf_threshold]
        stop = time.time()

        bbox_attrs = image_pred[:, :5]
        classes = image_pred[:, 5:]
        classes = np.argmax(classes, axis=-1)

        unique_classes = list(set(classes.reshape(-1)))
        for cls in unique_classes:
            cls_mask = classes == cls
            cls_boxes = bbox_attrs[np.nonzero(cls_mask)]
            cls_boxes = cls_boxes[cls_boxes[:, -1].argsort()[::-1]]
            cls_scores = cls_boxes[:, -1]
            cls_boxes = cls_boxes[:, :-1]

            while len(cls_boxes) > 0:
                box = cls_boxes[0]
                score = cls_scores[0]
                if cls not in result_per_image:
                    result_per_image[cls] = []
                result_per_image[cls].append((box, score))
                cls_boxes = cls_boxes[1:]
                cls_scores = cls_scores[1:]
                ious = np.array([_iou(box, x) for x in cls_boxes])
                iou_mask = ious < iou_threshold
                cls_boxes = cls_boxes[np.nonzero(iou_mask)]
                cls_scores = cls_scores[np.nonzero(iou_mask)]
        result[cam_id] = result_per_image 
        # sts = time.time()
        # print(f"time taken for complete iteration : {sts-st}")
    return result

def letterbox_image(image, expected_size):
    ih, iw, _ = image.shape
    eh, ew = expected_size
    scale = min(eh / ih, ew / iw)
    nh = int(ih * scale)
    nw = int(iw * scale)

    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    new_img = np.full((eh, ew, 3), 128, dtype='uint8')
    # fill new image with the resized image and centered it
    new_img[(eh - nh) // 2:(eh - nh) // 2 + nh,
            (ew - nw) // 2:(ew - nw) // 2 + nw,
            :] = image.copy()
    return new_img

def convert_to_original_size(box, size, original_size, is_letter_box_image):
    if is_letter_box_image:
        box = box.reshape(2, 2)
        box[0, :] = letter_box_pos_to_original_pos(box[0, :], size, original_size)
        box[1, :] = letter_box_pos_to_original_pos(box[1, :], size, original_size)
    else:
        ratio = original_size / size
        box = box.reshape(2, 2) * ratio
    box = box.reshape(-1)
    return [int(box[0]), int(box[1]), int(box[2]), int(box[3])]

def letter_box_pos_to_original_pos(letter_pos, current_size, ori_image_size)-> np.ndarray:
    """
    Parameters should have same shape and dimension space. (Width, Height) or (Height, Width)
    :param letter_pos: The current position within letterbox image including fill value area.
    :param current_size: The size of whole image including fill value area.
    :param ori_image_size: The size of image before being letter boxed.
    :return:
    """
    letter_pos = np.asarray(letter_pos, dtype=np.float)
    current_size = np.asarray(current_size, dtype=np.float)
    ori_image_size = np.asarray(ori_image_size, dtype=np.float)
    final_ratio = min(current_size[0]/ori_image_size[0], current_size[1]/ori_image_size[1])
    pad = 0.5 * (current_size - final_ratio * ori_image_size)
    pad = pad.astype(np.int32)
    to_return_pos = (letter_pos - pad) / final_ratio
    return to_return_pos

class YoloV3AIModel():
    def __init__(self, ckpt_path, label_path, number_of_classes):
        """
        :param ckpt_path: path to the tensorflow frozen graph
        :param label_path: path to tensorflow labels
        :param number_of_classes: number of classes availabel

        """
        super().__init__()
        self.PATH_TO_CKPT = ckpt_path
        self.PATH_TO_LABELS = label_path
        self.NUM_CLASSES = number_of_classes

        with open(self.PATH_TO_LABELS, 'r') as f:
            self.labels_map = [x.strip() for x in f]

        if use_ai_model:
            self.detection_graph = tf.Graph()
            with self.detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.image_tensor = self.detection_graph.get_tensor_by_name('inputs:0')
            self.boxes_tensor = self.detection_graph.get_tensor_by_name('output_boxes:0')

    @staticmethod
    def __str__():
        return 'YoloV3AIModel'

    def predict(self, image_batch, thresholds=None, bbox_threshold=None):
        if not use_ai_model:
            size = len(image_batch)
            return [list() for _ in range(size)], [list() for _ in range(size)], [list() for _ in range(size)]

        image_batch_resized = []

        minx = min([i.shape[1] for i in image_batch])
        
        miny = min([i.shape[0] for i in image_batch])
        # print(f"length of image_batch")
        for img in image_batch:
            boxed_image = letterbox_image(img, (MODEL_WIDTH, MODEL_HEIGHT))
            boxed_image = cv2.cvtColor(boxed_image, cv2.COLOR_BGR2RGB)
            image_data = np.array(boxed_image, dtype='float32')
            # image_np_expanded = np.expand_dims(image_data, axis=0)
            image_batch_resized.append(image_data)
        try:

            detected_tensor = self.sess.run(
                self.boxes_tensor, feed_dict={self.image_tensor: image_batch_resized})
            print(f"detection shape {detected_tensor.shape}")

            filtered_boxes = non_max_suppression(detected_tensor, thresholds)


            all_detections = len(filtered_boxes)
            if all_detections == 0:
                size = len(image_batch)
                return [list() for _ in range(size)], [list() for _ in range(size)], [list() for _ in range(size)]

            all_classes = []
            all_bboxes = []
            all_scores = []
            # logger.debug(f'Started reading detection tensor')
            for cam_index, detections_per_image in filtered_boxes.items():
                classes_per_img = []
                bboxes_per_img = []
                scores_per_img = []
                for cls, bboxs in detections_per_image.items():
                    for box, score in bboxs:
                        # print(f"box : {box} , score : {score}")
                        # if score < thresholds[cam_index]:
                        #     break
                        box = convert_to_original_size(box, np.array((MODEL_HEIGHT, MODEL_WIDTH)),
                                            np.array((minx, miny)),
                                            is_letter_box_image=True)
                        bboxes_per_img.append(box)
                        scores_per_img.append(score)
                        classes_per_img.append(self.labels_map[cls])
                    # print(f"inside classes : {classes_per_img}")
                    # print(f"inside boxes : {bboxes_per_img}")
                    # print(f"inside scores : {scores_per_img}")

                all_classes.append(classes_per_img)
                all_bboxes.append(bboxes_per_img)
                all_scores.append(scores_per_img)
                
            return all_classes, all_bboxes, all_scores
        except RuntimeError as e:
            size = len(image_batch)
            return [list() for _ in range(size)], [list() for _ in range(size)], [list() for _ in range(size)]

def main():
    model_path = "./frozen_darknet_yolov3_model.pb"
    label_path = "./coco.names"
    number_of_classes = 15
    tf_model = YoloV3AIModel(model_path, label_path, number_of_classes)


    input_path = "../output/"
    folder_list = os.listdir(input_path)
    for folder in folder_list:
        images_path = input_path+folder+"/original/"
        images = os.listdir(images_path)
        info = { }
        info["yolov3"] = {}
        if "aimodel" not in os.listdir(input_path+folder):
            os.makedirs(input_path+folder+"/aimodel")
        person_output_path = input_path+folder+"/aimodel/"
        count = 0
        frame_list = []
        for image in images:
            frame_id = image[:-13]
            info["yolov3"][frame_id] = { }
            
            frame = cv2.imread(os.path.join(images_path, image))

            classes, boxes, scores = tf_model.predict([frame], [0.4])

            info["yolov3"][frame_id]["class"] = classes[0]    
            info["yolov3"][frame_id]["bbox"] = boxes[0]
            info["yolov3"][frame_id]["score"] = scores

            person = False
            index = 0
            for (x,y,w,h) in boxes[0] :
                class_name = classes[0][index]
                score = scores[0][index]
                # print(class_name)
                if class_name == "person"  :
                    person = True
                    colour = (0,0,0)
                    print(f'{frame_id}: {x, y, w, h}')
                    name = class_name + "_" + str(score)
                    frame = cv2.rectangle(frame,(x,y),(w,h),colour,-1)
                    index += 1
                
            
            cv2.imwrite(f'{person_output_path}/{frame_id} aimodel.png', frame)

            if person == True :
                count = count + 1
                cv2.imwrite(f'{person_output_path}/{frame_id} aimodel.png', frame)

if __name__ == "__main__" :
    main()
