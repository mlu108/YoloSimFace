# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
from PIL import Image, ImageDraw, ImageFont
import os
import platform
import sys
from pathlib import Path
import shutil
from collections import Counter
import pdb
import pickle
from ultralytics import YOLO
import time

import torch
from deepface import DeepFace
from ultralytics.utils import LOGGER, TryExcept, ops, plt_settings, threaded

RECUR_DEPTH = 900
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def changed_save_one_box(xyxy, im, file=Path('im.jpg'), gain=1.02, pad=10, square=False, BGR=False, save=True):
    """Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop.

    This function takes a bounding box and an image, and then saves a cropped portion of the image according
    to the bounding box. Optionally, the crop can be squared, and the function allows for gain and padding
    adjustments to the bounding box.

    Args:
        xyxy (torch.Tensor or list): A tensor or list representing the bounding box in xyxy format.
        im (numpy.ndarray): The input image.
        file (Path, optional): The path where the cropped image will be saved. Defaults to 'im.jpg'.
        gain (float, optional): A multiplicative factor to increase the size of the bounding box. Defaults to 1.02.
        pad (int, optional): The number of pixels to add to the width and height of the bounding box. Defaults to 10.
        square (bool, optional): If True, the bounding box will be transformed into a square. Defaults to False.
        BGR (bool, optional): If True, the image will be saved in BGR format, otherwise in RGB. Defaults to False.
        save (bool, optional): If True, the cropped image will be saved to disk. Defaults to True.

    Returns:
        (numpy.ndarray): The cropped image.

    Example:
        ```python
        from ultralytics.utils.plotting import save_one_box

        xyxy = [50, 50, 150, 150]
        im = cv2.imread('image.jpg')
        cropped_im = save_one_box(xyxy, im, file='cropped.jpg', square=True)
        ```
    """

    if not isinstance(xyxy, torch.Tensor):  # may be list
        xyxy = torch.stack(xyxy)
    b = ops.xyxy2xywh(xyxy.view(-1, 4))  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = ops.xywh2xyxy(b).long()
    ops.clip_boxes(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
    if save:
        file.parent.mkdir(parents=True, exist_ok=True)  # make directory
        #f = str(increment_path(file).with_suffix('.jpg'))
        #Image.fromarray(crop[..., ::-1]).save(f, quality=95, subsampling=0)  # save RGB
        Image.fromarray(crop[..., ::-1]).save(file, quality=95, subsampling=0)  # save RGB
    return crop

def markTwoSameIdentities(face_counter,identity,compareSize,compareDistance,opposite_identities,frames_bbox_location,frame_num,pkl_dict,annotator,color_dict,i):
    score_list = []
    box_size_list = []
    
    for tuple in face_counter[identity]:
        face_xyxy = tuple[0]
        emotion_dict = tuple[1]
        conf = float(tuple[2])
        if not isinstance(face_xyxy, torch.Tensor):  # may be list
            xyxy = torch.stack(face_xyxy)
            b = ops.xyxy2xywh(xyxy.view(-1, 4))  # boxes
            xyxy = ops.xywh2xyxy(b).long() #(x1,y1,x2,y2)
        if (compareSize):
            box_size = abs((xyxy[:, 0] - xyxy[:, 2])*(xyxy[:, 1] - xyxy[:, 3]))
            box_size_list.append((box_size,face_xyxy,emotion_dict,conf))
        if (compareDistance):
            last_frame_xyxy = last_frame(identity, frames_bbox_location)
            if not last_frame_xyxy == None:
                distance = distance_bbox(xyxy, last_frame_xyxy)
                score = distance #smaller, the better
            else: ###!!!NEED TO CHANGE, delete this else as this condition does not exist any longer
                score = 1-conf #if it is the first frame and two mothers were identified, then  
            score_list.append((score,xyxy,emotion_dict,conf))
    #find the cloest_distance or the biggest face
    if (compareDistance):
        min_tuple = min(score_list, key=lambda x: x[0]) #min by distance
        output_xyxy = min_tuple[1]
        output_emotion_dict = min_tuple[2]
        selected_index = score_list.index(min_tuple)
        oppo = opposite_identities[identity]
    if (compareSize):
        biggest_face = max(box_size_list, key=lambda x: x[0])
        output_emotion_dict = biggest_face[2]
        score_list = box_size_list
        output_xyxy = biggest_face[1]
        selected_index = box_size_list.index(biggest_face)
        oppo = "other"   
    outputsFace(frame_num,f"{frame_num} REAL {identity}",identity,output_emotion_dict,pkl_dict,output_xyxy,frames_bbox_location,annotator,color_dict)
    index = 0 if selected_index == 1 else 1
    oppo_xyxy = score_list[index][1]
    oppo_demo = score_list[i][2]
    outputsFace(frame_num,f"{frame_num} RESET {oppo}",oppo,oppo_demo,pkl_dict,oppo_xyxy,frames_bbox_location,annotator,color_dict)

def chooseOne(face_counter,identity,compareSize,compareDistance,opposite_identities,frames_bbox_location,frame_num,pkl_dict,annotator,color_dict,i):
    score_list = []
    box_size_list = []
    
    for tuple in face_counter[identity]:
        face_xyxy = tuple[0]
        emotion_dict = tuple[1]
        conf = float(tuple[2])
        if not isinstance(face_xyxy, torch.Tensor):  # may be list
            xyxy = torch.stack(face_xyxy)
            b = ops.xyxy2xywh(xyxy.view(-1, 4))  # boxes
            xyxy = ops.xywh2xyxy(b).long() #(x1,y1,x2,y2)
        last_frame_xyxy = last_frame(identity, frames_bbox_location)
        if not last_frame_xyxy == None:
            distance = distance_bbox(xyxy, last_frame_xyxy)
            score = distance #smaller, the better
        else: ###!!!NEED TO CHANGE, delete this else as this condition does not exist any longer
            score = 1-conf #if it is the first frame and two mothers were identified, then  
        score_list.append((score,xyxy,emotion_dict,conf))
    min_tuple = min(score_list, key=lambda x: x[0]) #min by distance
    output_xyxy = min_tuple[1]
    output_emotion_dict = min_tuple[2]
    selected_index = score_list.index(min_tuple)
    oppo = opposite_identities[identity]
    #outputsFace(frame_num,f"CHOOSE {identity}",identity,output_emotion_dict,pkl_dict,output_xyxy,frames_bbox_location,annotator,color_dict)
    outputsFace(frame_num,f"{frame_num} CHOOSE {identity}",identity,output_emotion_dict,pkl_dict,output_xyxy,frames_bbox_location,annotator,color_dict)
  
    #with open('ProcessingResults/p24_debug.txt', 'a') as log_file:
   #    log_file.write(f"chooseOne at {frame_num}: {[score[0] for score in score_list]},{[score[1] for score in score_list]},{selected_index},\n")
    #return output_xyxy

def new_demo_xyxy(identity,face_counter):
    #returns the new face_xyxy and demography list
    face_tuple = face_counter[identity][0]
    face_xyxy = face_tuple[0]
    emotion_dict = face_tuple[1]
    return emotion_dict, face_xyxy

def isFace(last_frame_xyxy,save_dir,identity,imc,backends):
    
    expanded_last_xyxy = [torch.tensor(max(last_frame_xyxy[:, 0]-20,0)),
                        torch.tensor(max(last_frame_xyxy[:, 1]-20,0)),
                        torch.tensor(max(last_frame_xyxy[:, 2]+20,0)),
                        torch.tensor(max(last_frame_xyxy[:, 3]+20,0))]
    expanded_last_xyxy = torch.tensor(expanded_last_xyxy)
    output_path = save_dir / 'crops' / identity / 'manual_cropped.jpg'
    changed_save_one_box(expanded_last_xyxy, imc, file=output_path, BGR=True)

    #running deepface analysis on the cropped image
    try:
        
        demography = DeepFace.analyze(str(output_path), detector_backend = backends[3])
       
        return demography
    

    except ValueError as e:
        return None

def outputsFace(frame_num,text,identity,demography,pkl_dict,last_frame_xyxy,frames_bbox_location,annotator,color_dict):

    emotion_dict = demography
    if isinstance(demography, dict):
        emotion_dict = demography
    else:
        emotion_dict = demography[0]['emotion']
    
    #[{'emotion': {'angry': 4.6724050765024414e-15, 'disgust': 97.08793741988762, 'fear': 2.9120627663768985, 'happy': 1.6761929028481028e-27, 'sad': 9.362734868434089e-17, 'surprise': 1.3635123657850825e-31, 'neutral': 6.076284306665033e-35}, 'dominant_emotion': 'disgust', 'region': {'x': 67, 'y': 25, 'w': 88, 'h': 110}, 'age': 35,
    frames_bbox_location[-1][identity] = last_frame_xyxy
    max_emo = max(emotion_dict, key=lambda key: emotion_dict[key])
    if not (isinstance(last_frame_xyxy, list) and len(last_frame_xyxy) == 4):
        last_frame_xyxy = last_frame_xyxy[0]
    annotator.box_label(last_frame_xyxy, f"{text}: {max_emo}", colors(color_dict[identity], True)) #Add one xyxy box to image with label.
    pkl_dict = input_pkl(frame_num,pkl_dict,identity,emotion_dict)

def overlap_IOU(iou_value):
    if iou_value > 0.7:
        return True
    else:
        return False
def overlap_special_IOU(iou_value):
    if iou_value > 0.7:
        return True
    else:
        return False

def IOU(boxA, boxB,model):
    if boxA == None or boxB == None:
        return 0
    else:
        example_xyxy = torch.tensor([[284., 102., 408., 260.]])
        list = [boxA, boxB]
        for i in range(len(list)):
            xyxy = list[i]
            if not isinstance(xyxy, torch.Tensor) or xyxy.size() != example_xyxy.size():
                xyxy = [value.item() for value in xyxy]
                xyxy_new = torch.tensor([xyxy]) 
                list[i] = xyxy_new
        boxA, boxB = list[0], list[1]
        boxA = boxA.to(model.device)
        boxB = boxB.to(model.device)
        # Extract coordinates from boxes
        x1A, y1A, x2A, y2A = boxA[:, 0], boxA[:, 1], boxA[:, 2], boxA[:, 3]
        x1B, y1B, x2B, y2B = boxB[:, 0], boxB[:, 1], boxB[:, 2], boxB[:, 3]
        # Calculate the intersection coordinates
        x1_intersection = max(x1A, x1B)
        y1_intersection = max(y1A, y1B)
        x2_intersection = min(x2A, x2B)
        y2_intersection = min(y2A, y2B)
        # Calculate the area of intersection
        intersection_area = max(0, x2_intersection - x1_intersection + 1) * max(0, y2_intersection - y1_intersection + 1)
        # Calculate the area of both bounding boxes
        boxA_area = (x2A - x1A + 1) * (y2A - y1A + 1)
        boxB_area = (x2B - x1B + 1) * (y2B - y1B + 1)
        # Calculate the Union (sum of areas - intersection)
        union = boxA_area + boxB_area - intersection_area
        # Calculate the IOU
        iou = intersection_area / union
        return iou

def special_IOU(boxA, boxB,model):
    if boxA == None or boxB == None:
        return 0
    else:
        example_xyxy = torch.tensor([[284., 102., 408., 260.]])
        list = [boxA, boxB]
        for i in range(len(list)):
            xyxy = list[i]
            if not isinstance(xyxy, torch.Tensor) or xyxy.size() != example_xyxy.size():
                xyxy = [value.item() for value in xyxy]
                xyxy_new = torch.tensor([xyxy]) 
                list[i] = xyxy_new
        boxA, boxB = list[0], list[1]
        boxA = boxA.to(model.device)
        boxB = boxB.to(model.device)
        # Extract coordinates from boxes
        x1A, y1A, x2A, y2A = boxA[:, 0], boxA[:, 1], boxA[:, 2], boxA[:, 3]
        x1B, y1B, x2B, y2B = boxB[:, 0], boxB[:, 1], boxB[:, 2], boxB[:, 3]
        # Calculate the intersection coordinates
        x1_intersection = max(x1A, x1B)
        y1_intersection = max(y1A, y1B)
        x2_intersection = min(x2A, x2B)
        y2_intersection = min(y2A, y2B)
        # Calculate the area of intersection
        intersection_area = max(0, x2_intersection - x1_intersection + 1) * max(0, y2_intersection - y1_intersection + 1)
        # Calculate the area of both bounding boxes
        boxA_area = (x2A - x1A + 1) * (y2A - y1A + 1)
        boxB_area = (x2B - x1B + 1) * (y2B - y1B + 1)
        smaller_area = min(boxA_area,boxB_area)
        special_iou = intersection_area / smaller_area
        return special_iou


def orig_last_frame(label, frames_bbox_location): #[{frame1},{frame2}]
    #tensor([[164,   6, 289, 128]])
    #[[tensor(262.), tensor(89.), tensor(415.), tensor(261.)]]
    if len(frames_bbox_location)>=1:
        last_dict = frames_bbox_location[-1] #is a dictionary
        #[tensor(262.), tensor(89.), tensor(415.), tensor(261.)]
        if len(last_dict[label])>0:
            result_tensor = torch.cat([tensor.unsqueeze(0) for tensor in last_dict[label]], dim=0)
            return result_tensor.view(1, -1) #an array
        else:
            return last_frame(label,frames_bbox_location[:-1])
    else:
        return None


def last_frame(label, frames_bbox_location, recursion_depth=RECUR_DEPTH):
    # Base case: Stop recursion when recursion_depth is zero
    if recursion_depth == 0:
        return None

    if len(frames_bbox_location) >= 1:
        last_dict = frames_bbox_location[-1]  # Get the last dictionary
        # Check if label exists in the last dictionary and has non-empty values
        if label in last_dict and len(last_dict[label]) > 0:
            result_tensor = torch.cat([tensor.unsqueeze(0) for tensor in last_dict[label]], dim=0)
            return result_tensor.view(1, -1)
        else:
            # Recursive case: Call last_frame with frames_bbox_location[:-1] and reduced recursion_depth
            return last_frame(label, frames_bbox_location[:-1], recursion_depth - 1)
    else:
        return None


def distance_bbox(A_xyxy, B_xyxy):
    example_xyxy = torch.tensor([[284., 102., 408., 260.]])
    list = [A_xyxy, B_xyxy]
    for i in range(len(list)):
        xyxy = list[i]
        if not isinstance(xyxy, torch.Tensor) or xyxy.size() != example_xyxy.size():
            xyxy = [value.item() for value in xyxy]
            xyxy_new = torch.tensor([xyxy]) 
            list[i] = xyxy_new

    A_xyxy, B_xyxy = list[0], list[1]

    xyxy_center = torch.stack([(A_xyxy[:, 0] + A_xyxy[:, 2]) / 2, (A_xyxy[:, 1] + A_xyxy[:, 3]) / 2], dim=1)
    last_frame_center = torch.stack([(B_xyxy[:, 0] + B_xyxy[:, 2]) / 2, (B_xyxy[:, 1] + B_xyxy[:, 3]) / 2], dim=1)
    distance = torch.norm(xyxy_center - last_frame_center, dim=1)
    return distance


def clean_frames(frames_bbox_location):
    clean_list = []
    try:
        for frame_dict in reversed(frames_bbox_location):
            clean_list.append(frame_dict)
            if all(value and len(value) > 0 for value in frame_dict.values()):
                break
    except RuntimeError:
        return frames_bbox_location

    return list(reversed(clean_list))


def input_pkl(frame_num,pkl_dict,identity,output_emotion_dict):

    if "frame"+str(frame_num) not in pkl_dict:
        pkl_dict['frame'+str(frame_num)] = {}
    # if identity not in pkl_dict['frame'+str(frame_num)]:
    #     pkl_dict['frame'+str(frame_num)][identity] = {}

    pkl_dict['frame'+str(frame_num)][identity] = {'emotion': output_emotion_dict}
    new_dict =  pkl_dict

    return new_dict

def time_to_seconds(time_str):
    minutes, seconds = map(int, time_str.split(':'))
    return minutes * 60 + seconds

def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(300,300),#(640,480),#(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        sibling_confusion = False,
        sib_text_fp =ROOT / '../Downloads/SCtrim/trim41/1.txt'
):
    sibling_confusion_new = False
    start_time = time.time()
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    log_file_path = os.path.join(save_dir,'log.txt')

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    bad_frames = []


    frames_bbox_location = [] #memory of last frame's bbox location #[{1:{"mother":[xyxy],"child":[xyxy]}}]
    color_dict={"mother":0,"child":1,"other":3}
    pkl_dict = {}
    ignoreFirstFrame = False

    ##############Calculating frames#################################################################################
    import re

    for path, im, im0s, vid_cap, s in dataset:
        fps = vid_cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    duration_seconds = frame_count / fps 
    duration_minutes = duration_seconds / 60
 

    frame_duration = 1 / fps #each frame is how many seconds
    #print(f"frame_duration:{frame_duration}")
    #get frame nums that correspond to specific sibling-intrusion seconds 
    #text_path = "../Downloads/p39/p39_s02_2022-12-24_13_10_06/p39_s02_2022-12-24_13_10_06.txt"
    text_path = sib_text_fp
    siblingTime = []
    siblingFrames = []
    try:
        with open(text_path, 'r', encoding='utf-8') as text_file:
            for line in text_file:
                line_splits = re.split(',', line.strip())
                if len(line_splits)>0:
                    siblingTime.append(line_splits)
        print(f"siblingTime:{siblingTime}")
        
        
        for i in range(len(siblingTime)):
            start_time_str, end_time_str = siblingTime[i]
            start_time = time_to_seconds(start_time_str) #what seconds does sibling first come in 
            end_time = time_to_seconds(end_time_str) #what seconds does sibling leave
            start_frame = start_time/frame_duration
            end_frame = end_time/frame_duration
            siblingFrames.append((int(start_frame),int(end_frame)))

        print(f"siblingFrames:{siblingFrames}")
    except Exception as e:
        print(f"An error when looking at siblingFrames occurred: {sib_text_fp}")
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"An error when looking at siblingFrames occurred: {sib_text_fp}")
    #reload again for inference:
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    #[(2,400),(402,450)]
    ##############################################################################################################

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    log_start_time = time.time()

    for path, im, im0s, vid_cap, s in dataset:
        fps = vid_cap.get(cv2.CAP_PROP_FPS)
        frame_num = getattr(dataset, 'frame', 0)
        if frame_num % 500 == 0:
            with open(log_file_path, 'a') as log_file:
                log_file.write(f"Processing {frame_num} frames: {(time.time()-log_start_time)/60} minutes has passed.\n")
        #if frame_num equals 500, write to the log file how many time has passed (don't overwrite existing things in the log file)
        sibling_confusion_new = False

        for frames_tuple in siblingFrames:
            try:
                if frame_num>=frames_tuple[0] and frame_num<=frames_tuple[1]:
                    sibling_confusion_new = True
            except Exception as e:
                print(f"An error when looking at siblingFrames occurred: {sib_text_fp}")
                with open(log_file_path, 'a') as log_file:
                    log_file.write(f"An error when looking at siblingFrames occurred:  {sib_text_fp}")
            

        #print("frame_num"+str(frame_num))
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        #backup algorithms
        #sibling_confusion = False
        

        if sibling_confusion:
            #print("sibling confusion")
            compareDistance = False
            compareSize = True
        else:
            #print("NOT sibling confusion")
            compareDistance = True
            compareSize = False
            DISTANCE_THRESHOLD = 30



        # Process predictions
        backends = [
        'opencv',
        'ssd',
        'dlib',
        'mtcnn',
        'retinaface',
        'mediapipe',
        'yolov8',
        'yunet',
        ]
        txt_list = []
        #pkl_list = ["NEW FRAME"]

        label_counter = Counter()
       
        frames_bbox_location.append({"mother":[],"child":[]}) # [{frame1}, {frame2}]
        frames_bbox_location = clean_frames(frames_bbox_location)
        #print(f"length of frames_bbox_location{len(frames_bbox_location)}") #should see it increasing and decreasing over time

        for i, det in enumerate(pred):  # per image

            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)



            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            bad_save_path = save_path.split('.')[0]+"BAD"+"."+save_path.split('.')[1]
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop


            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if sibling_confusion_new:
                image_width, image_height = annotator.im.size if annotator.pil else (im.shape[1], im.shape[0])
                center_x = image_width // 2
                center_y = image_height // 2 
                annotator.text((center_x,center_y),"SIBLING CONFUSION NEW!!!")
    
        

            face_counter = {"mother":[],"child":[]}#{"mother":[(xyxy, emotion_dict),(xyxy, emotion_dict)], "child":[(xyxy, emotion_dict)]}
            altered_label_list = []
            others_list = []
    
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        ###!!!bounding box info
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class

                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        label_counter.update(label)
                    if save_crop:###extract the cropped and send to deepface for analysis, returned results print to image
                        start_time = time.time()  # Mark the end time
                        
                        output_path = save_dir / 'crops' / names[c] / 'cropped.jpg'
                        #save_one_box(xyxy, imc, file=output_path, BGR=True)
                        changed_save_one_box(xyxy, imc, file=output_path, BGR=True)
                        end_time = time.time()
                        total_time = end_time - start_time
                        #print(f"Crop and save: {total_time:.2f} seconds")
                        #running deepface analysis on the cropped image
                        try:
                            #demography = DeepFace.analyze(str(output_path), detector_backend = backends[0])
                            start_time = time.time()
                            #print(f"output_path:{output_path}")
                            demography = DeepFace.analyze(str(output_path), detector_backend = backends[3])
                            
                            #print(f"demography:{demography}")
                            emotion_dict = demography[0]['emotion']
                            identity = str(label).split()[0]
                            conf = str(label).split()[1]
                            face_counter[identity].append((xyxy,emotion_dict, conf)) #appending the xyxy position of the bounding box
                            end_time = time.time()
                            total_time = end_time - start_time
                            #print(f"Deepface: {total_time:.2f} seconds")
                        except ValueError as e:
                            pass
                            #print("DEEPFACE VALUE ERROR")
                            
                opposite_identities = {"mother":"child","child":"mother"}
                start_time = time.time()
                #print(f"face_counter:{face_counter}")
                if len(face_counter["mother"]) == 0 and len(face_counter["child"]) == 0:
                    #print("Case 1")
                    last_mother_xyxy = last_frame("mother", frames_bbox_location)
                    last_child_xyxy = last_frame("child", frames_bbox_location)
                    if not last_mother_xyxy == None:
                        demo = isFace(last_mother_xyxy,save_dir,"mother",imc,backends) #returns demo if is face, returns None if not face
                        if (demo):
                            outputsFace(frame_num,"CASE1 mother","mother",demo,pkl_dict,last_mother_xyxy,frames_bbox_location,annotator,color_dict)
                    if not last_child_xyxy == None:
                        demo = isFace(last_child_xyxy,save_dir,"child",imc,backends) 
                        if (demo):
                            outputsFace(frame_num,"CASE1 child","child",demo,pkl_dict,last_child_xyxy,frames_bbox_location,annotator,color_dict)

                elif len(face_counter["mother"]) == 1 and len(face_counter["child"]) == 1:
                    #print("Case 2")
                    #new_demo_xyxy(identity,face_counter)
                    mother_demo, mother_xyxy = new_demo_xyxy("mother",face_counter) 
                    child_demo, child_xyxy = new_demo_xyxy("child",face_counter) 
                    if not sibling_confusion_new:
                        outputsFace(frame_num,f"{frame_num}C2 mother","mother",mother_demo,pkl_dict,mother_xyxy,frames_bbox_location,annotator,color_dict)
                        outputsFace(frame_num,f"{frame_num}C2 child","child",child_demo,pkl_dict,child_xyxy,frames_bbox_location,annotator,color_dict)
                    else:
                        last_mother_xyxy = last_frame("mother", frames_bbox_location)
                        last_child_xyxy = last_frame("child", frames_bbox_location)
                        if not last_mother_xyxy == None:
                            if overlap_IOU(IOU(last_mother_xyxy,mother_xyxy,model)):
                                outputsFace(frame_num,f"{frame_num}C2 mother","mother",mother_demo,pkl_dict,mother_xyxy,frames_bbox_location,annotator,color_dict)
                        else:
                            outputsFace(frame_num,f"{frame_num}C2 mother","mother",mother_demo,pkl_dict,mother_xyxy,frames_bbox_location,annotator,color_dict)
                        if not last_child_xyxy == None:
                            if overlap_IOU(IOU(last_child_xyxy,child_xyxy,model)):
                                #print(last_child_xyxy)
                                outputsFace(frame_num,f"{frame_num}C2 child","child",child_demo,pkl_dict,child_xyxy,frames_bbox_location,annotator,color_dict)
                        else:
                            outputsFace(frame_num,f"{frame_num}C2 child","child",child_demo,pkl_dict,child_xyxy,frames_bbox_location,annotator,color_dict)

               
                elif (len(face_counter["mother"]) == 1 and len(face_counter["child"]) == 0) or (len(face_counter["mother"]) == 0 and len(face_counter["child"]) == 1):
                    #print("Case 3")
                    if  (len(face_counter["mother"]) == 1 and len(face_counter["child"]) == 0):
                        identity = "mother"
                        oppo = "child"
                    else:
                        identity = "child"
                        oppo = "mother"
                    #identity = "mother", oppo = "child"
                    last_oppo_xyxy = last_frame(oppo, frames_bbox_location)
                    last_identity_xyxy = last_frame(identity, frames_bbox_location)
                    new_identity_demo, new_identity_xyxy = new_demo_xyxy(identity,face_counter) 
                    if not last_oppo_xyxy == None:
                        demo = isFace(last_oppo_xyxy,save_dir,str(oppo),imc,backends) 
                        if (demo): #child is a person 
                            if overlap_IOU(IOU(last_oppo_xyxy,new_identity_xyxy,model)):# if child location = mother's old location
                                outputsFace(frame_num,f"{frame_num}C3 FOUND-EDIT {identity}",identity,demo,pkl_dict,last_identity_xyxy,frames_bbox_location,annotator,color_dict) #set the child as mother
                                outputsFace(frame_num,f"{frame_num}C3 EDIT {oppo}",oppo,new_identity_demo,pkl_dict,new_identity_xyxy,frames_bbox_location,annotator,color_dict) #set the mother as child

                            else: #child is child    
                                if sibling_confusion_new:
                                    if overlap_IOU(IOU(last_identity_xyxy,new_identity_xyxy,model)):
                                        outputsFace(frame_num,f"{frame_num}C3 {identity}",identity,new_identity_demo,pkl_dict,new_identity_xyxy,frames_bbox_location,annotator,color_dict)
                                else:
                                    outputsFace(frame_num,f"{frame_num}C3 {identity}",identity,new_identity_demo,pkl_dict,new_identity_xyxy,frames_bbox_location,annotator,color_dict)
                                if not overlap_special_IOU(special_IOU(last_oppo_xyxy,new_identity_xyxy,model)):  
                                    outputsFace(frame_num,f"{special_IOU(last_oppo_xyxy,new_identity_xyxy,model)}{frame_num}C3 FOUND {oppo}",oppo,demo,pkl_dict,last_oppo_xyxy,frames_bbox_location,annotator,color_dict)

                        else:
                            #if want to revert back, delete if else, only keep content in if: outputsFace
                            if not sibling_confusion_new:
                                outputsFace(frame_num,f"{frame_num}C3 {identity}",identity,new_identity_demo,pkl_dict,new_identity_xyxy,frames_bbox_location,annotator,color_dict)
                            else:
                                if overlap_IOU(IOU(last_identity_xyxy,new_identity_xyxy,model)):
                                    outputsFace(frame_num,f"{frame_num}C3 {identity}",identity,new_identity_demo,pkl_dict,new_identity_xyxy,frames_bbox_location,annotator,color_dict)

                elif (len(face_counter["mother"]) == 2 and len(face_counter["child"]) == 0) or (len(face_counter["mother"]) == 0 and len(face_counter["child"]) == 2):
                    #case 4
                    if sibling_confusion_new:
                        if (len(face_counter["mother"]) == 0 and len(face_counter["child"]) == 2):
                            identity = "child"
                            oppo = "mother"
                        else:
                            identity = "mother"
                            oppo = "child"
                        chosen_xyxy = chooseOne(face_counter,identity,compareSize,compareDistance,opposite_identities,frames_bbox_location,frame_num,pkl_dict,annotator,color_dict,i) 
                        last_oppo_xyxy = last_frame(oppo, frames_bbox_location)
                        if not last_oppo_xyxy == None:
                            demo = isFace(last_oppo_xyxy,save_dir,str(oppo),imc,backends) 
                            if (demo): #parent is a person 
                                if overlap_IOU(IOU(last_oppo_xyxy,chosen_xyxy,model)):
                                    outputsFace(frame_num,f"{frame_num}C4 FOUND {oppo}",oppo,demo,pkl_dict,last_oppo_xyxy,frames_bbox_location,annotator,color_dict)
                            #choose one for the child                            
                    else:
                        #print("Case 5")
                        if (len(face_counter["mother"]) == 2 and len(face_counter["child"]) == 0):
                            identity = "mother"
                            oppo = "child"
                        else:
                            identity = "child"
                            oppo = "mother"
                        
                        #sol2 for first-frame-problem
                        #note: if want sol1, comment out this if statement and shift everything to the left by one tab
                        if (ignoreFirstFrame):
                            if not (last_frame(identity, frames_bbox_location) == None and last_frame(oppo, frames_bbox_location)==None): #if it is the very first frame or mother and child changed seats   
                                markTwoSameIdentities(face_counter,identity,compareSize,compareDistance,opposite_identities,frames_bbox_location,frame_num,pkl_dict,annotator,color_dict,i) 
                        else:
                            markTwoSameIdentities(face_counter,identity,compareSize,compareDistance,opposite_identities,frames_bbox_location,frame_num,pkl_dict,annotator,color_dict,i) 

                else: #2 and 1 situation
                    #print("Case 6")
                    #if compareSize:
                    if sibling_confusion_new: 
                        last_mother_xyxy = last_frame("mother", frames_bbox_location)
                        last_child_xyxy = last_frame("child", frames_bbox_location)
                        if (len(face_counter["mother"]) == 1 and len(face_counter["child"]) == 2):      
                            mother_demo, mother_xyxy = new_demo_xyxy("mother",face_counter)
                            if not last_mother_xyxy == None:
                                #message += f"{frame_num}IOU last_mother_xyxy,mother_xyxy:{IOU(last_mother_xyxy,mother_xyxy,model)}"
                                if overlap_IOU(IOU(last_mother_xyxy,mother_xyxy,model)):
                                    outputsFace(frame_num,f"CASE6 mother","mother",mother_demo,pkl_dict,mother_xyxy,frames_bbox_location,annotator,color_dict)
                            chooseOne(face_counter,"child",compareSize,compareDistance,opposite_identities,frames_bbox_location,frame_num,pkl_dict,annotator,color_dict,i)        
                            #markTwoSameIdentities(face_counter,"child",compareSize,compareDistance,opposite_identities,frames_bbox_location,frame_num,pkl_dict,annotator,color_dict,i)        
                        elif(len(face_counter["child"]) == 1 and len(face_counter["mother"]) == 2):
                            child_demo, child_xyxy = new_demo_xyxy("child",face_counter)
                            if not last_child_xyxy == None:
                                #message += f"{frame_num}IOU last_child_xyxy,child_xyxy:{IOU(last_child_xyxy,child_xyxy,model)}"
                                if overlap_IOU(IOU(last_child_xyxy,child_xyxy,model)):
                                    outputsFace(frame_num,f"{frame_num}C6 child","child",child_demo,pkl_dict,child_xyxy,frames_bbox_location,annotator,color_dict)
                            chooseOne(face_counter,"mother",compareSize,compareDistance,opposite_identities,frames_bbox_location,frame_num,pkl_dict,annotator,color_dict,i)        
                        else:
                            bad_frames.append(im0)
                    else:
                        bad_frames.append(im0)
                end_time = time.time()
                total_time = end_time - start_time
                #print(f"Backup Algo: {total_time:.2f} seconds")

            #draw the "emotion result to image as well"
            start_txt_x, start_txt_y = 10, 50
            for altered_label in altered_label_list: #(real_identity, xyxy[0][0],xyxy[0][1])
                start_txt_x = int(altered_label[1])
                start_txt_y = int(altered_label[2])
                text = altered_label[0]
                annotator.text((start_txt_x, start_txt_y), text, txt_color=(255, 255, 0), anchor='top', box_style=True)
            for altered_label in others_list: #(real_identity, xyxy[0][0],xyxy[0][1])
                start_txt_x = int(altered_label[1])
                start_txt_y = int(altered_label[2])
                text = altered_label[0]
                annotator.text((start_txt_x, start_txt_y), text, txt_color=(0, 255, 0), anchor='top', box_style=True)

            # Stream results
            im0 = annotator.result()


            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)


            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'

                    #else:
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)


        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
     

    #Concatenate all the bad frames into a video
    bad_save_path = save_path.split('.')[0]+'_BAD'+'.'+save_path.split('.')[1]
    bad_vid_writer = cv2.VideoWriter(bad_save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for im_bad in bad_frames:
        bad_vid_writer.write(im_bad)
    # Print results

    with open(log_file_path, 'a') as log_file:
        total_seconds = time.time()-log_start_time
        total_minutes = total_seconds/60
        log_file.write(f"{time.time()}\nTotal Processing Time for this video of {frame_count,fps} frames, {duration_minutes}minutes: {total_minutes} minutes \n")

    #t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    #LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

    crop_path = save_dir / 'crops'
    shutil.rmtree(str(crop_path))

    try:
        path = os.path.join(save_dir,'emotion_res.pkl')
        # Try to open the pickle file in binary append mode ('ab')
        with open(path, 'ab') as file:
            # Append the data to the pickle file
            pickle.dump(pkl_dict, file)
    except FileNotFoundError:
        # If the file doesn't exist, create it and write the initial data
        with open(path, 'wb') as file:
            pickle.dump(pkl_dict, file)
    except Exception as e:
        # Handle other exceptions if needed
        print(f"An error occurred: {e}")



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--sibling-confusion', default=False, action='store_true', help='deals with sibling confusion by comparing face sizes instead of distances')
    parser.add_argument('--sib-text-fp', type=str,default=ROOT / '../Downloads/SCtrim/trim41/1.txt', help='(optional) sibling text')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))



if __name__ == '__main__':
    opt = parse_opt()
    main(opt)