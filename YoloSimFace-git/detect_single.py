import cv2
import os
import pickle
from numpy import argmax
from deepface import DeepFace
import sys

def input_pkl(frame_num, pkl_dict, identity, output_emotion_dict):
    if "frame"+str(frame_num) not in pkl_dict:
        pkl_dict['frame'+str(frame_num)] = {}
    pkl_dict['frame'+str(frame_num)][identity] = {'emotion': output_emotion_dict}
    return pkl_dict

def main(video_path):
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
    pkl_dict = {}
    cap = cv2.VideoCapture(video_path)
    # Initialize VideoWriter object outside the loop
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join('detect_single_output',os.path.basename(video_path))
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        print(ret)
        if not ret:
            break

        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        try:
            demography = DeepFace.analyze(frame, actions=['emotion'], detector_backend = backends[3])
            emotion_dict = demography[0]['emotion']
            face_objs = DeepFace.extract_faces(frame, target_size=(224, 224),detector_backend = backends[3])
            
            if len(face_objs) > 0:
                max_index = argmax([face_objs[i]['confidence'] for i in range(len(face_objs))])
                face = face_objs[max_index]
            else:
                face = face_objs[0]
            
            facial_area = face['facial_area']
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            pkl_dict = input_pkl(frame_num, pkl_dict, 'child', emotion_dict)

        except (ValueError, AttributeError) as e:
            print(f"An error occurred: {e}")
        
        output_video.write(frame)

    cap.release()
    output_video.release()
    cv2.destroyAllWindows()

    try:
        pkl_file = os.path.splitext(os.path.basename(video_path))[0]  # Extracting the file name without extension
        path = os.path.join('detect_single_output', f'{pkl_file}.pkl')
        with open(path, 'wb') as file:
            pickle.dump(pkl_dict, file)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py path_to_video_file")
        sys.exit()
    video_path = sys.argv[1]
    main(video_path)
