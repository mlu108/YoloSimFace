# YoloSimFace

denoiseEmo.py contains smoothing code to extract high-frequency features.
detect_deepface.py is the YoloSimFace model to run face differentiation and emotion detection.
detect_single.py is a backup plan to deal with failed YoloSimFace detections. It takes in cropped videos focusing on one person and outputs his/her emotion.
down.py generates a command-line for downloading the videos that have finished processing.
missValues.py finds how many missing values are there from the emotion_res.pkl.
pkl_merger.py contains code to edit the pkl to fix errors.
sibtest_triadic.py runs detect_deepface.py on different matlabers.
