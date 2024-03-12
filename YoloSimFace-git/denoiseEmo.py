#implement a PSmoothed (Personalized Smooth) 
#if the maximum value of the median-smoothed detected metrics exceeds the personalized threshold
#that metric’s interpreted value is 1 [Highest accuracy in paper]

#thres = mean + 2*sd
#'frame27733': {'mother': {'emotion': {'angry': 57.08763003349304, 'disgust': 1.477209385484457, 'fear': 6.009625643491745, 'happy': 
#0.06856126710772514, 'sad': 18.735910952091217, 'surprise': 0.006926297646714374, 'neutral': 16.61413460969925}},
# 'child': {'emotion': {'angry': 0.8877509273588657, 'disgust': 1.9581990109290848e-10, 'fear': 0.5832368973642588, 
#'happy': 0.00014386722568815458, 'sad': 5.3427476435899734, 'surprise': 0.0002398940978309838, 'neutral': 93.18588376045227}}},
import pickle, math
from statistics import median, mean, stdev


def median_smoothing(data, window_size):
    smoothed_data = []
    half_window = window_size // 2  
    for i in range(len(data)):
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        window_data = data[start:end]
        smoothed_data.append(median(window_data))    
    return smoothed_data
##################################################################
def create_empt_smoothlist():
    identity = ['mother','child']
    emotion_list = ['angry','disgust','fear','happy','sad','surprise','neutral']
    smoothed_data = {}
    for iden in identity:
        smoothed_data[iden]={}
        for emo in emotion_list:
            smoothed_data[iden][emo] = []
    return smoothed_data

"""frame-smooth is median smoothing"""
def frame_smooth(dict_data, window): #window is for how many frames
    identity = ['mother', 'child']
    emotion_list = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    smoothed_data = create_empt_smoothlist()
    last_key = list(dict_data.keys())[-1]
    total_frames = int(last_key.split('frame')[1])
    for i in range(total_frames - window + 1):
        keyname_list = [f'frame{j}' for j in range(i, i + window)]
        for iden, emo in [(idn, emotion) for idn in identity for emotion in emotion_list]:
            #print(iden,emo)
            value_list = []
            last_k = 0
            for k in keyname_list:
                try:
                    value_list.append(dict_data[k][iden]['emotion'][emo])
                    last_k = k
                except KeyError:
                    value_list.append(dict_data[last_k][iden]['emotion'][emo]) #linear interpolation #orig code: pass
            if len(value_list) > 0:
                smoothed_data[iden][emo].append(median(value_list))  # You need to define median()
    return smoothed_data

"""
PSmooth is applied after Median Smoothing
For each window, if the maximum value of the median-smoothed detected metrics,that metric interpreted value is 1. 
The threshold value for a given metric is set at the mean value of that metric plus two standard deviations. 
"""
def PSmooth(smoothed_data, new_window): #window is for how many (120 frames)
    identity = ['mother', 'child']
    emotion_list = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    PSmooth_data = create_empt_smoothlist()
    mean_sd_list = create_empt_smoothlist()
    for iden, emo in [(idn, emotion) for idn in identity for emotion in emotion_list]:
        mean_sd_list[iden][emo] = [0, None]
    #start smoothing 
   
    for iden, emo in [(idn, emotion) for idn in identity for emotion in emotion_list]:
        one =0
        zero = 0
        N = 0 #total number of "medians" we have calculated
        for i in range(0, len(smoothed_data[iden][emo]), new_window):
            #i = 0, 5, 10, 15
            value_list = [smoothed_data[iden][emo][j] for j in range(i, min(i+new_window, len(smoothed_data[iden][emo])))]
            max_val = max(value_list)
            #also need to update the mean and standard deviation 
            for x in value_list:
                old_mean, old_sd = mean_sd_list[iden][emo][0],mean_sd_list[iden][emo][1]
                N+=1
                new_mean = (old_mean * (N-1) + x)/N
                if N>2:
                    old_var = old_sd*old_sd
                    new_var = ((N-2)*old_var + (x-new_mean)*(x-old_mean))/(N-1)
                    new_sd = math.sqrt(new_var)
                elif N==2:
                    new_sd = stdev([fir_median,x])
                elif N==1:
                    fir_median = x
                    new_sd = None
                else:
                    new_sd = None
                mean_sd_list[iden][emo] = [new_mean, new_sd]
            #determine PSmooth value    
            threshold = new_mean + 2 * new_sd if new_sd is not None else None
            print(f'threshold{threshold}')
            #f = open("thresholdlog.txt", "a")
            #f.write(f"thres:{threshold}\n")
            #f.write(f"{max_val}\n")
            #f.close()
            #print(max_val)
            if threshold is not None and max_val > threshold:
                PSmooth_data[iden][emo].append(1)
                one+=1
            else:
                PSmooth_data[iden][emo].append(0)
                zero+=1
        print(f"{one}/{one+zero}")

    return PSmooth_data
    

def main():
    pkl_file_path = 'emotion_res.pkl'
    with open(pkl_file_path, 'rb') as file:
        dict_data = pickle.load(file)
        smoothed_data = frame_smooth(dict_data, 120) #a dictionary consisting values {'mother': {'emo1':[33,2,11,33...],'emow':[33,2,11,33...]}}
        PSmooth_data = PSmooth(smoothed_data, 5) #a dictionary consisting values {'mother': {'emo1':[33,2,11,33...],'emow':[33,2,11,33...]}}
    output_pkl = 'smoothed_emotion_res.pkl'
    with open(output_pkl, 'wb') as file:  # 'wb' for writing in binary mode
        pickle.dump(smoothed_data, file)
    output_pkl = 'PSmooth_emotion_res.pkl'
    with open(output_pkl, 'wb') as file:  # 'wb' for writing in binary mode
        pickle.dump(PSmooth_data, file)
    with open(output_pkl, 'rb') as file:
        dict_data = pickle.load(file) 
        #print(dict_data)
        
if __name__ == "__main__":
    main()
"""   
def win_smooth(sent_start_time,sent_end_time,identity, emotion):
    start_frame, end_frame = frame(sent_start_time), frame(sent_end_time)
    keyname_list = [f'frame{j}' for j in range(start_frame,end_frame+1)]
    value_list = []
    for k in keyname_list:
        for iden, emo in zip(identity, emotion_list):
            try:
                value_list.append(dict_data[k][iden][emo])
            except KeyError:
                pass
        if len(value_list) > 0:
            return median(value_list)
        else:
            return None #TODO: Need to Change!

#smoothed_data = median_smoothing(time_series_data, window_size)
def sentence_smoothing(dic_data):
    import os
    # Replace '/Volumes/YourCloudDrive' with the actual path to your CloudMounter-mounted disk
    cloud_drive_path = '/Users/meng.lu/Library/CloudStorage/CloudMounter-JenniferLu'
    folder_path = 'PRG/[1-1]REV-Annotated-EXPANDED-[dyadic]'
    path = os.path.join(cloud_drive_path,folder_path)
    # List files and directories in the mounted drive
    smoothed_data = create_empt_smoothlist()
    identity = ['mother','child']
    emotion_list = ['angry','disgust','fear','happy','sad','surprise','neutral']
    for subdir in os.listdir(path):
        print(subdir)
        subdir_path = os.path.join(path, subdir)
        for file in os.listdir(subdir):
            if '.csv' in file:
                fp = os.path.join(subdir_path, subdir)
                with open(fp, 'r') as csvfile:
                    line_list = csvfile.readlines()
                    i=0
                    while (i<len(line_list)-1):
                        j = i+1
                        sent_start_time = line_list[i][timestamp]
                        while (j<len(line_list)):
                            if line_list[j].strip()==line_list[i].strip():
                                j+=1
                            else:
                                sent_end_time = line_list[j-1]#frame1 to frame3 same sentence, now j = 4
                        for iden, emo in zip(identity, emotion_list):#TODO: what if sent_start_time = sent_end_time???ok.
                            smoothed_data[iden][emo].append(win_smooth(sent_start_time,sent_end_time))
        #saved the smoothed data for each file somewhere
        #the smoothed data is has one datapt for each sentence





    #get the csv file and find the frames corresponding to each sentence, find the medium value
"""

