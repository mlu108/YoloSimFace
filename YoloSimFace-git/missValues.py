import pickle
import numpy as np

def find_miss(dict_data):
    last_key = list(dict_data.keys())[-1]
    total_frames = int(last_key.split('frame')[1])
    data_len = len(dict_data)
    print(data_len/total_frames)



def exponential_moving_average(data_list, period, weighting_factor=0.2):
    ema = np.zeros(len(data_list))
    sma = np.mean(data_list[:period])
    ema[period - 1] = sma
    for i in range(period, len(data_list)):
        ema[i] = (data_list[i] * weighting_factor) + (ema[i - 1] * (1 - weighting_factor))
    return ema

def fill_miss(data_dict):
    last_key = list(data_dict.keys())[-1]
    total_frames = int(last_key.split('frame')[1])
    data_list = list(data_dict.values())
    ema_list = exponential_moving_average(data_list, 20)  # Adjust window size as needed
    return_dict = {}
    for i, v in enumerate(ema_list):
        return_dict['frame'+str(i)] = v
    return return_dict


def main():
    pkl_file_path = 'emotion_res.pkl'
    with open(pkl_file_path, 'rb') as file:
        dict_data = pickle.load(file)
        smoothed_data = find_miss(dict_data) #a dictionary consisting values {'mother': {'emo1':[33,2,11,33...],'emow':[33,2,11,33...]}}
        
if __name__ == "__main__":
    main()