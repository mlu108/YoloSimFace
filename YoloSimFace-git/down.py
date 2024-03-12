#go into Log.txt and grab the lines after "Downloaded"
#Grab the video names 
#Matlaberp6: SUCCESS: python ../yolov5/detect_deepface.py --weights '../yolov5/triadic_dataset/feature_extraction4/weights/best.pt' --conf 0.6 --source '../yolov5/triadic_videos_nosiblings/p78/p78_s06/p78_s06.mp4' --augment --name 'TRIADICp78_s06' --save-crop --line=3 --agnostic --device 0,1,2,3.
#Matlaber6: SUCCESS: python ../yolov5/detect_deepface.py --weights '../yolov5/triadic_dataset/feature_extraction4/weights/best.pt' --conf 0.6 --source '../yolov5/triadic_videos_nosiblings/p54/p54_s04/p54_s04.mp4' --augment --name 'TRIADICp54_s04' --save-crop --line=3 --agnostic --device 0,1,2,3.
#if it is p54_s06 find the "largest"
#put all the directories into one folder 
#generate commandline command for downloading that fodler 
#add Donwloaded to 

import os,csv
import shutil,argparse

def find_largest(output_name):
    """Finds the corresponding output file name."""
    output_dir = '../yolov5/runs/detect'
    possible = [dirname for dirname in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, dirname)) and output_name in dirname]
    print(possible)
    if not possible:
        return None
    biggest_name = max(possible, key=lambda x: int(x.split('_s')[1]))
    return biggest_name

def main(action):
    if action =='c':
        with open('record.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for file in os.listdir('ForDownload'):
                if 'DS' not in file:
                    num = (file.split('_s')[1]).strip()[:2]
                    #print((file.split('_s')[1]).strip())
                    writer.writerow([((file.split('_s')[0]).strip().split('IC')[1]+'_s'+num),'Success'])


    if action =='b': #b = delete
        output_dir = '../yolov5/runs/detect'
        download_folder = 'ForDownload'
        for file in os.listdir(download_folder): 
            source_fp = os.path.join(download_folder, file)  # Source file path
            dest_fp = os.path.join(output_dir, file)         # Destination file path
            shutil.move(source_fp, dest_fp)                  # Move the file back to the original directory
        os.rmdir(download_folder)
        print("Deleted ForDownload.")
    if action =='a':
        output_dir = '../yolov5/runs/detect'
        names_list = []
        line_num = 0
        # Read lines after "Downloaded" marker from the log file
        found_downloaded = False
        with open('Log.txt', 'r') as reader:
            for line in reader:
                if found_downloaded:
                    string = line
                    if "SUCCESS" in string:
                        name = string.split('--source')[1].split('--augment')[0].split('triadic_videos_nosiblings/')[1].split('/')[1].strip()
                        print(name)
                        output_name = 'TRIADIC' + name
                        largest_output = find_largest(output_name)
                        if largest_output is None:
                            print(f'ERROR: No Corresponding File: {name}')
                        else:
                            names_list.append(largest_output)
                        line_num += 1
                elif line.strip() == 'Downloaded':
                    found_downloaded = True

        # Find a unique folder name for the download folder
        download_folder = 'ForDownload'
        print(f"download_folder:{download_folder}")
        if os.path.exists(download_folder):
            print(f"ERROR: FOLDER {download_folder} EXISTS! (did not delete before.)")
        else:
            os.mkdir(download_folder)
            # Move output files to the download folder
            for name in names_list:
                output_fp = os.path.join(output_dir, name)
                shutil.move(output_fp, download_folder)
            # Append 'Downloaded' to the end of the log file
            if line_num == len(names_list):
                #for file in os.listdir(download_folder): 
                download_command = f'scp -r mlu108@matlaberp6.media.mit.edu:jl/TriadicYoloSimFace/{download_folder} Downloads'
                print(download_command)
                print('RobotsRock23!')
                with open('Log.txt', 'r') as reader:
                    lines = reader.readlines()
                with open('Log.txt', 'w') as writer:
                    for line in lines:
                        if 'downloaded' in line:
                            writer.write(' ')
                        else:
                            writer.write(line)
                with open('Log.txt', 'a') as writer:
                    writer.write('Downloaded\n')
            else:
                print("ERROR: Some files missing")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSV file and run detection commands.")
    parser.add_argument("action", choices=['a', 'b','c'], help="Specify the action to perform: 'a' for generating download command or 'b' for finishing downloading.")
    args = parser.parse_args()
    main(args.action)
