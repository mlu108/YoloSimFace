import argparse
import subprocess
import csv
import os

def generate_commands(csv_file,matlaber):
    # Base command template
    base_command = "python ../yolov5/detect_deepface.py --weights '../yolov5/triadic_dataset/feature_extraction4/weights/best.pt' --conf 0.6 --source '{source_path}' --augment --name '{video_name}' --save-crop --line=3 --agnostic --device 0,1,2,3"
    if matlaber =='p7' or matlaber =='11':
        base_command= "python ../yolov5/detect_deepface.py --weights '../yolov5/triadic_dataset/feature_extraction4/weights/best.pt' --conf 0.6 --source '{source_path}' --augment --name '{video_name}' --save-crop --line=3 --agnostic --device 0,1"
    commands = []
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header
        for row in csv_reader:
            if len(row) > 0:
                info = row[0]
                try:
                    source_path = os.path.join('../yolov5', info.strip())
                    video_name = 'TRIADIC' + os.path.basename(info).split('.')[0]
                    commands.append(base_command.format(source_path=source_path, video_name=video_name))
                except Exception as e:
                    print(f"UNABLE to generate command for {info}")
    return commands

def write_to_log(message):
    with open(log_fp, 'a') as log_file:
        log_file.write(message + '\n')

def delete_entry(matlaber, command):
    csv_file = f'nosib{matlaber}.csv'
    # Extracting the relevant part from the command
    part = command.split('--source')[1].split('--augment')[0].split('yolov5/')[1].split("'")[0].strip()   
    # Read the CSV file and store its content
    with open(csv_file, 'r') as file:
        lines = file.readlines()
    # Check if the part exists in any line of the CSV and remove that line
    lines = [line for line in lines if part not in line]
    # Write the updated content back to the CSV file
    with open(csv_file, 'w') as file:
        file.writelines(lines)


def run_detection(command,matlaber):
    try:
        subprocess.run(command, shell=True, check=True)
        write_to_log(f"Matlaber{matlaber}: SUCCESS: {command}.")
        #Go into the corresponding txt file and delete the entry
        delete_entry(matlaber,command)
    except subprocess.CalledProcessError as e:
        # Log the error
        error_message = f"Matlaber{matlaber}: ERROR: executing command: {e}"
        write_to_log(error_message)
        print(error_message)

def main(matlaber):
    global log_fp
    #log_fp = os.path.join(f'ProcessingLog{matlaber}.txt')
    log_fp = os.path.join(f'Log.txt')
    write_to_log(f"{matlaber} started processing.")
    csv_file = f'nosib{matlaber}.csv'
    commands = generate_commands(csv_file,matlaber)
    for each_command in commands:
        print(each_command)
    for command in commands:
        run_detection(command,matlaber)
    write_to_log(f"{matlaber} has finished all the processings.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSV file and run detection commands.")
    parser.add_argument("matlaber", help="Path to the CSV file containing video information.")
    args = parser.parse_args()
    main(args.matlaber)
