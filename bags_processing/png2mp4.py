import cv2
import os
import glob
from tqdm import tqdm

def convert_frames_to_video(pathIn, pathOut, fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if f.endswith('.png')]
    #for sorting the file names properly
    files.sort(key = lambda x: int(x[:-4]))
    for i in tqdm(range(len(files))):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        #inserting the frames into an image array
        frame_array.append(img)
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

input_root_path = "d435_images"
output_root_path = "d435_videos"
dirs = os.listdir(input_root_path)
for dir in dirs:
    print(dir)
    # check dir is not a file, but a folder
    if os.path.isfile(input_root_path + "/" + dir):
        continue
    
    pathIn= input_root_path + "/" + dir + "/"
    pathOut = output_root_path + "/" + dir + ".mp4"
    os.makedirs(output_root_path, exist_ok=True)
    fps = 30.0
    convert_frames_to_video(pathIn, pathOut, fps)
    print("done")
