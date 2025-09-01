import cv2
import numpy as np
import os
import glob 
import shutil
import argparse

class Blur():
    def __init__(self, datadir='./Images', blur_level=1):
        self.datadir = datadir
        print(f'Dataset path : {self.datadir} \n')
        self.scenes = sorted(os.listdir(self.datadir))
        print(f'Folder in dataset : \n {self.scenes} \n')
        self.scenes_len = len(self.scenes)
        self.blur_level = blur_level
        self.datadir_blur = [[] for _ in range(self.blur_level)]
        
    def copy_folder(self):
        for i in range(self.scenes_len):
            for level in range(1, self.blur_level + 1):
                blur_folder_name = self.scenes[i] + f'_blur{level}'
                self.datadir_blur[level-1].append(blur_folder_name)

                target_dir = os.path.join(self.datadir, blur_folder_name)
                if not os.path.isdir(target_dir):
                    print(f"Copying files to {target_dir}")
                    shutil.copytree(os.path.join(self.datadir, self.scenes[i]), target_dir)
                else:
                    print(f"Folder {target_dir} already exists")
        print(f'Folders created with blur level {self.blur_level} in the dataset.')

    def read_img(self):
        blur_list = []
        for num in range(1, self.blur_level+1): 
            datadir_blur = self.datadir_blur[num-1]
            for folder in datadir_blur:
                blur_list.append(folder)
        
        print(blur_list)
        
        blur_level = 0
        for scene in blur_list:    
            blur_level += 1
            images_path_png = sorted([img for img in glob.glob(os.path.join(self.datadir, scene, '*.png')) if 'seg' not in img])
            print(f'Number of images in {scene}= {len(images_path_png)}\n')
            for img_path in images_path_png:
                blur_img = self.motion_blur_type(img=img_path, size=blur_level+1)
                filename = os.path.basename(img_path)
                new_file = os.path.join(self.datadir, scene, filename)
                cv2.imwrite(new_file, blur_img)
        print('Done') 
        
    def motion_blur_type(self, img, size=3):
        img = cv2.imread(img)
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size

        output = img
        for _ in range(3):  
            output = cv2.filter2D(output, -1, kernel_motion_blur)
        return output          
    
    def main(self):
        self.copy_folder()
        self.read_img()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Apply motion blur to image datasets")
    parser.add_argument('--datadir', type=str, default='./Images', help='Path to the dataset directory')
    parser.add_argument('--blur_level', type=int, default=1, help='Number of blur levels to generate')

    args = parser.parse_args()

    for subdir in os.listdir(args.datadir):
        dir_path = os.path.join(args.datadir, subdir)
        blur_runner = Blur(datadir=dir_path, blur_level=args.blur_level)
        blur_runner.main()

# Usage:
# python3 gaussian_blur.py --datadir Images --blur_level 1