import cv2
import numpy as np
import os
import glob 
import shutil

class blur():
    def __init__(self, datadir = './Boat_dataset_unity/Images/boats1-6', blur_level = 1):
        self.datadir = datadir
        print(f'Dataset path : {self.datadir} \n')
        self.scenes = sorted(os.listdir(self.datadir))
        print(f'Folder in dataset : \n {self.scenes} \n')
        self.scenes_len = len(self.scenes)
        self.blur_level = blur_level
        self.datadir_blur = [[] for i in range(self.blur_level)]
        
    def copy_folder(self):
        # Loop through each scene and create blur folders accordingly
        for i in range(self.scenes_len):
            for level in range(1, self.blur_level + 1):
                # Define the folder name based on the blur level
                blur_folder_name = self.scenes[i] + f'_blur{level}'
                self.datadir_blur[level-1].append(blur_folder_name)

                # Full path for the target directory
                target_dir = os.path.join(self.datadir, blur_folder_name)

                # Check if the directory already exists
                if not os.path.isdir(target_dir):
                    print(f"Copying files to {target_dir}")
                    # Copy files from the original folder to the new blur folder
                    shutil.copytree(os.path.join(self.datadir, self.scenes[i]), target_dir)
                else:
                    print(f"Folder {target_dir} already exists")

        # Print the created folders after the copying process
        print(f'Folders created with blur level {self.blur_level} in the dataset.')

        
    def read_img(self):
        blur_list = []
        
        # Loop through the blur levels
        for num in range(1, self.blur_level+1): 
            datadir_blur = self.datadir_blur[num-1]  # Access the appropriate list in self.datadir_blur
            
            # Loop through the folders in the blur level and add them to the blur_list
            for k in range(len(datadir_blur)):
                folder = str(datadir_blur[k])
                blur_list.append(folder)
        
        print(blur_list)
        
        blur_level = 0
        for scene in blur_list:    
            blur_level += 1
            images_path_jpg = sorted([img for img in glob.glob(os.path.join(self.datadir + '/' + scene + '/*.png')) if 'seg' not in img])

            print(f'Number of images in {scene}= {len(images_path_jpg)}\n')
            for i in range (len(images_path_jpg)):
                blur_img = self.motion_blur_type(img = images_path_jpg[i], size = blur_level+1)
                path_parts = images_path_jpg[i].split("/")
                filename = path_parts[-1]
                filename_without_extension = filename.split(".")[0]
                number = filename_without_extension
                new_file = self.datadir + '/' + scene + '/' + str(number) + '.png'
                cv2.imwrite(new_file, blur_img)
        print('Done') 
        
    def motion_blur_type(self, img, size=3):
        # Blur the image with a larger kernel size
        img = cv2.imread(img)
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size

        # Apply the kernel to the input image multiple times for stronger blur effect
        output = img
        for _ in range(3):  # Apply the filter multiple times
            output = cv2.filter2D(output, -1, kernel_motion_blur)
        return output          
    
    def main(self):
        self.copy_folder()
        self.read_img()
        
        
if __name__ == '__main__':
    blur1_6 = blur(datadir = './Images/boats1-6', blur_level = 2)
    blur1_6.main()
    blur7_13 = blur(datadir = './Images/boats7-13', blur_level = 2)
    blur7_13.main()