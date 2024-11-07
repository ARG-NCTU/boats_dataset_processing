import cv2
import stitch
import utils
import os

# Resize setting, enter 1 to resize the resolution to 4x lower
resize = 0
for i in range(20477):
    frame_number = f"frame{i:04d}.jpg"
    # caculate execution time
    #print("Processing....")
    #start = timeit.default_timer()

    # load images
    list_images = utils.loadImages("/home/cj/Panorama/data/ocean/", frame_number, resize)

    # create panorama, default using ORB with nfeatures=3000, u can change to SIFT, SURF in features.py or add some argument
    iteration = i
    panorama = stitch.multiStitching(list_images, iteration)

    result_number = f"result{i:04d}.jpg"
    # save
    output_dir = "/home/cj/Panorama/result"
    if os.path.exists(output_dir):
        cv2.imwrite(os.path.join(output_dir, result_number), panorama)
    else:
        os.makedirs(output_dir)
        cv2.imwrite(os.path.join(output_dir, result_number), panorama)

    #stop = timeit.default_timer()
    #print("Complete!")
    #print("Execution time: ", stop - start)