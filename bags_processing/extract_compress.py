import os
import cv2
from cv_bridge import CvBridge
import rosbag
import numpy as np

class arg_bag_extracter:
    def __init__(self, bag_file, output_dir, image_topic): 
        self.bag_file = bag_file
        self.bag = rosbag.Bag(self.bag_file, "r")
        self.output_dir = output_dir
        self.image_topic = image_topic

    def extract(self):
        bridge = CvBridge()
        count = 0
        dismiss = 0
        for topic, msg, t in self.bag.read_messages(topics=[self.image_topic]):
            # if dismiss < 4:
            #     dismiss += 1
            #     continue
            dismiss = 0
            count += 1
            np_arr = np.frombuffer(msg.data, np.uint8)
            # Decode the compressed image
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # cv_img = cv2.flip(cv_img, 0)
            #cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2.imwrite(os.path.join(self.output_dir, "%i.png" % count), cv_img)
        self.bag.close()
        print("done, count is "+ str(count) + " and output to " + self.output_dir)

path = "d435_bags/"
output_root_dir = "d435_images/"
# image_topic_list = ["/camera_left/color/image_raw/compressed",
#                     "/camera_middle/color/image_raw/compressed",
#                     "/camera_right/color/image_raw/compressed"]
image_topic_list = ["/camera_middle/color/image_raw/compressed"]
bags = os.listdir(path)
for bag in bags:
    bag_file = path + bag
    print(bag_file)
    # output_dir_list = [output_root_dir + bag[:-4] + "_left/",
    #                    output_root_dir + bag[:-4] + "_mid/",
    #                    output_root_dir + bag[:-4] + "_right/"]
    output_dir_list = [output_root_dir + bag[:-4] + "_mid/"]
    for output_dir, image_topic in zip(output_dir_list, image_topic_list): 
        print(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        extracter = arg_bag_extracter(bag_file, output_dir, image_topic)
        extracter.extract()
