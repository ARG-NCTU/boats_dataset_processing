#!/usr/bin/env python3

import rospy
import rospkg
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import os
from datetime import datetime
import threading
import sys
import select
import termios
import tty

rospack = rospkg.RosPack()

class SaveImages:
    def __init__(self):
        rospy.init_node('compressed_to_raw', anonymous=True)
        self.bridge = CvBridge()

        # 訂閱相機影像
        sub_topic_name = rospy.get_param('~sub_camera_topic', "/camera_pano_stitched/image_raw/compressed")
        self.image_sub = rospy.Subscriber(sub_topic_name, CompressedImage, self.image_callback)

        # 儲存資料夾
        self.save_dir = os.path.join(rospack.get_path('image_processing'), 'output', 'images')
        os.makedirs(self.save_dir, exist_ok=True)

        # 設定 FPS 參數
        self.fps = rospy.get_param('~fps', 18)
        self.output_fps = rospy.get_param('~output_fps', 2)
        self.max_image_count = self.fps // self.output_fps
        # self.max_image_count = 1
        self.image_count = 0

        # 暫停與恢復 flag
        self.paused = False
        self._start_key_listener()

    def _start_key_listener(self):
        def key_listener():
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            print("[按空白鍵切換暫停/繼續]")

            try:
                while not rospy.is_shutdown():
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1)
                        if key == ' ':
                            self.paused = not self.paused
                            state = "暫停中" if self.paused else "繼續中"
                            print(f"\n[狀態] {state}")
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        thread = threading.Thread(target=key_listener, daemon=True)
        thread.start()

    def image_callback(self, msg):
        if self.paused:
            return

        img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')

        if self.image_count == self.max_image_count:
            now = datetime.now()
            now_str = now.strftime("%Y%m%d%H%M%S") + f"{now.microsecond // 1000:03d}"
            save_path = os.path.join(self.save_dir, f"{now_str}.png")
            cv2.imwrite(save_path, img)
            rospy.loginfo(f"Saved image: {save_path}")
            self.image_count = 0
        else:
            self.image_count += 1

if __name__ == '__main__':
    try:
        node = SaveImages()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
