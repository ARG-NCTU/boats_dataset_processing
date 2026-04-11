import os
import cv2

# ====== 設定區 ======
img_dir = os.path.expanduser(
    "~/opencv-cuda-docker/bags/mascoma-20221007/mascoma-20221007/undist_images/CAM_FRONT"
)
output_video = "output.mp4"
fps = 10  # 每秒幾張，可自行調整
# ===================

# 取得並排序所有圖片
images = sorted([
    f for f in os.listdir(img_dir)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

if not images:
    raise RuntimeError("找不到任何圖片")

# 讀第一張來取得影像大小
first_img_path = os.path.join(img_dir, images[0])
frame = cv2.imread(first_img_path)
if frame is None:
    raise RuntimeError(f"無法讀取圖片: {images[0]}")

height, width, _ = frame.shape

# 建立 VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# 逐張寫入影片
for img_name in images:
    img_path = os.path.join(img_dir, img_name)
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"⚠️ 跳過無法讀取的圖片: {img_name}")
        continue

    video.write(frame)

video.release()
print(f"🎬 影片已產生: {output_video}")
