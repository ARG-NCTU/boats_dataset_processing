import cv2
import json
import numpy as np
import argparse

# 全域變數: 四個頂點, -1 表示不在拖曳
rect_pts = [(100, 100), (400, 100), (400, 400), (100, 400)]
dragging = -1

# 更新矩形頂點, 同時保持矩形形狀
def update_rectangle(x, y, idx):
    global rect_pts
    # 固定相對邊
    if idx == 0:  # 左上
        rect_pts[0] = (x, y)
        rect_pts[1] = (rect_pts[1][0], y)
        rect_pts[3] = (x, rect_pts[3][1])
    elif idx == 1:  # 右上
        rect_pts[1] = (x, y)
        rect_pts[0] = (rect_pts[0][0], y)
        rect_pts[2] = (x, rect_pts[2][1])
    elif idx == 2:  # 右下
        rect_pts[2] = (x, y)
        rect_pts[3] = (rect_pts[3][0], y)
        rect_pts[1] = (x, rect_pts[1][1])
    elif idx == 3:  # 左下
        rect_pts[3] = (x, y)
        rect_pts[2] = (rect_pts[2][0], y)
        rect_pts[0] = (x, rect_pts[0][1])

# 顯示矩形與尺寸
def update_frame(frame):
    global rect_pts
    temp = frame.copy()
    # 畫頂點與邊
    for p in rect_pts:
        cv2.circle(temp, p, 5, (0, 255, 0), -1)
    for i in range(4):
        cv2.line(temp, rect_pts[i], rect_pts[(i+1)%4], (0, 255, 0), 2)
    # 計算 width, height
    w = int(np.linalg.norm(np.array(rect_pts[0]) - np.array(rect_pts[1])))
    h = int(np.linalg.norm(np.array(rect_pts[0]) - np.array(rect_pts[3])))
    # 顯示文字
    cv2.putText(temp, f"W:{w}px H:{h}px", (rect_pts[0][0], rect_pts[0][1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imshow("Video", temp)

# 輸入尺寸, 重新設定矩形 (以左上角為基準)
def input_dimensions():
    global rect_pts
    try:
        w = int(input("請輸入寬度 (px): "))
        h = int(input("請輸入高度 (px): "))
    except ValueError:
        print("輸入錯誤，請輸入整數")
        return
    x0, y0 = rect_pts[0]
    rect_pts = [(x0, y0), (x0+w, y0), (x0+w, y0+h), (x0, y0+h)]
    print(f"矩形已設定為 W={w}px, H={h}px")

# 滑鼠事件
def click_and_drag(event, x, y, flags, param):
    global dragging
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, p in enumerate(rect_pts):
            if abs(x-p[0])<10 and abs(y-p[1])<10:
                dragging = i
                break
    elif event == cv2.EVENT_MOUSEMOVE and dragging != -1:
        update_rectangle(x, y, dragging)
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = -1

# 儲存裁剪參數
def save_cropped_info(json_path, pts, w, h):
    info = {"points": pts, "width": w, "height": h}
    with open(json_path, 'w') as f:
        json.dump(info, f, indent=2)

# 執行裁剪
def crop_video(cap, output_path, json_path):
    pts_src = np.array(rect_pts, dtype=np.float32)
    w = int(np.linalg.norm(pts_src[0] - pts_src[1]))
    h = int(np.linalg.norm(pts_src[0] - pts_src[3]))
    pts_dst = np.array([[0,0], [w-1,0], [w-1,h-1], [0,h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        warped = cv2.warpPerspective(frame, M, (w, h))
        out.write(warped)
    out.release()
    save_cropped_info(json_path, rect_pts, w, h)
    print("裁剪完成")

# 主程式
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a video to show various processing results in quadrants.")
    parser.add_argument('-i', '--input', type=str, default="sonar/20251028051742.mp4", help='Path to the input video file')
    parser.add_argument('-o', '--output', type=str, default="sonar/20251028051742-output.mp4", help='Path to the output cropped video file')
    parser.add_argument('-j', '--json', type=str, default="sonar/20251028051742.json", help='Path to the output cropped info file')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print("無法開啟影片")
        exit(1)

    cv2.namedWindow("Video")
    cv2.resizeWindow("Video", 800, 600)
    cv2.setMouseCallback("Video", click_and_drag)

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        update_frame(frame)
        key = cv2.waitKey(30) & 0xFF
        if key == 13:  # Enter 開始裁剪
            crop_video(cap, args.output, args.json)
            break
        elif key == ord('i'):  # 按 i 直接輸入尺寸
            input_dimensions()
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

