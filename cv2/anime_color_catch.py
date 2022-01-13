import os
import shutil
from pathlib import Path

import cv2
import numpy as np

if __name__ == '__main__':
    # 获取项目根目录
    project_root_path = Path(__file__).parent.parent.absolute()
    # 读取视频
    video_cap = cv2.VideoCapture(os.path.join(project_root_path, "res/anime.flv"))
    cache_center_file = os.path.join(project_root_path, "cache/anime_color_cache.txt")
    # 读取缓存的地址
    if os.path.exists(cache_center_file):
        with open(cache_center_file) as f:
            center_str = f.readline()
            frame_center_list = list(map(lambda s: int(s), center_str.split(",")))
    else:
        frame_center_list = []
        frame_index = -1
        last_center_index = -1
        lower_color = (220, 190, 218)
        upper_color = (245, 216, 240)

        save_path = "C:\\Users\\xzp\\Desktop\\frames"
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs("C:\\Users\\xzp\\Desktop\\frames")
        while True:
            frame_index += 1
            _, frame = video_cap.read()
            if frame is None:
                # last_center = frame_center_list[last_center_index] if last_center_index >= 0 else 300
                # diff_distance = 300 - last_center
                # for i in range(last_center_index + 1, frame_index):
                #     frame_center_list.append(last_center + diff_distance * (
                #             i + 1 - last_center_index) / (frame_index - last_center_index))
                #     print("the frame %d center is: %d" % (i, frame_center_list[i]))
                break

                # 灰度并高斯模糊化处理帧
            if frame_index < 200:
                continue
            mask = cv2.inRange(frame, lower_color, upper_color)
            remove = cv2.bitwise_and(frame, frame, mask=mask)
            blue_frame = cv2.cvtColor(remove, cv2.COLOR_BGR2GRAY)
            _, canny = cv2.threshold(blue_frame, 200, 0, 3)
            _, canny = cv2.threshold(canny, 230, 0, 4)

            # cv2.imshow("aa", canny)
            # cv2.waitKey(0)
            contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                print("frame %d not have color" % frame_index)
                continue
            contour = contours[np.argmax(list(map(cv2.contourArea, contours)))]
            (x, y, width, height) = cv2.boundingRect(contour)
            if width / height > 4 or height / width > 4 or height * width < 500:
                print("frame %d is not a right color" % frame_index)
                continue
            mask = np.zeros_like(frame)
            mask[:, :, :] = 255
            mask = cv2.drawContours(mask, [contour], 0, (0, 0, 0), cv2.FILLED)
            result = cv2.bitwise_or(frame, mask)
            crop_result = frame[y:y + height, x:x + width]
            cv2.imwrite("C:\\Users\\xzp\\Desktop\\frames\\frame%d.png" % frame_index, crop_result)
