import os
from pathlib import Path

import cv2
import numpy as np

from output_video import VideoOuter


def find_the_most_right_face(face_list):
    right_face = None
    for (x, y, width, height) in face_list:
        if right_face is None:
            right_face = (x, y, width, height)
        (_, _, right_width, right_height) = right_face
        if right_width * right_height > width * height:
            right_face = (x, y, width, height)
    return right_face


if __name__ == '__main__':
    # 获取项目根目录
    project_root_path = Path(__file__).parent.parent.absolute()
    cache_center_file = os.path.join(project_root_path, "cache/anime_center_cache.txt")
    # 读取缓存的地址
    if os.path.exists(cache_center_file):
        with open(cache_center_file) as f:
            center_str = f.readline()
            frame_center_list = list(map(lambda s: int(s), center_str.split(",")))
    else:
        cache_face_file = os.path.join(project_root_path, "cache/anime_face_cache.txt")
        if os.path.exists(cache_face_file):
            with open(cache_face_file) as f:
                face_str = f.readline()
                frame_center_list = list(map(lambda s: int(s), face_str.split(",")))
        else:
            # 读取视频
            video_cap = cv2.VideoCapture(os.path.join(project_root_path, "res/anime.flv"))
            # 读取动漫人脸级联分类器
            face_cascade = cv2.CascadeClassifier(os.path.join(project_root_path, "res/lbpcascade_animeface.xml"))
            frame_center_list = []
            frame_index = -1
            while True:
                frame_index += 1
                _, frame = video_cap.read()
                if frame is None:
                    break

                # 灰度化处理帧
                gray_frame = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

                # 检测人脸
                faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1,
                                                      minNeighbors=5,
                                                      minSize=(24, 24))
                if faces is None or len(faces) == 0:
                    print("frame %d not have face" % frame_index)
                    frame_center_list.append(0)
                    continue
                print("frame %d find the face, face count: %d" % (frame_index, len(faces)))
                (x, y, width, height) = find_the_most_right_face(faces)
                print("the most right face: x=%d, y=%d, width:%d, height:%d" % (x, y, width, height))
                center = x + width / 2
                frame_center_list.append(center)
                print("the frame %d center is: %d" % (frame_index, frame_center_list[frame_index]))
                # frame = cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 5)
            video_cap.release()
            cache_dir = os.path.join(project_root_path, "cache")
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            with open(cache_face_file, 'w+') as f:
                f.write(','.join(map(lambda x: str(int(x)), frame_center_list)))
        video_cap = cv2.VideoCapture(os.path.join(project_root_path, "res/anime.flv"))
        frame_index = -1
        first_find_face = 0
        list_size = len(frame_center_list)
        for i in range(0, list_size):
            if frame_center_list[i] > 0:
                first_find_face = i
                break
        reversed_list = list(reversed(frame_center_list))
        last_find_face = 0
        for i in range(0, list_size):
            if reversed_list[i] > 0:
                last_find_face = list_size - i - 1
                break
        while True:
            frame_index += 1
            _, frame = video_cap.read()
            if frame is None:
                break
            # 200帧内有个粉色的窗帘 没法很好的判断 就跳过
            if frame_index < 200:
                continue
            if first_find_face <= frame_index <= last_find_face:
                continue
            print("frame %d try to find her hair" % frame_index)
            mask = cv2.inRange(frame, (220, 190, 218), (245, 216, 240))
            remove = cv2.bitwise_and(frame, frame, mask=mask)
            blue_frame = cv2.cvtColor(remove, cv2.COLOR_BGR2GRAY)
            _, canny = cv2.threshold(blue_frame, 200, 0, 3)
            _, canny = cv2.threshold(canny, 230, 0, 4)
            contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                print("the frame not have hair")
                continue
            contour = contours[np.argmax(list(map(cv2.contourArea, contours)))]
            (x, y, width, height) = cv2.boundingRect(contour)
            if height * width < 1100:
                print("the frame can't find a hair image")
                continue
            print("find her hair: x=%d, y=%d, width:%d, height:%d" % (x, y, width, height))
            center = x + width / 2
            frame_center_list[frame_index] = center
        video_cap.release()
        frame_center_list.append(1280)
        last_center = 960
        last_center_index = -1
        for i in range(0, len(frame_center_list)):
            center = frame_center_list[i]
            if center == 0:
                continue
            diff_distance = center - last_center
            for j in range(last_center_index + 1, i):
                frame_center_list[j] = (last_center + diff_distance * (
                        j - last_center_index) / (i - last_center_index))
                print("the frame %d center is: %d" % (j, frame_center_list[j]))
            last_center = center
            last_center_index = i
        cache_dir = os.path.join(project_root_path, "cache")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        with open(cache_center_file, 'w+') as f:
            f.write(','.join(map(lambda x: str(int(x)), frame_center_list)))
    VideoOuter(frame_center_list).start()
