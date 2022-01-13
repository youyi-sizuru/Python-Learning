import os

import cv2
from pathlib import Path
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
    # 读取视频
    video_cap = cv2.VideoCapture(os.path.join(project_root_path, "res/anime.flv"))
    # 读取动漫人脸级联分类器
    face_cascade = cv2.CascadeClassifier(os.path.join(project_root_path, "res/lbpcascade_animeface.xml"))
    cache_center_file = os.path.join(project_root_path, "cache/anime_face_cache.txt")
    # 读取缓存的地址
    if os.path.exists(cache_center_file):
        with open(cache_center_file) as f:
            center_str = f.readline()
            frame_center_list = list(map(lambda s: int(s), center_str.split(",")))
    else:
        frame_center_list = []
        frame_index = 0
        last_center_index = -1

        while True:
            _, frame = video_cap.read()
            if frame is None:
                last_center = frame_center_list[last_center_index] if last_center_index >= 0 else 300
                diff_distance = 300 - last_center
                for i in range(last_center_index + 1, frame_index):
                    frame_center_list.append(last_center + diff_distance * (
                            i + 1 - last_center_index) / (frame_index - last_center_index))
                    print("the frame %d center is: %d" % (i, frame_center_list[i]))
                break

            # 灰度化处理帧
            gray_frame = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

            # 检测人脸
            faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1,
                                                  minNeighbors=5,
                                                  minSize=(24, 24))
            if faces is None or len(faces) == 0:
                print("frame %d not have face" % frame_index)
            else:
                print("frame %d find the face, face count: %d" % (frame_index, len(faces)))
                (x, y, width, height) = find_the_most_right_face(faces)
                print("the most right face: x=%d, y=%d, width:%d, height:%d" % (x, y, width, height))
                center = x + width / 2
                last_center = frame_center_list[last_center_index] if last_center_index >= 0 else 300
                diff_distance = center - last_center
                for i in range(last_center_index + 1, frame_index):
                    frame_center_list.append(last_center + diff_distance * (
                            i - last_center_index) / (frame_index - last_center_index))
                    print("the frame %d center is: %d" % (i, frame_center_list[i]))
                frame_center_list.append(center)
                last_center_index = frame_index
                print("the frame %d center is: %d" % (frame_index, frame_center_list[frame_index]))
            # frame = cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 5)
            frame_index += 1
        video_cap.release()
        cache_dir = os.path.join(project_root_path, "cache")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        with open(cache_center_file, 'w+') as f:
            f.write(','.join(map(lambda x: str(int(x)), frame_center_list)))
    VideoOuter(frame_center_list).start()
