import os
from pathlib import Path

import cv2


class VideoOuter:
    def __init__(self, frame_center_list: list):
        self.frame_center_list = frame_center_list

    def start(self):
        project_root_path = Path(__file__).parent.parent.absolute()
        video_cap = cv2.VideoCapture(os.path.join(project_root_path, "res/anime.flv"))
        fourcc = cv2.VideoWriter_fourcc('F', 'L', 'V', '1')
        video_writer = cv2.VideoWriter(os.path.join(project_root_path, "cache/anime_output.flv"), fourcc, 24.0,
                                       (600, 1080))
        frame_index = 0
        while True:
            _, frame = video_cap.read()
            if frame is None or frame_index >= len(self.frame_center_list):
                break
            center = self.frame_center_list[frame_index]
            if center < 300:
                left_x = 0
                right_x = 600
            elif center > 1920 - 300:
                left_x = 1920 - 600
                right_x = 1920
            else:
                left_x = center - 300
                right_x = center + 300
            # 裁剪中心点
            frame = frame[0:1080, int(left_x):int(right_x)]
            print("write %d frame to video" % frame_index)
            try:
                video_writer.write(frame)
            except:
                print("write error, change plan")
                save_image = os.path.join(project_root_path, "cache/temp.png")
                cv2.imwrite(save_image, frame)
                read_frame = cv2.imread(save_image)
                video_writer.write(read_frame)
            frame_index += 1
        video_cap.release()
        video_writer.release()
