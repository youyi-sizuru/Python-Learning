import os

import cv2
from pathlib import Path
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # 读取视频
    video_cap = cv2.VideoCapture(os.path.join(Path(__file__).parent.parent.absolute(), "res/anime.flv"))
    ret, frame = video_cap.read()
    plt.imshow(frame), plt.show()
