import os
import sys
import cv2
from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg

sidecheck = False

if len(sys.argv) < 2:
    print("illegal parameters")
    sys.exit(0)
elif len(sys.argv) >= 3:
    sidecheck = True

f = open(sys.argv[1], "r")
lines = [line.strip() for line in f.readlines()]
f.close()

for i,line in enumerate(lines):
    filepath = os.path.join("./data", line.split(' ')[0])

    if os.path.exists(filepath):
        if not sidecheck:
            print(line)
            continue

        cap = cv2.VideoCapture(filepath)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if height < 200 or frame_count < 24:
            continue

        if width/height < 1.2 or width/height > 2:
            continue

        scene_list = detect(filepath, AdaptiveDetector())

        print(line.split(' ')[0], fps, width, height, len(scene_list))
        if len(scene_list) == 0:
            print(line)
