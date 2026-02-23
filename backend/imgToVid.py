import cv2
import os

image_folder = "val/Color_Images"
output_video = "input_video.mp4"
fps = 30

images = sorted(os.listdir(image_folder))
first_frame = cv2.imread(os.path.join(image_folder, images[0]))
h, w, _ = first_frame.shape

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

for img_name in images:
    frame = cv2.imread(os.path.join(image_folder, img_name))
    video.write(frame)

video.release()
print("Video created successfully!")