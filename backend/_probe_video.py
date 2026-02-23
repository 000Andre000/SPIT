import cv2
from pathlib import Path

files = sorted(
    Path("D:/SPIT/backend/result").glob("segmented_*.mp4"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)

if not files:
    print("NO_FILES")
    raise SystemExit(0)

path = str(files[0])
cap = cv2.VideoCapture(path)
print("FILE", path)
print("OPENED", cap.isOpened())

fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
codec = "".join([chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)])
print("FOURCC", repr(codec))
print("FPS", cap.get(cv2.CAP_PROP_FPS))
print("WIDTH", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("HEIGHT", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("FRAMES", cap.get(cv2.CAP_PROP_FRAME_COUNT))

ret, frame = cap.read()
print("FIRST_FRAME_OK", ret)
if ret and frame is not None:
    print("FIRST_FRAME_SHAPE", frame.shape)

cap.release()
