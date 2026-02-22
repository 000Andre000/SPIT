import subprocess
from pathlib import Path

folder = r"C:\Users\hp\Desktop\ai_FRONTEND\frontend\public\videos"

videos = list(Path(folder).glob("*.mp4"))
print(f"Found {len(videos)} videos\n")

for video in videos:
    output = video.parent / f"fixed_{video.name}"
    print(f"Converting: {video.name}")
    result = subprocess.run([
        'ffmpeg', '-y',
        '-i', str(video),
        '-vcodec', 'libx264',
        '-profile:v', 'baseline',
        '-level', '3.0',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        str(output)
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"  ✓ Done → fixed_{video.name}")
    else:
        print(f"  ✗ Failed: {result.stderr[-200:]}")

print("\nAll done!")