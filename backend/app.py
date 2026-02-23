from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import FileResponse, StreamingResponse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import shutil
import os
import uuid
import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path

app = FastAPI(title="Video Segmentation API")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# ================= SETTINGS =================
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILENAME = "randomCrop_shaun.pth"
MODEL_CANDIDATES = [
    BASE_DIR / "models" / MODEL_FILENAME,
    BASE_DIR.parent / "models" / MODEL_FILENAME,
]
MODEL_PATH = next((path for path in MODEL_CANDIDATES if path.exists()), MODEL_CANDIDATES[0])
ALPHA = 0.5  # transparency overlay

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

JOB_PROGRESS = {}

# ================= SEGMENTATION HEAD =================
class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=7, padding=3),
            nn.GELU()
        )
        self.block = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block(x)
        return self.classifier(x)

# ================= LOAD BACKBONE =================
print("Loading DINOv2 backbone...")
backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
backbone.eval().to(DEVICE)

# ================= MATCH TRAINING SIZE =================
w = int(((960 / 2) // 14) * 14)
h = int(((540 / 2) // 14) * 14)

dummy = torch.randn(1, 3, h, w).to(DEVICE)
with torch.no_grad():
    out = backbone.forward_features(dummy)["x_norm_patchtokens"]
embed_dim = out.shape[2]
print("Embedding dimension:", embed_dim)

# ================= LOAD TRAINED CHECKPOINT =================
model = SegmentationHeadConvNeXt(
    in_channels=embed_dim,
    out_channels=10,
    tokenW=w // 14,
    tokenH=h // 14
).to(DEVICE)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

checkpoint = torch.load(str(MODEL_PATH), map_location=DEVICE)
state_dict = checkpoint.get("model_state_dict", checkpoint)
clean_state_dict = {k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k: v
                    for k, v in state_dict.items()}
model.load_state_dict(clean_state_dict)
model.eval()
print("Segmentation head loaded successfully!")

# ================= TRANSFORMS =================
transform = transforms.Compose([
    transforms.Resize((h, w)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ================= COLOR PALETTE =================
color_palette = np.array([
    [0, 0, 0],
    [34, 139, 34],
    [0, 255, 0],
    [210, 180, 140],
    [139, 90, 43],
    [128, 128, 0],
    [139, 69, 19],
    [128, 128, 128],
    [160, 82, 45],
    [135, 206, 235],
], dtype=np.uint8)

def mask_to_color(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i in range(10):
        color_mask[mask == i] = color_palette[i]
    return color_mask

from fastapi.staticfiles import StaticFiles
import os

# Mount result folder as static (add this after app = FastAPI(...))
os.makedirs("result", exist_ok=True)
app.mount("/results", StaticFiles(directory="result"), name="results")

@app.get("/result/videos")
async def list_result_videos():
    """Returns list of all videos in the result folder."""
    result_folder = "result"
    os.makedirs(result_folder, exist_ok=True)
    videos = []
    for f in os.listdir(result_folder):
        if f.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            full_path = os.path.join(result_folder, f)
            videos.append({
                "filename": f,
                "url": f"/results/{f}",
                "size_mb": round(os.path.getsize(full_path) / 1024 / 1024, 2),
                "modified": os.path.getmtime(full_path)
            })
    # Sort newest first
    videos.sort(key=lambda x: x["modified"], reverse=True)
    return {"videos": videos}


def create_video_writer(output_path: str, fps: int, frame_size: tuple[int, int]):
    codec_candidates = ["avc1", "H264", "mp4v"]
    for codec in codec_candidates:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        if writer.isOpened():
            print(f"Using output codec: {codec}")
            return writer, codec
        writer.release()
    return None, None


def transcode_to_browser_mp4(input_path: str, output_path: str):
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return False, "ffmpeg not found"

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", input_path,
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path,
    ]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True)
        if completed.returncode != 0:
            return False, completed.stderr.strip()[:400]
        if not os.path.exists(output_path) or os.path.getsize(output_path) <= 0:
            return False, "transcoded file missing or empty"
        return True, "ok"
    except Exception as exc:
        return False, str(exc)

# ================= API ENDPOINT: PROCESS VIDEO =================
@app.post("/predict/video")
async def predict_video(
    file: UploadFile = File(...),
    request_id: str | None = Header(default=None, alias="X-Request-ID")
):
    """
    Processes video synchronously with streaming progress updates.
    Returns the processed video file.
    """
    RESULT_FOLDER = "result"
    os.makedirs(RESULT_FOLDER, exist_ok=True)

    job_id = request_id or uuid.uuid4().hex
    JOB_PROGRESS[job_id] = {
        "request_id": job_id,
        "status": "upload_received",
        "frame": 0,
        "total_frames": 0,
        "percent": 0,
        "message": "Upload received"
    }

    # Save uploaded video temporarily
    temp_input_path = f"temp_{uuid.uuid4().hex}.mp4"
    with open(temp_input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Output path in 'result' folder
    temp_output_path = os.path.join(RESULT_FOLDER, f"segmented_{uuid.uuid4().hex}.mp4")
    browser_output_path = os.path.join(RESULT_FOLDER, f"segmented_browser_{uuid.uuid4().hex}.mp4")

    try:
        # Open video
        cap = cv2.VideoCapture(temp_input_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Failed to open video file")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_width = frame_width if frame_width % 2 == 0 else frame_width - 1
        output_height = frame_height if frame_height % 2 == 0 else frame_height - 1
        output_width = max(output_width, 2)
        output_height = max(output_height, 2)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = max(total_frames, 1)

        JOB_PROGRESS[job_id].update({
            "status": "processing",
            "frame": 0,
            "total_frames": total_frames,
            "percent": 0,
            "message": "Processing video frames"
        })
        
        print(f"Processing video: {total_frames} frames @ {fps}fps, {frame_width}x{frame_height}")
        if output_width != frame_width or output_height != frame_height:
            print(f"Adjusting output resolution for browser playback: {output_width}x{output_height}")

        out_video, active_codec = create_video_writer(temp_output_path, fps, (output_width, output_height))
        if out_video is None:
            raise HTTPException(status_code=500, detail="Failed to initialize video encoder for output")

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                features = backbone.forward_features(input_tensor)["x_norm_patchtokens"]
                logits = model(features)
                logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
                pred = torch.argmax(logits, dim=1)[0].cpu().numpy()

            # Create colored overlay
            colored_mask = mask_to_color(pred)
            colored_mask = cv2.resize(colored_mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
            overlay = cv2.addWeighted(frame, 1 - ALPHA, colored_mask, ALPHA, 0)
            if output_width != frame_width or output_height != frame_height:
                overlay = cv2.resize(overlay, (output_width, output_height), interpolation=cv2.INTER_AREA)
            
            # Write frame
            out_video.write(overlay)

            frame_count += 1
            if frame_count % 10 == 0:
                progress = int((frame_count / total_frames) * 100)
                print(f"  Frame {frame_count}/{total_frames} ({progress}%)")
                JOB_PROGRESS[job_id].update({
                    "status": "processing",
                    "frame": frame_count,
                    "total_frames": total_frames,
                    "percent": progress,
                    "message": f"Frame {frame_count}/{total_frames} ({progress}%)"
                })

        # Properly release resources
        cap.release()
        out_video.release()

        output_size = os.path.getsize(temp_output_path) if os.path.exists(temp_output_path) else 0
        if output_size <= 0:
            raise HTTPException(status_code=500, detail="Output video was empty after processing")

        response_path = temp_output_path
        transcoded, transcode_note = transcode_to_browser_mp4(temp_output_path, browser_output_path)
        if transcoded:
            response_path = browser_output_path
            try:
                os.remove(temp_output_path)
            except OSError:
                pass
            print(f"✓ Browser transcode complete: {browser_output_path}")
        else:
            print(f"⚠ Browser transcode skipped: {transcode_note}")
        
        final_size = os.path.getsize(response_path) if os.path.exists(response_path) else 0
        print(f"✓ Processing complete: {response_path} (codec={active_codec}, size={final_size} bytes)")

        JOB_PROGRESS[job_id].update({
            "status": "completed",
            "frame": total_frames,
            "total_frames": total_frames,
            "percent": 100,
            "message": "Processing complete"
        })

        # Delete temp input
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)

        # Return the processed video file
        return FileResponse(response_path, media_type="video/mp4", filename="segmented_video.mp4")

    except Exception as e:
        print(f"Error processing video: {str(e)}")
        JOB_PROGRESS[job_id].update({
            "status": "error",
            "message": str(e)
        })
        # Cleanup
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        if os.path.exists(browser_output_path):
            os.remove(browser_output_path)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/progress/{request_id}")
async def predict_progress(request_id: str):
    if request_id not in JOB_PROGRESS:
        raise HTTPException(status_code=404, detail="Progress not found for request")
    return JOB_PROGRESS[request_id]


def run_final_model_script(input_path: str, output_path: str, job_id: str = None):
    import re
    script_path = BASE_DIR / "test_segmentation.py"
    if not script_path.exists():
        return False, f"Missing script: {script_path}"

    venv_python = BASE_DIR / "env" / "Scripts" / "python.exe"
    python_bin = str(venv_python if venv_python.exists() else Path(sys.executable))
    input_abs = str(Path(input_path).resolve())
    output_abs = str(Path(output_path).resolve())

    cmd = [
        python_bin,
        "-u",  # Unbuffered output for real-time progress
        str(script_path),
        "--input",
        input_abs,
        "--output",
        output_abs,
    ]
    print(f"[FINAL] Launching subprocess: {' '.join(cmd)}")

    try:
        # Stream stdout to parse frame progress in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(BASE_DIR)
        )

        stdout_lines = []
        
        # Read stdout line by line and parse frame progress
        while True:
            line = process.stdout.readline()
            if line == "" and process.poll() is not None:
                break
            if not line:
                continue
            stdout_lines.append(line)
            print(line, end="")
            
            # Parse "Frame X/Y (Z%)" progress lines
            match = re.search(r'Frame\s+(\d+)/(\d+)\s+\((\d+)%\)', line)
            if match and job_id:
                current = int(match.group(1))
                total = int(match.group(2))
                percent = int(match.group(3))
                JOB_PROGRESS[job_id].update({
                    "status": "processing",
                    "frame": current,
                    "total_frames": total,
                    "percent": percent,
                    "message": f"Processing frame {current}/{total}"
                })

        # Wait for process to complete
        process.wait()
        
        if process.returncode != 0:
            stdout_text = ''.join(stdout_lines)
            tail = (stdout_text or "Unknown error")[-1200:]
            return False, tail
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


@app.post("/predict/video/final")
async def predict_video_final(
    file: UploadFile = File(...),
    request_id: str | None = Header(default=None, alias="X-Request-ID")
):
    """
    Processes video using the router-based test_segmentation.py subprocess.
    Uses SegFormer + DINO with gradient-based model selection.
    """
    RESULT_FOLDER = "result"
    os.makedirs(RESULT_FOLDER, exist_ok=True)

    job_id = request_id or uuid.uuid4().hex
    print(f"[FINAL] Request received: job_id={job_id}, filename={file.filename}")
    JOB_PROGRESS[job_id] = {
        "request_id": job_id,
        "status": "upload_received",
        "frame": 0,
        "total_frames": 0,
        "percent": 0,
        "message": "Upload received"
    }

    # Save uploaded video temporarily
    temp_input_path = f"temp_{uuid.uuid4().hex}.mp4"
    with open(temp_input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        uploaded_size = os.path.getsize(temp_input_path)
        print(f"[FINAL] Upload saved: {temp_input_path} ({uploaded_size} bytes)")
    except OSError:
        print(f"[FINAL] Upload saved: {temp_input_path}")

    # Output path in 'result' folder
    temp_output_path = os.path.join(RESULT_FOLDER, f"segmented_{uuid.uuid4().hex}.mp4")
    browser_output_path = os.path.join(RESULT_FOLDER, f"segmented_browser_{uuid.uuid4().hex}.mp4")

    try:
        JOB_PROGRESS[job_id].update({
            "status": "processing",
            "message": "Starting router-based processing"
        })

        # Run the subprocess-based router model
        success, message = run_final_model_script(temp_input_path, temp_output_path, job_id)

        if not success:
            raise HTTPException(status_code=500, detail=f"Router processing failed: {message}")

        if not os.path.exists(temp_output_path) or os.path.getsize(temp_output_path) == 0:
            raise HTTPException(status_code=500, detail="Output video not generated")

        # Transcode for browser compatibility (best effort)
        JOB_PROGRESS[job_id].update({
            "status": "transcoding",
            "message": "Transcoding for browser playback"
        })

        transcode_ok, transcode_msg = transcode_to_browser_mp4(temp_output_path, browser_output_path)
        if transcode_ok and os.path.exists(browser_output_path):
            final_path = browser_output_path
            try:
                os.remove(temp_output_path)
            except OSError:
                pass
        else:
            print(f"⚠ Browser transcode skipped in /predict/video/final: {transcode_msg}")
            final_path = temp_output_path

        JOB_PROGRESS[job_id].update({
            "status": "completed",
            "percent": 100,
            "message": "Processing complete"
        })

        return FileResponse(
            final_path,
            media_type="video/mp4",
            headers={"X-Request-ID": job_id}
        )

    except HTTPException:
        raise
    except Exception as e:
        JOB_PROGRESS[job_id].update({
            "status": "error",
            "message": str(e)
        })
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp input
        if os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except:
                pass