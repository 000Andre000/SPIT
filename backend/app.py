from fastapi import FastAPI, File, UploadFile
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
from io import BytesIO

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
MODEL_PATH = r"C:\Users\hp\Desktop\ai_FRONTEND\backend\models\randomCrop_shaun.pth"
ALPHA = 0.5  # transparency overlay

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

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

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
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

# ================= API ENDPOINT: PROCESS VIDEO =================
@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    """
    Processes video synchronously with streaming progress updates.
    Returns the processed video file.
    """
    RESULT_FOLDER = "result"
    os.makedirs(RESULT_FOLDER, exist_ok=True)

    # Save uploaded video temporarily
    temp_input_path = f"temp_{uuid.uuid4().hex}.mp4"
    with open(temp_input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Output path in 'result' folder
    temp_output_path = os.path.join(RESULT_FOLDER, f"segmented_{uuid.uuid4().hex}.mp4")

    try:
        # Open video
        cap = cv2.VideoCapture(temp_input_path)
        if not cap.isOpened():
            return {"error": "Failed to open video file"}

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {total_frames} frames @ {fps}fps, {frame_width}x{frame_height}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_video = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width, frame_height))

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
            
            # Write frame
            out_video.write(overlay)

            frame_count += 1
            if frame_count % 10 == 0:
                progress = int((frame_count / total_frames) * 100)
                print(f"  Frame {frame_count}/{total_frames} ({progress}%)")

        # Properly release resources
        cap.release()
        out_video.release()
        
        print(f"✓ Processing complete: {temp_output_path}")

        # Delete temp input
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)

        # Return the processed video file
        return FileResponse(temp_output_path, media_type="video/mp4", filename="segmented_video.mp4")

    except Exception as e:
        print(f"Error processing video: {str(e)}")
        # Cleanup
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        return {"error": str(e)}