"""
Video Segmentation with Dual-Model Router (SegFormer + DINO)
Routes frames to SegFormer (fast) or DINO (complex edges) based on Sobel gradient
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from transformers import SegformerForSemanticSegmentation

os.environ["XFORMERS_DISABLED"] = "1"
os.environ["DINO_DISABLE_XFORMERS"] = "1"

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "result"

SEGFORMER_CKPT = MODELS_DIR / "segformer_head.pth"
DINO_HEAD_CKPT = MODELS_DIR / "segmentation_head_best.pth"

N_CLASSES = 10
ALPHA_DEFAULT = 0.5
DINO_PERCENTILE_DEFAULT = 50

COLOR_PALETTE = np.array([
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


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i in range(N_CLASSES):
        color_mask[mask == i] = COLOR_PALETTE[i]
    return color_mask


class MultiScaleDinoV2(nn.Module):
    def __init__(self, backbone, hook_block_ids=(5, 11, 17, 23)):
        super().__init__()
        self.backbone = backbone
        self.hook_block_ids = hook_block_ids
        self._features = {}
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for block_id in self.hook_block_ids:
            block = self.backbone.blocks[block_id]
            handle = block.register_forward_hook(self._make_hook(block_id))
            self._hooks.append(handle)

    def _make_hook(self, block_id):
        def hook(module, input_tensor, output):
            self._features[block_id] = output[:, 1:, :]

        return hook

    def forward(self, x):
        self._features.clear()
        _ = self.backbone.forward_features(x)
        return [self._features[b] for b in self.hook_block_ids]


class MultiScaleSegHead(nn.Module):
    def __init__(self, in_channels, num_scales, out_channels, hidden=256):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels, hidden),
                nn.GELU(),
            )
            for _ in range(num_scales)
        ])

        self.fusion = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(16, hidden),
            nn.GELU(),
        )

        # Double-conv upsampling blocks to match checkpoint structure
        # Each block: Upsample -> Conv -> GN -> GELU -> Conv -> GN -> GELU
        def double_conv_up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.GroupNorm(16, out_ch),
                nn.GELU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.GroupNorm(16, out_ch),
                nn.GELU(),
            )

        # Checkpoint structure: 256 -> 128 -> 64 -> 32
        self.up1 = double_conv_up_block(hidden, hidden // 2)      # 256 -> 128
        self.up2 = double_conv_up_block(hidden // 2, hidden // 4) # 128 -> 64
        self.up3 = double_conv_up_block(hidden // 4, hidden // 8) # 64 -> 32

        self.classifier = nn.Conv2d(hidden // 8, out_channels, 1)  # 32 -> 10

    def forward(self, multi_scale_feats):
        B = multi_scale_feats[0].size(0)
        L = multi_scale_feats[0].size(1)
        H_patch = W_patch = int(L ** 0.5)

        fused = []
        for feat, proj in zip(multi_scale_feats, self.projections):
            x = proj(feat)
            x = x.permute(0, 2, 1).reshape(B, -1, H_patch, W_patch)
            fused.append(x)

        x = torch.stack(fused, dim=0).mean(dim=0)
        x = self.fusion(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return self.classifier(x)


class GradientRouter:
    def __init__(self, threshold: float):
        self.threshold = threshold

    def should_use_dino(self, img_rgb: np.ndarray) -> bool:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(dx * dx + dy * dy)
        score = mag.mean()
        return score >= self.threshold


def load_segformer(device):
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        num_labels=N_CLASSES,
        ignore_mismatched_sizes=True,
    )
    ckpt = torch.load(str(SEGFORMER_CKPT), map_location=device, weights_only=False)
    model.load_state_dict(ckpt)
    model.eval().to(device)
    return model


def load_dino(device):
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    backbone.eval().to(device)
    
    ms_dino = MultiScaleDinoV2(backbone)
    ms_dino.eval().to(device)

    ckpt = torch.load(str(DINO_HEAD_CKPT), map_location=device, weights_only=False)
    
    # Handle wrapped checkpoint structure (epoch, classifier, backbone_partial, val_iou)
    if "classifier" in ckpt and isinstance(ckpt["classifier"], dict):
        classifier_state = ckpt["classifier"]
    else:
        classifier_state = ckpt
    
    # Determine embed_dim from the first projection layer
    if "projections.0.0.weight" in classifier_state:
        embed_dim = classifier_state["projections.0.0.weight"].shape[1]
    else:
        # Fallback: assume standard vitl14 embed_dim
        embed_dim = 1024
        print(f"[Warning] Could not find projections in checkpoint, using default embed_dim={embed_dim}")
    
    head = MultiScaleSegHead(in_channels=embed_dim, num_scales=4, out_channels=N_CLASSES)
    head.load_state_dict(classifier_state, strict=False)
    head.eval().to(device)
    
    return ms_dino, head


def compute_video_threshold(cap, dino_percentile: float):
    scores = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_count = min(frame_count, 50)
    step = max(1, frame_count // sample_count)
    
    print(f"[Auto-Threshold] Sampling {sample_count} frames from video...")
    
    for i in range(0, frame_count, step):
        if len(scores) >= sample_count:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(dx * dx + dy * dy)
        scores.append(mag.mean())
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    threshold = np.percentile(scores, dino_percentile)
    print(f"[Router] Sobel threshold={threshold:.2f} (target top {dino_percentile:.0f}% -> DINO)")
    return threshold


def create_video_writer(output_path, fps, width, height):
    codecs = ["avc1", "H264", "mp4v"]
    for codec in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if writer.isOpened():
            print(f"Using output codec: {codec}")
            return writer, codec
        writer.release()
    return None, None


def process_video(input_path: str, output_path: str, device, dino_percentile=50, alpha=0.5, max_frames=0):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Ensure even dimensions for video codec
    output_width = frame_width if frame_width % 2 == 0 else frame_width - 1
    output_height = frame_height if frame_height % 2 == 0 else frame_height - 1

    print(f"Processing: {total_frames} frames @ {fps}fps, {frame_width}x{frame_height}")
    if output_width != frame_width or output_height != frame_height:
        print(f"Adjusted output size for browser compatibility: {output_width}x{output_height}")

    # Compute routing threshold
    threshold = compute_video_threshold(cap, dino_percentile)
    router = GradientRouter(threshold)

    # Load models
    print("Loading models...")
    segformer_model = load_segformer(device)
    dino_model, dino_head = load_dino(device)

    # Transforms
    tf_fast = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tf_dino = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    writer, active_codec = create_video_writer(output_path, fps, output_width, output_height)
    if writer is None:
        raise RuntimeError("Failed to initialize video writer")

    frame_count = 0
    routed_fast = 0
    routed_dino = 0
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Route decision
        use_dino = router.should_use_dino(rgb)

        if use_dino:
            routed_dino += 1
            img_t = tf_dino(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                feats = dino_model(img_t)
                logits = dino_head(feats)
                logits_up = F.interpolate(logits, size=(frame_height, frame_width), mode="bilinear", align_corners=False)
                pred = torch.argmax(logits_up, dim=1)[0].cpu().numpy()
        else:
            routed_fast += 1
            img_t = tf_fast(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = segformer_model(pixel_values=img_t).logits
                logits_up = F.interpolate(logits, size=(frame_height, frame_width), mode="bilinear", align_corners=False)
                pred = torch.argmax(logits_up, dim=1)[0].cpu().numpy()

        # Create overlay
        colored_mask = mask_to_color(pred.astype(np.uint8))
        overlay = cv2.addWeighted(frame, 1 - alpha, colored_mask, alpha, 0)

        # Resize if needed
        if output_width != frame_width or output_height != frame_height:
            overlay = cv2.resize(overlay, (output_width, output_height), interpolation=cv2.INTER_AREA)

        writer.write(overlay)

        frame_count += 1
        if frame_count % 10 == 0:
            progress = int((frame_count / total_frames) * 100)
            print(f"  Frame {frame_count}/{total_frames} ({progress}%)", flush=True)

        if max_frames > 0 and frame_count >= max_frames:
            print(f"Reached max_frames={max_frames}, stopping early for test run.")
            break

    cap.release()
    writer.release()

    elapsed = time.time() - start
    print(f"Done. Routed -> SegFormer: {routed_fast}, DINO: {routed_dino}")
    print(f"Average latency/frame: {(elapsed / max(frame_count, 1)) * 1000:.2f} ms")
    print(f"Output saved: {output_path} (codec={active_codec})")


def parse_args():
    parser = argparse.ArgumentParser(description="Video segmentation test runner (SegFormer + DINO router)")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", default=None, help="Output video path (default: result/segmented_<timestamp>.mp4)")
    parser.add_argument("--dino-percentile", type=float, default=DINO_PERCENTILE_DEFAULT, help="Top percentile routed to DINO")
    parser.add_argument("--alpha", type=float, default=ALPHA_DEFAULT, help="Overlay transparency (0-1)")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to process (0=all)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        RESULT_DIR.mkdir(exist_ok=True)
        output_path = RESULT_DIR / f"segmented_{int(time.time())}.mp4"

    process_video(
        str(input_path),
        str(output_path),
        device,
        dino_percentile=args.dino_percentile,
        alpha=args.alpha,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
