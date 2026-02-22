"""
Dynamic Cascade Inference Router
Routes images to SegFormer MiT-B0 (Fast) or DINOv2 Large (Complex)
based on Edge Density (OpenCV Sobel Gradient).

Threshold is AUTO-COMPUTED via a quick pre-pass over all images:
  DINO_PERCENTILE = 50  →  top 50% of images (by edge complexity) → DINOv2
                           bottom 50%                              → SegFormer
Adjust DINO_PERCENTILE to taste (e.g. 30 = only 30% go to DINOv2).
"""
import os
os.environ["XFORMERS_DISABLED"] = "1"
os.environ["DINO_DISABLE_XFORMERS"] = "1"
os.environ["USE_FLASH_ATTENTION"] = "1"

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import cv2
import time
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import v2
from transformers import SegformerForSemanticSegmentation

# ============================================================================
# Config — edit these if paths change
# ============================================================================
DATA_DIR        = "/home/dwayne/SPIT/Offroad_Segmentation_testImages"
SEGFORMER_CKPT  = "/home/dwayne/SPIT/src/Segformal/segformer_head.pth"
DINO_HEAD_CKPT  = "/home/dwayne/SPIT/src/RandomCrop/segmentation_head_best.pth"
# What % of images should be routed to DINOv2 (the heavy model)?
DINO_PERCENTILE = 50

# ============================================================================
# Class map + palette
# ============================================================================
VALUE_MAP   = {0: 0, 100: 1, 200: 2, 300: 3, 500: 4, 550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9}
CLASS_NAMES = ["background", "tree", "grass", "sand", "dirt", "olive", "brown", "gray", "sienna", "sky"]
N_CLASSES   = len(VALUE_MAP)

COLOR_PALETTE = np.array([
    [0, 0, 0], [34, 139, 34], [0, 255, 0], [210, 180, 140], [139, 90, 43],
    [128, 128, 0], [139, 69, 19], [128, 128, 128], [160, 82, 45], [135, 206, 235]
], dtype=np.uint8)

_RAW_TO_CLASS = np.full(10001, 255, dtype=np.uint8)
for raw_val, cls_idx in VALUE_MAP.items():
    _RAW_TO_CLASS[raw_val] = cls_idx

def remap_gt(mask_u16: np.ndarray) -> np.ndarray:
    return _RAW_TO_CLASS[np.clip(mask_u16.astype(np.int32), 0, 10000)]

def mask_to_color(mask: np.ndarray) -> np.ndarray:
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c in range(N_CLASSES):
        color[mask == c] = COLOR_PALETTE[c]
    return color

# ============================================================================
# Metrics
# ============================================================================
class SegmentationMetrics:
    def __init__(self, num_classes: int):
        self.C  = num_classes
        self.cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, gt: np.ndarray):
        valid = (gt >= 0) & (gt < self.C)
        np.add.at(self.cm, (gt[valid].astype(np.int64), pred[valid].astype(np.int64)), 1)

    def iou(self):
        out = np.full(self.C, np.nan)
        for c in range(self.C):
            tp    = self.cm[c, c]
            denom = self.cm[c].sum() + self.cm[:, c].sum() - tp
            if denom > 0: out[c] = tp / denom
        return out

    def precision(self):
        out = np.full(self.C, np.nan)
        for c in range(self.C):
            tp = self.cm[c, c]
            fp = self.cm[:, c].sum() - tp
            if tp + fp > 0: out[c] = tp / (tp + fp)
        return out

    def recall(self):
        out = np.full(self.C, np.nan)
        for c in range(self.C):
            tp = self.cm[c, c]
            fn = self.cm[c].sum() - tp
            if tp + fn > 0: out[c] = tp / (tp + fn)
        return out

    def dice(self):
        p, r = self.precision(), self.recall()
        d    = np.full(self.C, np.nan)
        for c in range(self.C):
            if not (np.isnan(p[c]) or np.isnan(r[c])) and (p[c] + r[c]) > 0:
                d[c] = 2 * p[c] * r[c] / (p[c] + r[c])
        return d

    def miou(self):
        v = self.iou(); v = v[~np.isnan(v)]
        return v.mean() if len(v) else np.nan

    def pixel_acc(self):
        t = self.cm.sum()
        return np.diag(self.cm).sum() / t if t > 0 else np.nan

    def print_report(self, title="SEGMENTATION METRICS", class_names=None):
        iou, prec, rec, dice = self.iou(), self.precision(), self.recall(), self.dice()
        names = class_names or [f"class_{i}" for i in range(self.C)]
        W = 72
        print(f"\n{'='*W}")
        print(title)
        print(f"{'='*W}")
        print(f"{'Class':<20} {'IoU':>8} {'Precision':>10} {'Recall':>8} {'Dice':>8}")
        print(f"{'-'*W}")
        for c in range(self.C):
            if self.cm[c].sum() == 0 and self.cm[:, c].sum() == 0:
                continue
            fmt = lambda v: f"{v:.4f}" if not np.isnan(v) else "   N/A"
            print(f"{names[c]:<20} {fmt(iou[c]):>8} {fmt(prec[c]):>10} {fmt(rec[c]):>8} {fmt(dice[c]):>8}")
        print(f"{'-'*W}")
        print(f"{'mIoU':<20} {self.miou():.4f}")
        print(f"{'Pixel Accuracy':<20} {self.pixel_acc():.4f}")
        print(f"{'='*W}")

# ============================================================================
# Router — Sobel edge density
# ============================================================================
class GradientRouter:
    def __init__(self, threshold: float):
        self.threshold = threshold

    def score(self, img_chw: torch.Tensor) -> float:
        img = img_chw.permute(1, 2, 0).cpu().numpy()
        if img.dtype != np.uint8:
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mag  = cv2.magnitude(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3),
                             cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
        return float(np.mean(mag))

    def route(self, img_chw: torch.Tensor):
        s = self.score(img_chw)
        return s > self.threshold, s


def compute_auto_threshold(image_dir: str, percentile: float):
    print(f"\n[Auto-Threshold] Scanning all images in {image_dir} ...")
    files  = sorted(os.listdir(image_dir))
    scores = []
    for fname in tqdm(files, desc="  Sobel pre-pass", ncols=80):
        fpath = os.path.join(image_dir, fname)
        try:
            img  = np.array(Image.open(fpath).convert("RGB"), dtype=np.uint8)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mag  = cv2.magnitude(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3),
                                 cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
            scores.append(float(np.mean(mag)))
        except Exception:
            pass

    threshold = float(np.percentile(scores, 100 - percentile))
    print(f"[Auto-Threshold] Sobel scores  min={min(scores):.1f}  "
          f"mean={np.mean(scores):.1f}  max={max(scores):.1f}")
    print(f"[Auto-Threshold] Threshold = {threshold:.2f}  "
          f"(approx {percentile:.0f}% of images will go to DINOv2)\n")
    return threshold, scores

# ============================================================================
# Dataset
# ============================================================================
class InferenceDataset(Dataset):
    GT_DIR = "Segmentation"

    def __init__(self, data_dir: str):
        self.img_dir  = os.path.join(data_dir, "Color_Images")
        self.ids      = sorted(os.listdir(self.img_dir))
        self.mask_dir = os.path.join(data_dir, self.GT_DIR)
        self.has_gt   = os.path.isdir(self.mask_dir)
        print(f"[Dataset] {len(self.ids)} images | GT masks: {'YES -> IoU will be computed' if self.has_gt else 'NO'}")

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        fid    = self.ids[idx]
        img_np = np.array(Image.open(os.path.join(self.img_dir, fid)).convert("RGB"), dtype=np.uint8)
        img_t  = torch.from_numpy(img_np).permute(2, 0, 1)   # CHW uint8

        gt = None
        if self.has_gt:
            mp = os.path.join(self.mask_dir, os.path.splitext(fid)[0] + ".png")
            if os.path.exists(mp):
                gt = remap_gt(np.array(Image.open(mp)))

        return img_t, fid, gt

# ============================================================================
# Multi-Scale DINOv2 Backbone (matches training script exactly)
# ============================================================================
class MultiScaleDinoV2(nn.Module):
    def __init__(self, backbone, hook_block_ids=(5, 11, 17, 23)):
        super().__init__()
        self.backbone       = backbone
        self.hook_block_ids = hook_block_ids
        self._features      = {}
        self._hooks         = []
        self._register_hooks()

    def _register_hooks(self):
        for block_id in self.hook_block_ids:
            block = self.backbone.blocks[block_id]
            handle = block.register_forward_hook(self._make_hook(block_id))
            self._hooks.append(handle)

    def _make_hook(self, block_id):
        def hook(module, input, output):
            self._features[block_id] = output[:, 1:, :]   # strip CLS token
        return hook

    def forward(self, x):
        self._features.clear()
        _ = self.backbone.forward_features(x)
        return [self._features[b] for b in self.hook_block_ids]

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

# ============================================================================
# Multi-Scale Segmentation Head (matches training script exactly)
# ============================================================================
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

        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.GroupNorm(16, out_ch),
                nn.GELU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.GroupNorm(16, out_ch),
                nn.GELU(),
            )

        self.up1 = up_block(hidden, hidden // 2)
        self.up2 = up_block(hidden // 2, hidden // 4)
        self.up3 = up_block(hidden // 4, hidden // 8)

        self.classifier = nn.Conv2d(hidden // 8, out_channels, 1)

    def forward(self, features, token_h, token_w, target_size):
        fused = None
        for feat, proj in zip(features, self.projections):
            B, N, C = feat.shape
            x = proj(feat)
            x = x.reshape(B, token_h, token_w, -1).permute(0, 3, 1, 2)
            fused = x if fused is None else fused + x

        x = self.fusion(fused)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return self.classifier(x)

# ============================================================================
# Main
# ============================================================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir    = os.path.join(script_dir, "cascade_predictions")
    os.makedirs(os.path.join(out_dir, "masks"),       exist_ok=True)
    os.makedirs(os.path.join(out_dir, "masks_color"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # STEP 1 — Auto-threshold pre-pass
    # ------------------------------------------------------------------
    img_dir               = os.path.join(DATA_DIR, "Color_Images")
    threshold, all_scores = compute_auto_threshold(img_dir, DINO_PERCENTILE)
    router                = GradientRouter(threshold)

    # ------------------------------------------------------------------
    # STEP 2 — Load SegFormer (fast model)
    # ------------------------------------------------------------------
    print("Loading SegFormer MiT-B0 ...")
    fast_model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0", num_labels=N_CLASSES, ignore_mismatched_sizes=True)
    fast_model.load_state_dict(torch.load(SEGFORMER_CKPT, map_location=device))
    fast_model = fast_model.to(device).eval()

    # ------------------------------------------------------------------
    # STEP 3 — Load DINOv2 Large + MultiScaleSegHead (heavy model)
    # ------------------------------------------------------------------
    print("Loading DINOv2 Large (vitl14) ...")
    raw_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", verbose=False)
    raw_backbone = raw_backbone.to(device).eval()

    ms_backbone = MultiScaleDinoV2(raw_backbone, hook_block_ids=(5, 11, 17, 23))
    ms_backbone = ms_backbone.to(device).eval()

    dino_head = MultiScaleSegHead(
        in_channels=1024,   # ViT-L embedding dim
        num_scales=4,
        out_channels=N_CLASSES,
        hidden=256,
    ).to(device)

    ckpt = torch.load(DINO_HEAD_CKPT, map_location=device, weights_only=False)
    dino_head.load_state_dict(ckpt["classifier"])
    dino_head = dino_head.eval()
    print(f"[DINO] Loaded checkpoint — epoch {ckpt['epoch']}, val_iou={ckpt['val_iou']:.4f}")

    # ------------------------------------------------------------------
    # STEP 4 — Transforms
    # ------------------------------------------------------------------
    tf_fast = v2.Compose([
        v2.Resize((544, 960), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tf_dino = v2.Compose([
        v2.Resize((546, 966), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ------------------------------------------------------------------
    # STEP 5 — Dataset + metric trackers
    # ------------------------------------------------------------------
    dataset = InferenceDataset(DATA_DIR)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False,
                         num_workers=2, pin_memory=True,
                         collate_fn=lambda b: b)

    metrics_all  = SegmentationMetrics(N_CLASSES)
    metrics_fast = SegmentationMetrics(N_CLASSES)
    metrics_dino = SegmentationMetrics(N_CLASSES)
    pred_counts  = np.zeros(N_CLASSES, dtype=np.int64)
    fast_times, dino_times = [], []
    fast_count, dino_count = 0, 0

    print("\nStarting inference ...\n")
    total_start = time.time()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Routing", ncols=90):
            img_t, fid, gt = batch[0]

            use_dino, score = router.route(img_t)
            t0 = time.time()

            if use_dino:
                dino_count += 1
                x       = tf_dino(img_t).unsqueeze(0).to(device)
                token_h = x.shape[2] // 14
                token_w = x.shape[3] // 14
                feats   = ms_backbone(x)
                logits  = dino_head(feats, token_h, token_w, target_size=x.shape[2:])
                # logits already at target_size — no extra interpolate needed
                pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
                dino_times.append(time.time() - t0)
            else:
                fast_count += 1
                x      = tf_fast(img_t).unsqueeze(0).to(device)
                logits = fast_model(pixel_values=x).logits
                logits_up = F.interpolate(logits, size=x.shape[2:],
                                          mode="bilinear", align_corners=False)
                pred = torch.argmax(logits_up, dim=1)[0].cpu().numpy().astype(np.uint8)
                fast_times.append(time.time() - t0)

            for c in range(N_CLASSES):
                pred_counts[c] += int((pred == c).sum())

            if gt is not None:
                if gt.shape != pred.shape:
                    gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
                metrics_all.update(pred, gt)
                (metrics_dino if use_dino else metrics_fast).update(pred, gt)

            stem = os.path.splitext(fid)[0]
            Image.fromarray(pred).save(
                os.path.join(out_dir, "masks", f"{stem}_pred.png"))
            cv2.imwrite(
                os.path.join(out_dir, "masks_color", f"{stem}_pred_color.png"),
                cv2.cvtColor(mask_to_color(pred), cv2.COLOR_RGB2BGR))

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------
    total   = fast_count + dino_count
    elapsed = time.time() - total_start

    print(f"\n{'='*57}")
    print("ROUTING SUMMARY & PERFORMANCE")
    print(f"{'='*57}")
    print(f"Total images           : {total}")
    print(f"-> SegFormer (fast)    : {fast_count:>5}  ({fast_count/total*100:.1f}%)")
    print(f"-> DINOv2   (heavy)   : {dino_count:>5}  ({dino_count/total*100:.1f}%)")
    print(f"Auto threshold         : {threshold:.2f}  (target: top {DINO_PERCENTILE}% -> DINOv2)")
    print(f"Sobel min/mean/max     : {min(all_scores):.1f} / {np.mean(all_scores):.1f} / {max(all_scores):.1f}")
    print()
    if fast_times: print(f"SegFormer avg latency  : {np.mean(fast_times)*1000:.1f} ms")
    if dino_times: print(f"DINOv2    avg latency  : {np.mean(dino_times)*1000:.1f} ms")
    print(f"Overall avg / frame    : {elapsed/total*1000:.1f} ms")
    print(f"{'='*57}")

    total_px = pred_counts.sum()
    print(f"\n{'='*57}")
    print("PREDICTED CLASS DISTRIBUTION")
    print(f"{'='*57}")
    print(f"{'Class':<20} {'Pixels':>12} {'%':>7}")
    print(f"{'-'*57}")
    for c in range(N_CLASSES):
        pct = pred_counts[c] / total_px * 100 if total_px else 0
        print(f"{CLASS_NAMES[c]:<20} {pred_counts[c]:>12,} {pct:>6.2f}%")
    print(f"{'='*57}")

    if dataset.has_gt:
        metrics_all.print_report("OVERALL METRICS (all frames)", CLASS_NAMES)
        if fast_count > 0:
            metrics_fast.print_report(
                f"SEGFORMER-ONLY METRICS ({fast_count} frames)", CLASS_NAMES)
        if dino_count > 0:
            metrics_dino.print_report(
                f"DINOV2-ONLY METRICS ({dino_count} frames)", CLASS_NAMES)
    else:
        print("\n[Metrics] No GT Segmentation/ folder found — skipping IoU/Dice.")


if __name__ == "__main__":
    main()
