# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# import cv2
# import numpy as np
# from PIL import Image

# # ================= SETTINGS =================
# video_path = "input_video.mp4"
# output_path = "initial_latest_checkpoint_segmented_video.mp4"
# model_path = "initial_latest_checkpoint.pth"
# alpha = 0.5  # overlay transparency

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)

# # ================= SEGMENTATION HEAD =================
# class SegmentationHeadConvNeXt(nn.Module):
#     def __init__(self, in_channels, out_channels, tokenW, tokenH):
#         super().__init__()
#         self.H, self.W = tokenH, tokenW

#         self.stem = nn.Sequential(
#             nn.Conv2d(in_channels, 128, kernel_size=7, padding=3),
#             nn.GELU()
#         )

#         self.block = nn.Sequential(
#             nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128),
#             nn.GELU(),
#             nn.Conv2d(128, 128, kernel_size=1),
#             nn.GELU(),
#         )

#         self.classifier = nn.Conv2d(128, out_channels, 1)

#     def forward(self, x):
#         B, N, C = x.shape
#         x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
#         x = self.stem(x)
#         x = self.block(x)
#         return self.classifier(x)

# # ================= LOAD DINOv2 BACKBONE =================
# print("Loading DINOv2 backbone...")
# backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
# backbone.eval().to(device)

# # ================= MATCH TRAINING SIZE =================
# w = int(((960 / 2) // 14) * 14)
# h = int(((540 / 2) // 14) * 14)

# # Get embedding dimension automatically
# dummy = torch.randn(1, 3, h, w).to(device)
# with torch.no_grad():
#     out = backbone.forward_features(dummy)["x_norm_patchtokens"]
# embed_dim = out.shape[2]

# print("Embedding dimension:", embed_dim)

# # ================= LOAD TRAINED HEAD =================
# model = SegmentationHeadConvNeXt(
#     in_channels=embed_dim,
#     out_channels=10,
#     tokenW=w // 14,
#     tokenH=h // 14
# ).to(device)

# model.load_state_dict(torch.load(model_path, map_location=device))
# model.eval()

# print("Segmentation head loaded.")

# # ================= TRANSFORMS =================
# transform = transforms.Compose([
#     transforms.Resize((h, w)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# # ================= COLOR PALETTE =================
# color_palette = np.array([
#     [0, 0, 0],          # background
#     [34, 139, 34],      # trees
#     [0, 255, 0],        # lush bushes
#     [210, 180, 140],    # dry grass
#     [139, 90, 43],      # dry bushes
#     [128, 128, 0],      # ground clutter
#     [139, 69, 19],      # logs
#     [128, 128, 128],    # rocks
#     [160, 82, 45],      # landscape
#     [135, 206, 235],    # sky
# ], dtype=np.uint8)

# def mask_to_color(mask):
#     h, w = mask.shape
#     color_mask = np.zeros((h, w, 3), dtype=np.uint8)
#     for i in range(10):
#         color_mask[mask == i] = color_palette[i]
#     return color_mask

# # ================= PROCESS VIDEO =================
# cap = cv2.VideoCapture(video_path)

# if not cap.isOpened():
#     print("Error opening video file.")
#     exit()

# fps = int(cap.get(cv2.CAP_PROP_FPS))
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# out_video = cv2.VideoWriter(output_path, fourcc, fps,
#                             (frame_width, frame_height))

# print("Processing video...")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     pil_img = Image.fromarray(rgb)

#     input_tensor = transform(pil_img).unsqueeze(0).to(device)

#     with torch.no_grad():
#         features = backbone.forward_features(input_tensor)["x_norm_patchtokens"]
#         logits = model(features)
#         logits = F.interpolate(logits,
#                                size=(h, w),
#                                mode="bilinear",
#                                align_corners=False)
#         pred = torch.argmax(logits, dim=1)[0].cpu().numpy()

#     colored_mask = mask_to_color(pred)

#     # Resize back to original resolution
#     colored_mask = cv2.resize(colored_mask,
#                               (frame_width, frame_height),
#                               interpolation=cv2.INTER_NEAREST)

#     overlay = cv2.addWeighted(frame, 1 - alpha,
#                               colored_mask, alpha, 0)

#     out_video.write(overlay)

# cap.release()
# out_video.release()

# print("Done! Segmented video saved to:", output_path)# import torch
# # import torch.nn.functional as F
# # import torchvision.transforms as transforms
# # import cv2
# # import numpy as np
# # from PIL import Image

# # # ===== SETTINGS =====
# # video_path = "input_video.mp4"
# # output_path = "segmented_video.mp4"
# # model_path = "segmentation_head.pth"
# # alpha = 0.5   # transparency

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # # ===== Load Backbone =====
# # backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
# # backbone.eval().to(device)

# # # ===== Load Segmentation Head =====
# # from your_training_script import SegmentationHeadConvNeXt  # or paste class here

# # # Must match training size
# # w = int(((960 / 2) // 14) * 14)
# # h = int(((540 / 2) // 14) * 14)

# # # Get embedding size
# # dummy = torch.randn(1,3,h,w).to(device)
# # with torch.no_grad():
# #     out = backbone.forward_features(dummy)["x_norm_patchtokens"]
# # embed_dim = out.shape[2]

# # model = SegmentationHeadConvNeXt(
# #     in_channels=embed_dim,
# #     out_channels=10,
# #     tokenW=w//14,
# #     tokenH=h//14
# # ).to(device)

# # model.load_state_dict(torch.load(model_path, map_location=device))
# # model.eval()

# # # ===== Transforms =====
# # transform = transforms.Compose([
# #     transforms.Resize((h, w)),
# #     transforms.ToTensor(),
# #     transforms.Normalize(mean=[0.485,0.456,0.406],
# #                          std=[0.229,0.224,0.225])
# # ])

# # # ===== Color Palette =====
# # color_palette = np.array([
# #     [0,0,0],
# #     [34,139,34],
# #     [0,255,0],
# #     [210,180,140],
# #     [139,90,43],
# #     [128,128,0],
# #     [139,69,19],
# #     [128,128,128],
# #     [160,82,45],
# #     [135,206,235],
# # ], dtype=np.uint8)

# # def mask_to_color(mask):
# #     h, w = mask.shape
# #     color_mask = np.zeros((h, w, 3), dtype=np.uint8)
# #     for i in range(10):
# #         color_mask[mask == i] = color_palette[i]
# #     return color_mask

# # # ===== Process Video =====
# # cap = cv2.VideoCapture(video_path)
# # fps = int(cap.get(cv2.CAP_PROP_FPS))
# # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# # out_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# # print("Processing video...")

# # while True:
# #     ret, frame = cap.read()
# #     if not ret:
# #         break

# #     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #     pil_img = Image.fromarray(rgb)
# #     input_tensor = transform(pil_img).unsqueeze(0).to(device)

# #     with torch.no_grad():
# #         features = backbone.forward_features(input_tensor)["x_norm_patchtokens"]
# #         logits = model(features)
# #         logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
# #         pred = torch.argmax(logits, dim=1)[0].cpu().numpy()

# #     colored_mask = mask_to_color(pred)

# #     # Resize mask back to original frame size
# #     colored_mask = cv2.resize(colored_mask, (frame_width, frame_height))

# #     overlay = cv2.addWeighted(frame, 1-alpha, colored_mask, alpha, 0)

# #     out_video.write(overlay)

# # cap.release()
# # out_video.release()

# # print("Done! Segmented video saved.")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image

# ================= SETTINGS =================
video_path = r"D:\Offroad_Segmentation_Scripts\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\vids\input_video.mp4"
output_path = "randomCrop_shaun_input_video.mp4"
model_path = r"C:\Users\hp\Desktop\ai_FRONTEND\backend\models\randomCrop_shaun.pth"
alpha = 0.5  # overlay transparency

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


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
backbone.eval().to(device)

# ================= MATCH TRAINING SIZE =================
w = int(((960 / 2) // 14) * 14)
h = int(((540 / 2) // 14) * 14)

dummy = torch.randn(1, 3, h, w).to(device)
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
).to(device)

print("Loading checkpoint...")

checkpoint = torch.load(
    model_path,
    map_location=device,
    weights_only=False  # required for PyTorch 2.6+
)

# Extract model_state_dict
if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

# Remove _orig_mod. prefix if present
clean_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("_orig_mod."):
        clean_state_dict[k.replace("_orig_mod.", "")] = v
    else:
        clean_state_dict[k] = v

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
    [0, 0, 0],          # background
    [34, 139, 34],      # trees
    [0, 255, 0],        # lush bushes
    [210, 180, 140],    # dry grass
    [139, 90, 43],      # dry bushes
    [128, 128, 0],      # ground clutter
    [139, 69, 19],      # logs
    [128, 128, 128],    # rocks
    [160, 82, 45],      # landscape
    [135, 206, 235],    # sky
], dtype=np.uint8)


def mask_to_color(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(10):
        color_mask[mask == i] = color_palette[i]
    return color_mask


# ================= PROCESS VIDEO =================
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_video = cv2.VideoWriter(output_path, fourcc, fps,
                            (frame_width, frame_height))

print("Processing video...")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        features = backbone.forward_features(input_tensor)["x_norm_patchtokens"]
        logits = model(features)
        logits = F.interpolate(
            logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy()

    colored_mask = mask_to_color(pred)

    colored_mask = cv2.resize(
        colored_mask,
        (frame_width, frame_height),
        interpolation=cv2.INTER_NEAREST
    )

    overlay = cv2.addWeighted(frame, 1 - alpha,
                              colored_mask, alpha, 0)

    out_video.write(overlay)

    frame_count += 1
    if frame_count % 10 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
out_video.release()

print("Done!")
print("Saved segmented video to:", output_path)