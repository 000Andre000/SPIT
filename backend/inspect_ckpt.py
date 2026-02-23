import torch

ckpt = torch.load(r'models\segmentation_head_best.pth', map_location='cpu', weights_only=False)

print('Top-level keys:', list(ckpt.keys()))
print()

if 'classifier' in ckpt:
    print('Classifier keys (first 15):')
    for i, k in enumerate(list(ckpt['classifier'].keys())[:15]):
        v = ckpt['classifier'][k]
        shape = v.shape if hasattr(v, 'shape') else type(v).__name__
        print(f'  {k}: {shape}')
    print()

if 'backbone_partial' in ckpt:
    print('Backbone partial keys (first 5):')
    for i, k in enumerate(list(ckpt['backbone_partial'].keys())[:5]):
        v = ckpt['backbone_partial'][k]
        shape = v.shape if hasattr(v, 'shape') else type(v).__name__
        print(f'  {k}: {shape}')
