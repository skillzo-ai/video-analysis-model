"""
Lightweight player appearance embedding (body patch) for Re-ID — usable without the video pipeline.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


class PlayerReID:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        self.model = models.mobilenet_v3_small(weights=weights).to(self.device).eval()
        self.model.classifier = nn.Identity()
        self.preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((128, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @torch.no_grad()
    def extract_embedding(self, crop: np.ndarray) -> np.ndarray | None:
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img_t = self.preprocess(crop_rgb).unsqueeze(0).to(self.device)
        embedding = self.model(img_t).cpu().numpy().flatten()
        return embedding / (np.linalg.norm(embedding) + 1e-6)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))
