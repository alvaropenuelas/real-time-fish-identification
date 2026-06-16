import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import build_model
from src.species_map import DISPLAY_NAMES

# VERIFY: confirm num_classes=33 matches model.pt
NUM_CLASSES = 33
HARD_CONF_FLOOR = 0.3

_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class FishClassifier:
    def __init__(self, weights_path: str, conf_threshold: float = 0.5, device: str = "cpu"):
        self.threshold = conf_threshold
        self.device = torch.device(device)

        # ASSUMPTION: weights_path is a state_dict saved via torch.save(model.state_dict(), path)
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model = build_model(num_classes=NUM_CLASSES)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        getattr(self.model, "eval")()  # nn.Module.eval(), not Python built-in

        # ASSUMPTION: classes.json lives next to weights_path or at outputs/classes.json
        project_root = Path(__file__).parent.parent
        weights_dir = Path(weights_path).parent
        for candidate in [weights_dir / "classes.json", project_root / "outputs" / "classes.json"]:
            if candidate.exists():
                with open(candidate) as f:
                    self.class_names = json.load(f)
                break
        else:
            raise FileNotFoundError(
                f"classes.json not found in {weights_dir} or outputs/"
            )

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        # Input: BGR frame (H×W×3, uint8) from OpenCV → normalized CHW tensor
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        return _TRANSFORMS(img)

    def predict_probs_batch(self, frames: list[np.ndarray]) -> np.ndarray:
        """Softmax probabilities [N, NUM_CLASSES] for N BGR crops in ONE forward pass."""
        if not frames:
            return np.empty((0, NUM_CLASSES), dtype=np.float32)
        batch = torch.stack([self._preprocess(f) for f in frames]).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(batch), dim=1)
        return probs.cpu().numpy()

    def decode(self, probs: np.ndarray) -> dict | None:
        """Turn a softmax vector into the same top3 dict predict() returns.
        Applies the confidence threshold and the hard floor identically."""
        top_idxs = probs.argsort()[::-1][:3]
        top3 = []
        for idx in top_idxs:
            conf = float(probs[idx])
            if conf < self.threshold:
                break
            # Hard floor: never display low-confidence predictions
            if conf < HARD_CONF_FLOOR:
                break
            folder_name = self.class_names[idx]
            label = DISPLAY_NAMES.get(folder_name, folder_name.replace("_", " "))
            top3.append({"label": label, "confidence": conf})
        if not top3:
            return None
        return {"top3": top3}

    def predict(self, frame: np.ndarray) -> dict | None:
        # Single-crop convenience wrapper over the batch path.
        return self.decode(self.predict_probs_batch([frame])[0])
