import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
WEIGHTS = ROOT / "weights" / "model.pt"
sys.path.insert(0, str(ROOT))


class TestBuildModel(unittest.TestCase):
    """Weights-free — always runs in CI (model.pt is gitignored / absent on a fresh clone)."""

    def test_output_shape_tracks_num_classes(self):
        from src.model import build_model

        for n in (5, 33, 101):
            model = build_model(num_classes=n)
            model.eval()
            with torch.no_grad():
                out = model(torch.randn(2, 3, 224, 224))
            self.assertEqual(out.shape, (2, n))


@unittest.skipIf(not WEIGHTS.exists(), f"Skipping: weights not found at {WEIGHTS}")
class TestFishClassifier(unittest.TestCase):
    def test_predict_black_frame(self):
        from src.classifier import FishClassifier

        clf = FishClassifier(str(WEIGHTS), conf_threshold=0.5, device="cpu")
        frame = np.zeros((224, 224, 3), dtype=np.uint8)  # black BGR frame
        result = clf.predict(frame)

        # predict() returns None (nothing above threshold) or {"top3": [{label, confidence}, ...]}
        if result is None:
            return
        self.assertIn("top3", result)
        self.assertGreaterEqual(len(result["top3"]), 1)
        for pred in result["top3"]:
            self.assertIsInstance(pred["label"], str)
            self.assertIsInstance(pred["confidence"], float)
            self.assertGreaterEqual(pred["confidence"], 0.0)
            self.assertLessEqual(pred["confidence"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
