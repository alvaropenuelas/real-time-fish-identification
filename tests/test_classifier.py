import sys
import unittest
from pathlib import Path

import numpy as np

WEIGHTS = Path(__file__).parent.parent / "weights" / "model.pt"

sys.path.insert(0, str(Path(__file__).parent.parent))


@unittest.skipIf(not WEIGHTS.exists(), f"Skipping: weights not found at {WEIGHTS}")
class TestFishClassifier(unittest.TestCase):
    def test_predict_black_frame(self):
        from src.classifier import FishClassifier

        clf = FishClassifier(str(WEIGHTS), conf_threshold=0.5, device="cpu")
        frame = np.zeros((224, 224, 3), dtype=np.uint8)  # black BGR frame
        result = clf.predict(frame)

        if result is None:
            print(f"PASS — predict returned None (confidence below threshold)")
        else:
            self.assertIsInstance(result, dict, "result must be a dict")
            self.assertIn("label", result, "result must have 'label' key")
            self.assertIn("confidence", result, "result must have 'confidence' key")
            self.assertIsInstance(result["label"], str, "'label' must be a str")
            self.assertIsInstance(result["confidence"], float, "'confidence' must be a float")
            self.assertGreaterEqual(result["confidence"], 0.0)
            self.assertLessEqual(result["confidence"], 1.0)
            print(f"PASS — predict returned {result}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
