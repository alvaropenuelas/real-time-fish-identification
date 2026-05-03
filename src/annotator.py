import cv2
import numpy as np


class Annotator:
    _FONT = cv2.FONT_HERSHEY_SIMPLEX
    _SCALE = 0.6
    _THICKNESS = 1
    _BOX_COLOR = (0, 200, 0)    # green
    _TEXT_FG = (255, 255, 255)  # white
    _TEXT_BG = (0, 0, 0)        # black pill

    def draw(
        self,
        frame: np.ndarray,
        label: str,
        confidence: float,
        bbox: tuple | None = None,
        alt_predictions: list | None = None,
    ) -> np.ndarray:
        out = frame.copy()
        text = f"{label} {confidence:.0%}"
        (tw, th), baseline = cv2.getTextSize(text, self._FONT, self._SCALE, self._THICKNESS)
        pad = 4

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), self._BOX_COLOR, 2)
            tx, ty = x1, y1 - pad
        else:
            tx, ty = pad, th + pad

        # Black pill behind text
        cv2.rectangle(
            out,
            (tx, ty - th - baseline - pad),
            (tx + tw + pad * 2, ty + pad),
            self._TEXT_BG,
            -1,
        )
        cv2.putText(
            out, text,
            (tx + pad, ty),
            self._FONT, self._SCALE, self._TEXT_FG,
            self._THICKNESS, cv2.LINE_AA,
        )

        if alt_predictions:
            alt_scale = 0.4
            alt_color = (180, 180, 180)
            ay = ty + th + pad + 2
            for alt_label, alt_conf in alt_predictions:
                alt_text = f"{alt_label} {alt_conf:.0%}"
                cv2.putText(
                    out, alt_text,
                    (tx + pad, ay),
                    self._FONT, alt_scale, alt_color,
                    self._THICKNESS, cv2.LINE_AA,
                )
                (_, alt_th), _ = cv2.getTextSize(alt_text, self._FONT, alt_scale, self._THICKNESS)
                ay += alt_th + 2

        return out
