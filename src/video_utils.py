import cv2
import numpy as np

def is_ground_view(frame):
    """Checks if the frame is mostly green grass (cricket field)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    ratio = np.sum(mask > 0) / (frame.shape[0] * frame.shape[1])
    return ratio > 0.4  # Keeps frame if >40% is green

def is_crowd_view(frame):
    """Checks for high texture/edges characteristic of an audience crowd shot."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_ratio = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    return edge_ratio > 0.08  # Typically high in crowd shots