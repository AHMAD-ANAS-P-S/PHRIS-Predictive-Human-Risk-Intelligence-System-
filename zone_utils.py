# ============================================
# ZONE UTILS - DANGER ZONE MANAGEMENT
# ============================================

import cv2
import numpy as np

# DEFINE YOUR DANGER ZONES
# Each zone is a polygon (list of corner points)
# FORMAT: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

DANGER_ZONES = {
    "HEAVY_MACHINERY": {
        "coords": np.array([[300, 200], [900, 200], [900, 600], [300, 600]]),
        "color": (0, 0, 255),      # Red in BGR format
        "risk": 40,                # Base risk if person enters
        "name": "HEAVY MACHINERY AREA",
        "description": "Industrial machinery - highest danger"
    },

    "ELECTRICAL": {
        "coords": np.array([[500, 150], [700, 150], [700, 400], [500, 400]]),
        "color": (0, 165, 255),    # Orange
        "risk": 35,
        "name": "ELECTRICAL AREA",
        "description": "High voltage equipment"
    },

    "CHEMICAL": {
        "coords": np.array([[100, 100], [250, 100], [250, 300], [100, 300]]),
        "color": (0, 255, 255),    # Yellow
        "risk": 30,
        "name": "CHEMICAL STORAGE",
        "description": "Hazardous materials area"
    }
}

# Safe areas (for reference)
SAFE_ZONES = {
    "OFFICE": {
        "coords": np.array([[1000, 100], [1280, 100], [1280, 300], [1000, 300]]),
        "color": (0, 255, 0),      # Green
        "risk": 0,
        "name": "OFFICE AREA",
        "description": "Safe zone"
    }
}

def draw_danger_zones(frame):
    """
    Draw all danger zones on the frame (visual feedback)

    Args:
        frame: video frame (numpy array)

    Returns:
        frame with zones drawn
    """
    # Draw danger zones (filled + outlined)
    for zone_name, zone_info in DANGER_ZONES.items():
        coords = zone_info["coords"]
        color = zone_info["color"]
        name = zone_info["name"]

        # Create semi-transparent filled polygon
        overlay = frame.copy()
        cv2.fillPoly(overlay, [coords], color)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

        # Draw border
        cv2.polylines(frame, [coords], True, color, 2)

        # Draw label at first corner point
        top_left = tuple(coords[0].astype(int))
        cv2.putText(
            frame,
            name,
            top_left,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return frame

def is_person_in_danger_zone(cx, cy):
    """
    Check if person (at center point cx, cy) is in any danger zone

    Args:
        cx: center x coordinate
        cy: center y coordinate

    Returns:
        (in_danger: bool, zone_name: str, risk_value: int)
    """
    point = (cx, cy)

    for zone_name, zone_info in DANGER_ZONES.items():
        coords = zone_info["coords"]

        # Check if point is inside polygon
        # Returns: >0 if inside, 0 if on edge, <0 if outside
        result = cv2.pointPolygonTest(coords, point, False)

        if result >= 0:  # Person is in this danger zone
            return True, zone_name, zone_info["risk"]

    return False, "SAFE", 0

def get_zone_info(cx, cy):
    """
    Get detailed information about current zone

    Returns: dictionary with zone details
    """
    in_danger, zone_name, risk = is_person_in_danger_zone(cx, cy)

    if in_danger:
        info = DANGER_ZONES[zone_name].copy()
        info["is_safe"] = False
    else:
        info = {"is_safe": True, "name": "SAFE AREA", "risk": 0}

    return info
