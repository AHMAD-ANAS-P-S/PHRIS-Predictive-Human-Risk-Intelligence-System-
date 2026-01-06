# ============================================
# POSE UTILS - DETECT BODY POSITION
# ============================================

from ultralytics import YOLO
import numpy as np

# Load pose model (detects 17 body joints)
pose_model = YOLO("yolov8n-pose.pt")

def detect_pose(frame):
    """
    Detect body pose (skeleton with joints)
    
    Returns: poses array with 17 keypoints per person
    """
    results = pose_model(frame, conf=0.5)
    
    if len(results) == 0:
        return []
    
    poses = []
    for result in results:
        if result.keypoints is None:
            continue
        
        keypoints = result.keypoints.xy  # Get keypoints
        poses.append(keypoints)
    
    return poses


def analyze_posture(keypoints):
    """
    Analyze what person is doing based on body position
    
    Keypoints:
    0-4: Head and neck area
    5-10: Arms
    11-16: Legs and torso
    
    Returns: (posture_type, risk_score)
    """
    
    if keypoints is None or len(keypoints) < 17:
        return "UNKNOWN", 0
    
    # Extract key joints (index numbers from YOLO pose)
    # 0=nose, 5=left_shoulder, 6=right_shoulder
    # 11=left_hip, 12=right_hip, 15=left_ankle, 16=right_ankle
    
    try:
        # Shoulder positions
        left_shoulder = keypoints   # [x, y]
        right_shoulder = keypoints
        
        # Hip positions  
        left_hip = keypoints
        right_hip = keypoints
        
        # Ankle positions
        left_ankle = keypoints
        right_ankle = keypoints
        
        # If keypoints not detected (confidence = 0)
        if (left_shoulder == 0 or left_hip == 0):
            return "UNKNOWN", 0
        
        # Calculate body angles
        shoulder_to_hip_dist = abs(left_shoulder - left_hip)
        hip_to_ankle_dist = abs(left_hip - left_ankle)
        
        # If person bending (shoulder-to-hip distance small)
        if shoulder_to_hip_dist < 40:
            return "BENDING", 25  # Risky near machinery
        
        # If person lying down (hip-to-ankle distance small)
        elif hip_to_ankle_dist < 30:
            return "LYING", 50  # Very risky
        
        # If person kneeling
        elif shoulder_to_hip_dist > 100 and hip_to_ankle_dist < 40:
            return "KNEELING", 15  # Somewhat risky
        
        # Normal standing
        else:
            return "STANDING", 0
    
    except:
        return "UNKNOWN", 0


def get_pose_risk(poses):
    """
    Get risk scores for all detected poses
    
    Returns: dictionary of {person_index: risk_score}
    """
    pose_risks = {}
    
    for idx, keypoints in enumerate(poses):
        posture, risk = analyze_posture(keypoints)
        pose_risks[idx] = {"posture": posture, "risk": risk}
    
    return pose_risks
