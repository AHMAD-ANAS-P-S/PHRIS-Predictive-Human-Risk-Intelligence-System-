# ============================================
# PHRIS - MAIN PROGRAM (COMPLETE VERSION)
# Proactive Human Risk Intelligence System
# ============================================

import cv2
import time
import winsound
import numpy as np
from ultralytics import YOLO

# Import our custom modules
from tracker_utils import track_people, get_person_center, calculate_speed
from zone_utils import draw_danger_zones, is_person_in_danger_zone, get_zone_info
from risk_engine import calculate_risk, cleanup_old_people
from pose_utils import detect_pose, get_pose_risk

print("=" * 70)
print("🚀 PHRIS - Proactive Human Risk Intelligence System")
print("=" * 70)
print("\n📦 Loading AI models (first time takes ~30 seconds)...")
print("   ├─ Loading YOLO person detector...")

# Load YOLO models
person_detector = YOLO("yolov8n.pt")  # Person detection
print("   ├─ YOLO person detection... ✅")
print("   ├─ Loading pose estimation model...")
pose_detector = YOLO("yolov8n-pose.pt")  # Pose estimation
print("   ├─ Pose estimation... ✅")

print("\n📷 Opening camera...")
cap = cv2.VideoCapture(0)

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("❌ ERROR: Could not open camera!")
    print("   Try: cv2.VideoCapture(0, cv2.CAP_DSHOW)  [Windows]")
    exit()

print("   ├─ Camera opened... ✅")
print("\n🎬 Starting video stream... Press 'q' to quit")
print("=" * 70)

# ===== STATISTICS =====
frame_count = 0
total_alerts = 0
start_time = time.time()
last_alert_time = 0

# ===== MAIN LOOP =====
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Cannot read from camera")
        break
    
    frame_count += 1
    current_time = time.time()
    
    # === STEP 1: DETECT PEOPLE ===
    person_results = person_detector(frame, conf=0.5)
    detections = []
    
    for r in person_results:
        for box in r.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            
            if cls == 0 and conf > 0.5:  # Class 0 = person
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                detections.append([x1, y1, x2, y2, conf])
    
    # === STEP 2: TRACK PEOPLE (UNIQUE IDS) ===
    tracks = track_people(detections, frame)
    
    # === STEP 3: DRAW DANGER ZONES ===
    draw_danger_zones(frame)
    
    # === STEP 4: POSE DETECTION ===
    poses = detect_pose(frame)
    pose_risks = get_pose_risk(poses)
    
    # === STEP 5: ANALYZE EACH PERSON ===
    critical_people = []
    
    for track_idx, (track_id, x1, y1, x2, y2) in enumerate(tracks):
        # Get person center
        cx, cy = get_person_center(x1, y1, x2, y2)
        
        # Check if in danger zone
        in_danger, zone_name, zone_risk = is_person_in_danger_zone(cx, cy)
        
        # Calculate speed
        speed = calculate_speed(track_id, cx, cy, current_time)
        
        # Get posture information
        posture = "STANDING"
        posture_risk = 0
        if track_idx in pose_risks:
            posture = pose_risks[track_idx]["posture"]
            posture_risk = pose_risks[track_idx]["risk"]
        
        # === STEP 6: CALCULATE RISK ===
        risk_info = calculate_risk(
            person_id=track_id,
            cx=cx, cy=cy,
            in_danger_zone=in_danger,
            zone_name=zone_name,
            zone_risk=zone_risk,
            speed=speed,
            posture=posture,
            posture_risk=posture_risk
        )
        
        risk_score = risk_info["score"]
        factors = risk_info["factors"]
        trend = risk_info["trend"]
        status = risk_info["status"]
        color = risk_info["color"]
        
        # === STEP 7: DRAW PERSON BOX ===
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Main info line
        info_text = f"ID:{track_id} RISK:{risk_score}/100 [{status}]"
        cv2.putText(frame, info_text,
                   (x1, y1 - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Trend line
        trend_text = f"Trend: {trend}"
        cv2.putText(frame, trend_text,
                   (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Zone info
        zone_text = f"Zone: {zone_name}"
        cv2.putText(frame, zone_text,
                   (x1, y2 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Speed info
        speed_text = f"Speed: {speed:.1f} px/s"
        cv2.putText(frame, speed_text,
                   (x1, y2 + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Posture info
        posture_text = f"Posture: {posture}"
        cv2.putText(frame, posture_text,
                   (x1, y2 + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Risk factors breakdown
        y_offset = y2 + 80
        for factor_name, factor_value in factors.items():
            factor_text = f"├─ {factor_name}: +{factor_value}"
            cv2.putText(frame, factor_text,
                       (x1, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y_offset += 18
        
        # === STEP 8: CRITICAL ALERT ===
        if risk_score > 70:
            critical_people.append((track_id, risk_score))
    
    # === HANDLE CRITICAL ALERTS ===
    if critical_people:
        if current_time - last_alert_time > 1:  # Alert every 1 second max
            total_alerts += 1
            last_alert_time = current_time
            
            # Sound alert
            try:
                winsound.Beep(1000, 200)
            except:
                pass
        
        # Red overlay alert
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (800, 120), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        cv2.putText(frame, "🚨 CRITICAL RISK ALERT! 🚨",
                   (30, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        alert_text = f"IMMEDIATE ACTION NEEDED FOR: "
        for person_id, score in critical_people:
            alert_text += f"Person {person_id} (Risk:{score}) "
        
        cv2.putText(frame, alert_text,
                   (30, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # === DRAW DASHBOARD ===
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 360), (400, 710), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # Dashboard title
    cv2.putText(frame, "📊 PHRIS DASHBOARD",
               (20, 390),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Statistics
    elapsed_time = current_time - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    
    cv2.putText(frame, f"Frames: {frame_count}",
               (20, 420),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.putText(frame, f"FPS: {fps:.1f}",
               (20, 450),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.putText(frame, f"People: {len(tracks)}",
               (20, 480),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.putText(frame, f"Alerts: {total_alerts}",
               (20, 510),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    runtime = int(elapsed_time)
    cv2.putText(frame, f"Runtime: {runtime}s",
               (20, 540),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Legend
    cv2.rectangle(frame, (10, 580), (80, 610), (0, 255, 0), 2)
    cv2.putText(frame, "SAFE", (15, 605), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.rectangle(frame, (90, 580), (180, 610), (0, 255, 255), 2)
    cv2.putText(frame, "WARNING", (100, 605), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    cv2.rectangle(frame, (190, 580), (270, 610), (0, 0, 255), 2)
    cv2.putText(frame, "CRITICAL", (200, 605), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # === DISPLAY FRAME ===
    cv2.imshow("PHRIS - Proactive Human Risk Intelligence System", frame)
    
    # === QUIT ON 'Q' ===
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n" + "=" * 70)
        print("🛑 Shutting down PHRIS...")
        print(f"   ├─ Total frames processed: {frame_count}")
        print(f"   ├─ Total alerts triggered: {total_alerts}")
        print(f"   ├─ Runtime: {int(elapsed_time)} seconds")
        print(f"   ├─ Average FPS: {fps:.1f}")
        print("=" * 70)
        break
    
    # Cleanup old person data
    if frame_count % 30 == 0:  # Every 30 frames
        cleanup_old_people()

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("✅ PHRIS closed successfully")
