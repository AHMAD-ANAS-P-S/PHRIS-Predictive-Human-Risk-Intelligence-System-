# ============================================
# RISK ENGINE - CALCULATE DANGER LEVEL (FIXED)
# ============================================

from collections import deque
import time

# Store person profiles (track their history)
person_profiles = {}

class PersonProfile:
    """Store information about a tracked person"""
    def __init__(self, person_id):
        self.person_id = person_id
        self.times = deque(maxlen=100)  # Last 100 timestamps
        self.positions = deque(maxlen=100)  # Last 100 positions
        self.zones = deque(maxlen=100)  # Last 100 zones
        self.speeds = deque(maxlen=100)  # Last 100 speeds
        self.risk_scores = deque(maxlen=100)  # Last 100 risk scores
        self.first_danger_time = None  # When person entered danger
        self.last_update = time.time()

def get_profile(person_id):
    """Get or create person profile"""
    if person_id not in person_profiles:
        person_profiles[person_id] = PersonProfile(person_id)
    return person_profiles[person_id]

def calculate_risk(person_id, cx, cy, in_danger_zone, zone_name, zone_risk, 
                   speed, posture, posture_risk):
    """
    Calculate overall risk score for a person (0-100)
    
    Args:
        person_id: unique person identifier
        cx, cy: center position
        in_danger_zone: boolean, is person in danger zone?
        zone_name: name of zone
        zone_risk: base risk from zone (0-50)
        speed: speed in pixels/second
        posture: body posture (STANDING, BENDING, LYING, KNEELING)
        posture_risk: risk from posture (0-50)
    
    Returns:
        risk_info: dict with score, factors, trend, status, color
    """
    
    profile = get_profile(person_id)
    profile.last_update = time.time()
    
    # === FACTOR 1: ZONE RISK (0-40) ===
    factor_zone = 0
    if in_danger_zone:
        factor_zone = zone_risk  # 30-40 points
    
    # === FACTOR 2: TIME IN ZONE (0-20) ===
    factor_time = 0
    if in_danger_zone:
        if profile.first_danger_time is None:
            profile.first_danger_time = time.time()
        
        time_in_zone = time.time() - profile.first_danger_time
        
        if time_in_zone > 5:      # More than 5 seconds
            factor_time = 20
        elif time_in_zone > 3:    # More than 3 seconds
            factor_time = 15
        elif time_in_zone > 1:    # More than 1 second
            factor_time = 10
    else:
        # Reset timer when leaving zone
        profile.first_danger_time = None
        factor_time = 0
    
    # === FACTOR 3: SPEED (0-20) ===
    factor_speed = 0
    if speed > 200:              # Very fast (pixels/sec)
        factor_speed = 20
    elif speed > 100:            # Fast
        factor_speed = 15
    elif speed > 50:             # Moderate
        factor_speed = 10
    elif speed > 20:             # Slow but moving
        factor_speed = 5
    
    # === FACTOR 4: POSTURE RISK (0-20) ===
    factor_posture = min(20, posture_risk)
    
    # === FACTOR 5: PROXIMITY RISK (0-10) ===
    # (Can add proximity to machinery detection here)
    factor_proximity = 0
    if in_danger_zone and cx < 100:  # Very close to left edge
        factor_proximity = 10
    elif in_danger_zone and cx > 1180:  # Very close to right edge
        factor_proximity = 10
    
    # === FACTOR 6: ACCELERATION (0-10) ===
    factor_acceleration = 0
    if len(profile.speeds) >= 2:
        prev_speed = list(profile.speeds)[-2]
        curr_speed = speed
        acceleration = curr_speed - prev_speed
        
        if acceleration > 50:    # Sudden speed increase
            factor_acceleration = 10
        elif acceleration > 25:
            factor_acceleration = 5
    
    profile.speeds.append(speed)
    
    # === CALCULATE TOTAL RISK ===
    total_risk = (factor_zone + factor_time + factor_speed + 
                  factor_posture + factor_proximity + factor_acceleration)
    
    # Cap at 100
    total_risk = min(100, total_risk)
    
    # Store in history
    profile.risk_scores.append(total_risk)
    profile.zones.append(zone_name)
    profile.times.append(time.time())
    profile.positions.append((cx, cy))
    
    # === DETERMINE TREND ===
    trend = "STABLE"
    if len(profile.risk_scores) >= 3:
        recent_scores = list(profile.risk_scores)[-3:]
        if recent_scores[-1] > recent_scores[0] + 10:
            trend = "↑ INCREASING"
        elif recent_scores[-1] < recent_scores[0] - 10:
            trend = "↓ DECREASING"
        else:
            trend = "→ STABLE"
    
    # === DETERMINE STATUS ===
    if total_risk < 30:
        status = "SAFE"
        color = (0, 255, 0)  # Green
    elif total_risk < 60:
        status = "WARNING"
        color = (0, 255, 255)  # Yellow
    else:
        status = "CRITICAL"
        color = (0, 0, 255)  # Red
    
    # === BUILD FACTORS DICTIONARY ===
    factors = {}
    if factor_zone > 0:
        factors["Zone"] = factor_zone
    if factor_time > 0:
        factors["Time"] = factor_time
    if factor_speed > 0:
        factors["Speed"] = factor_speed
    if factor_posture > 0:
        factors["Posture"] = factor_posture
    if factor_proximity > 0:
        factors["Proximity"] = factor_proximity
    if factor_acceleration > 0:
        factors["Accel"] = factor_acceleration
    
    return {
        "score": int(total_risk),
        "factors": factors,
        "trend": trend,
        "status": status,
        "color": color,
        "zone": zone_name
    }

def cleanup_old_people(max_age=60):
    """
    Remove people who haven't been seen for max_age seconds
    Prevents memory leak from tracking ghost people
    """
    global person_profiles
    current_time = time.time()
    
    to_delete = []
    for person_id, profile in person_profiles.items():
        if current_time - profile.last_update > max_age:
            to_delete.append(person_id)
    
    for person_id in to_delete:
        del person_profiles[person_id]
    
    return len(to_delete)