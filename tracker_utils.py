# ============================================
# TRACKER UTILS - TRACK UNIQUE PEOPLE
# ============================================

from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

# Create global tracker
tracker = DeepSort(
    max_age=30,  # Remember person for 30 frames if missing
    n_init=3     # Need 3 confirmed detections to add person
)

# Store position history for speed calculation
position_history = {}

def track_people(detections, frame):
    """
    Track each person with UNIQUE ID
    
    Args:
        detections: list of [x1, y1, x2, y2, confidence]
        frame: video frame
    
    Returns:
        list of (track_id, x1, y1, x2, y2)
    """
    
    if not detections:
        return []
    
    # Format detections for Deep SORT
    # Need: [left, top, width, height] format
    formatted = []
    for d in detections:
        x1, y1, x2, y2, conf = d
        w = x2 - x1
        h = y2 - y1
        bbox = [x1, y1, w, h]
        formatted.append((bbox, conf, 'person'))
    
    # Update tracker with detections
    tracks = tracker.update_tracks(formatted, frame=frame)
    
    results = []
    for t in tracks:
        if not t.is_confirmed():  # Ignore unconfirmed tracks
            continue
        
        track_id = t.track_id  # UNIQUE ID for this person
        x1, y1, x2, y2 = map(int, t.to_ltrb())
        results.append((track_id, x1, y1, x2, y2))
    
    return results


def get_person_center(x1, y1, x2, y2):
    """Get center point of bounding box"""
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return cx, cy


def calculate_speed(person_id, cx, cy, current_time):
    """
    Calculate how fast person is moving
    
    Args:
        person_id: unique person ID
        cx, cy: current center position
        current_time: current time
    
    Returns:
        speed in pixels/second
    """
    
    global position_history
    
    if person_id not in position_history:
        position_history[person_id] = {
            'positions': [],
            'times': [],
            'speed': 0
        }
    
    hist = position_history[person_id]
    
    # Add current position
    hist['positions'].append((cx, cy))
    hist['times'].append(current_time)
    
    # Keep only last 10 positions
    if len(hist['positions']) > 10:
        hist['positions'].pop(0)
        hist['times'].pop(0)
    
    speed = 0
    
    # Calculate speed from last 2 positions
    if len(hist['positions']) >= 2:
        prev_x, prev_y = hist['positions'][-2]
        curr_x, curr_y = hist['positions'][-1]
        
        # Distance in pixels
        distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
        
        # Time in seconds
        time_diff = hist['times'][-1] - hist['times'][-2]
        
        if time_diff > 0:
            speed = distance / time_diff
    
    hist['speed'] = speed
    return speed


def cleanup_old_tracks():
    """Remove old position history to save memory"""
    global position_history
    
    # This prevents memory leak from building up
    if len(position_history) > 100:
        # Keep only most recent 50 people
        recent_ids = list(position_history.keys())[-50:]
        position_history = {pid: position_history[pid] for pid in recent_ids}
