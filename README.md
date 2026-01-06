PHRIS – Predictive Human Risk Intelligence System

PHRIS is a real-time AI safety system that detects people, tracks them across frames, and computes a risk score when they enter configurable industrial danger zones (machinery, electrical, chemical, etc.).

## Features

- YOLOv8-based **person detection** and **pose estimation**  
- DeepSORT **multi-person tracking** with unique IDs  
- Configurable polygon **danger zones**  
- Multi-factor **risk scoring** (zone, time in zone, speed, posture, proximity, acceleration)  
- On-screen **dashboard overlay** with risk, status, FPS, and alerts  
- **Audio + visual alerts** for critical risk levels  

## Quick Start

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
python main.py
