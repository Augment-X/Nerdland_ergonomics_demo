import cv2
import mediapipe as mp
import numpy as np
import os
import time
import screeninfo
from datetime import datetime
import csv

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================================
# MODEL
# ==========================================================

MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task")

# zoom settings
zoom_factor = 1.0

# if we make a recording
RECORDING = False

latest_result = None

def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=result_callback
)

# running_mode = vision.RunningMode.VIDEO (no livestream)
landmarker = vision.PoseLandmarker.create_from_options(options)

# ==========================================================
# CAMERA SETTINGS
# ==========================================================

def choose_camera():
    print("Detecting available cameras...")
    available_cameras = []

    # Test camera indices 0–5
    for i in range(6):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            print(f"✅ Camera {i} is available")
            cap.release()
        else:
            print(f"⚠️ Camera {i} not available")

    if not available_cameras:
        raise RuntimeError("❌ No cameras detected on this system.")

    # Let user choose
    while True:
        cam = input(f"\nChoose a camera index from {available_cameras}: ")
        if cam.isdigit() and int(cam) in available_cameras:
            return int(cam)
        print("Invalid choice. Try again.")

def digital_zoom(frame, zoom_factor):
 
    h, w = frame.shape[:2]
    new_w = int(w / zoom_factor)
    new_h = int(h / zoom_factor)

    x1 = (w - new_w) // 2
    y1 = (h - new_h) // 2

    cropped = frame[y1:y1+new_h, x1:x1+new_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

# ==========================================================
# Visualisation
# ==========================================================

camera_index = choose_camera()
print(f"\n🎥 Using camera {camera_index}\n")

# Open camera
cap = cv2.VideoCapture(camera_index)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Set camera resolution to FULL SCREEN dimensions
screen = screeninfo.get_monitors()[0]
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen.height)

# Open fullscreen display window
cv2.namedWindow("Real-Time REBA Assessment", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Real-Time REBA Assessment", screen.width, screen.height)
cv2.setWindowProperty("Real-Time REBA Assessment", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("Press ESC to quit")

# Small updating graph in the lower left corner
score_history = []  # list of (timestamp, score)
WINDOW_SECONDS = 10
GRAPH_W, GRAPH_H = int(screen.width/3), int(screen.height/4.5)

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def pt(landmarks, i, w, h):
    return np.array([landmarks[i].x*w, landmarks[i].y*h])

def small_graph_color(score):
    """Return BGR (blue-green-red) color for the mini graph."""
    if score == 1: # Negligible
        return (0, 255, 0)       # green
    elif 2 <= score <= 3: # Low
        return (0, 255, 255)     # yellow
    elif 4 <= score <= 7: # Medium
        return (0, 165, 255)     # orange
    elif 8 <= score <= 10: # High
        return (0, 0, 255)           
    else: # Very high
        return (6,6,149)       # dark red

# ==========================================================
# REBA SCORING
# ==========================================================

def score_trunk(a): return 1 if a<5 else 2 if a<20 else 3 if a<60 else 4
def score_neck(a): return 1 if a<10 else 2 if a<20 else 3
def score_legs(a): return 1 if a>160 else 2
def score_upper_arm(a): return 1 if a<20 else 2 if a<45 else 3 if a<90 else 4
def score_lower_arm(a): return 1 if 60<=a<=100 else 2
def score_wrist(a): return 1 if a<15 else 2
def score_force(a): return 1 if a>5 else 0 # TO DO!

TABLE_A = [
    [[1,2,3],[2,3,4],[3,4,5],[4,5,6]],
    [[2,3,4],[3,4,5],[4,5,6],[5,6,7]]
]

TABLE_B = [
    [[1,2],[2,3],[3,4],[4,5]],
    [[2,3],[3,4],[4,5],[5,6]]
]

TABLE_C = [
[1,2,3,4,5,6,7],
[2,3,4,5,6,7,8],
[3,4,5,6,7,8,9],
[4,5,6,7,8,9,10],
[5,6,7,8,9,10,11],
[6,7,8,9,10,11,12],
[7,8,9,10,11,12,12]
]

def compute_reba(trunk, neck, legs, upper, lower, wrist):
    a = TABLE_A[legs-1][trunk-1][neck-1]
    b = TABLE_B[wrist-1][upper-1][lower-1]
    return TABLE_C[a-1][b-1]

def reba_risk(score):
    if score==1: return "Verwaarloosbaar" #"Negligible"
    elif score<=3: return "Laag" #"Low"
    elif score<=7: return "Medium" #"Medium"
    elif score<=10: return "Hoog" #"High"
    else: return "Heel hoog" #"Very High"

def risk_color(score):
    # Clamp
    score = max(1, min(score, 12))
    alpha = (score - 1) / 11.0  # 0 → 1

    if alpha < 0.5:
        # Green → Yellow
        t = alpha / 0.5
        red = int(255 * t)
        green = 255
    else:
        # Yellow → Red
        t = (alpha - 0.5) / 0.5
        red = 255
        green = int(255 * (1 - t))

    return (0, green, red)  # BGR

# ==========================================================
# MAIN LOOP
# ==========================================================

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = digital_zoom(frame, zoom_factor)  # adjust with "+" and "-" button

    frame = cv2.resize(frame, (screen.width, screen.height), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    timestamp = int(time.time()*1000)
    landmarker.detect_async(mp_image, timestamp)

    reba_score = 1  # default

    if latest_result and latest_result.pose_landmarks:
        landmarks = latest_result.pose_landmarks[0]
        h, w, _ = frame.shape

        r_shoulder = pt(landmarks,12,w,h)
        r_elbow = pt(landmarks,14,w,h)
        r_wrist = pt(landmarks,16,w,h)
        r_hip = pt(landmarks,24,w,h)
        r_knee = pt(landmarks,26,w,h)
        r_ankle = pt(landmarks,28,w,h)
        l_shoulder = pt(landmarks,11,w,h)
        l_hip = pt(landmarks,23,w,h)

        mid_shoulder = (l_shoulder + r_shoulder)/2
        mid_hip = (l_hip + r_hip)/2

        trunk_angle = calculate_angle(mid_hip+[0,-100], mid_hip, mid_shoulder)
        neck_angle = calculate_angle(mid_shoulder+[0,-100], mid_shoulder, pt(landmarks,0,w,h))
        knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
        upper_arm = calculate_angle(r_elbow, r_shoulder, r_hip)
        lower_arm = calculate_angle(r_shoulder, r_elbow, r_wrist)
        wrist = 0

        trunk_s = score_trunk(trunk_angle)
        neck_s = score_neck(neck_angle)
        leg_s = score_legs(knee_angle)
        upper_s = score_upper_arm(upper_arm)
        lower_s = score_lower_arm(lower_arm)
        wrist_s = score_wrist(wrist)

        reba_score = compute_reba(trunk_s, neck_s, leg_s, upper_s, lower_s, wrist_s)

        # ======================================================
        # DETERMINE MOST STRESSED JOINT
        # ======================================================

        joint_scores = {
            "trunk": (trunk_s, mid_hip),
            "leg": (leg_s, r_knee),
            "upper_arm": (upper_s, r_shoulder),
            "lower_arm": (lower_s, r_elbow),
            "wrist": (wrist_s, r_wrist)
        }

        # Find highest subscore
        max_joint = max(joint_scores, key=lambda k: joint_scores[k][0])

        # Only highlight if significant
        if joint_scores[max_joint][0] >= 2:
            _, max_joint_pos = joint_scores[max_joint]

        color = risk_color(reba_score)

        # ======================================================
        # DRAW SKELETON (HIGH VISIBILITY)
        # ======================================================

        connections = [
            (11,13),(13,15),(12,14),(14,16),
            (11,12),(11,23),(12,24),(23,24),
            (23,25),(25,27),(24,26),(26,28)
        ]

        for s,e in connections:
            x1,y1 = int(landmarks[s].x*w), int(landmarks[s].y*h)
            x2,y2 = int(landmarks[e].x*w), int(landmarks[e].y*h)

            # Draw thick black outline first (contrast)
            cv2.line(frame, (x1,y1), (x2,y2), (255,255,255), 8)

            # Draw colored skeleton on top
            #cv2.line(frame, (x1,y1), (x2,y2), color, 4)

        # ======================================================
        # DRAW JOINTS (HIGH VISIBILITY)
        # ======================================================

        for i in [11,12,13,14,15,16,23,24,25,26,27,28]:
            x = int(landmarks[i].x*w)
            y = int(landmarks[i].y*h)

            # White joint
            cv2.circle(frame, (x,y), 8, (255,255,255), -1)

            # OR Colored joint
            # cv2.circle(frame, (x,y), 5, color, -1)
        if joint_scores[max_joint][0] >= 2:
            x, y = int(max_joint_pos[0]), int(max_joint_pos[1])
            # coloured circle around heaviest loaded joint
            cv2.circle(frame, (x,y), 14, color, 3)  

    # =====================================================
    # GRAPH WITH LIVE UPDATES
    # =====================================================

    # --- Update History ---
    now = time.time()
    score_history.append((now, reba_score))

    # Keep only last 10 seconds
    score_history = [(t, s) for (t, s) in score_history if now - t <= WINDOW_SECONDS]

    # --- Create REBA History Graph ---
    graph = np.zeros((GRAPH_H, GRAPH_W, 3), dtype=np.uint8)

    if len(score_history) > 1:
        # Extract times & scores
        times = np.array([t for (t, s) in score_history])
        scores = np.array([s for (t, s) in score_history])

        # Time axis normalized 0 → 1
        t_norm = (times - times.min()) / (times.max() - times.min() + 1e-6)

        # Score axis normalized 0 → 1
        s_norm = scores / 12.0

        xs = (t_norm * (GRAPH_W - 1)).astype(int)
        ys = (GRAPH_H - 1 - s_norm * (GRAPH_H - 1)).astype(int)

        # --- Draw color-coded REBA score ---
        for i in range(1, len(xs)):
            seg_color = small_graph_color(scores[i])
            cv2.line(graph, (xs[i-1], ys[i-1]), (xs[i], ys[i]), seg_color, 2)

    # Draw border
    cv2.rectangle(graph, (0,0), (GRAPH_W-1, GRAPH_H-1), (255,255,255), 1)

    # time labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (255, 255, 255)

    left_label = f"nu-{WINDOW_SECONDS}s"
    (left_w, left_h), _ = cv2.getTextSize(left_label, font, font_scale, thickness)
    cv2.putText(graph, left_label,  (5, GRAPH_H - 5), font, font_scale, color, thickness)
    
    right_label = "nu"
    (right_w, right_h), _ = cv2.getTextSize(right_label, font, font_scale, thickness)
    cv2.putText(graph, right_label,  (GRAPH_W - right_w - 5, GRAPH_H - 5), font, font_scale, color, thickness)
    
    # Display Score
    risk_text = reba_risk(reba_score)
    
    # ======================================================
    # TRANSPARENT RISK OVERLAY
    # ======================================================

    """overlay = frame.copy()
    color = risk_color(reba_score)
    overlay[:] = color

    alpha_overlay = 0.5
    frame = cv2.addWeighted(overlay, alpha_overlay, frame, 1-alpha_overlay, 0)"""

    cv2.putText(graph,f"REBA-score: {reba_score}",
                (15,40),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),3)
    cv2.putText(graph,f"Risico Level: {risk_text}",(15, 80),cv2.FONT_HERSHEY_SIMPLEX,1.2,small_graph_color(reba_score),3)

    
    # --- Pop-up screen, with graph at top-left ---
    y_offset = 40   # adjust as needed
    x_offset = frame.shape[1] - GRAPH_W - 40

    frame[y_offset:y_offset+GRAPH_H,
        x_offset:x_offset+GRAPH_W] = graph
    
    if RECORDING:
        (tw, th), _ = cv2.getTextSize("Recording", cv2.FONT_HERSHEY_SIMPLEX, 2.0, 1)
        x = frame.shape[1] - tw - 20
        y = th + 20
        # Red dot
        #if time.time()
        cv2.circle(frame, (x - 20, y - 20), 12, (0, 0, 255), -1)
        # Text
        cv2.putText(frame, "Recording", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3)

        # write video and csv
        writer.write(frame)
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        csv_writer.writerow([timestamp, reba_score])

    cv2.imshow("Real-Time berekening van REBA-Score", frame)


    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # escape
        break

    
    # Make a recording of your session
    if key == ord('r') and not RECORDING:
        RECORDING = True
        print("🔴 Recording started")

        # define files
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"reba_sessions/{session_id}"
        os.makedirs(out_dir, exist_ok=True)
        video_path = os.path.join(out_dir, "recording.mp4")
        csv_path = os.path.join(out_dir, "reba_scores.csv")
        
        # write video in case of recording
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None:
            fps = 30  # fallback

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (frame.shape[1], frame.shape[0]))

        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file, delimiter=';')
        csv_writer.writerow(["timestamp", "reba_score"])

    if key == ord('s') and RECORDING:
        RECORDING = False
        print("⏹ Recording stopped")
        writer.release()
        csv_file.close()
    
    # zoom in and out
    if key == ord('+'):
        zoom_factor = min(3.0, zoom_factor + 0.1)  # maximum zoom is a factor 3
    if key == ord('-'):
        zoom_factor = max(1.0, zoom_factor - 0.1)  # You can van not go beyond the original zoom


cap.release()
cv2.destroyAllWindows()
    
