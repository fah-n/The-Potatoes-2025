from pyfirmata2 import Arduino, util        
import time      
import cv2
import numpy as np
from ultralytics import YOLO      
import math               

board = Arduino('COM4')
print("im setting up bro chill")

it = util.Iterator(board)
it.start()
button_pin = board.get_pin('d:13:i')
button_pin.enable_reporting()

servos = {
    1: board.get_pin('d:2:s'),
    2: board.get_pin('d:3:s'),
    3: board.get_pin('d:4:s'),
    4: board.get_pin('d:5:s'),
}

def move(servonum, angle, duration):
    servo = servos[servonum]
    start = time.time()
    while (time.time() - start) < duration:
        servo.write(angle)
        time.sleep(0.02)  # 20 ms pulse interval
        
def movespeed(servonum, current_angle, target_angle, duration):

    servo = servos[servonum]
    steps = abs(target_angle - current_angle)
    min_step_time = 0.01  # 10 ms

    if steps == 0:
        servo.write(target_angle)
        time.sleep(duration)
        return target_angle

    step = 1 if target_angle > current_angle else -1
    step_time = max(duration / steps, min_step_time)

    moved = False
    for angle in range(current_angle, target_angle + step, step):
        servo.write(angle)
        time.sleep(step_time)
        moved = True

    if not moved or steps < 2:
        servo.write(target_angle)
        time.sleep(duration)

    return target_angle

def setuppos(): # setup at a safe known angle --> normally once initialized it moves to 0
    move(1, 90, 0.1)
    move(2, 159, 0.1)
    move(3, 22, 0.1)
    move(4, 0, 0.1)

setuppos() #calling it as early as possible
    
def pickup(from_angle, to_angle, expected_color=None):
    # 1) pre-position roughly to the donut
    movespeed(1, int(from_angle), int(to_angle), 2)

    # 2) fine align with ArUco + YOLO (this moves servo 1 itself)
    aligned_angle, seen_color = turn_until_align(
        start_angle_deg=int(to_angle),
        last_color_hint=expected_color  # or None
    )
    current = int(aligned_angle)

    # 3) do the pickup sequence (use `current` as the true pan position)
    move(2, 123, 0.1)   # arm up
    move(4, 60, 0.1)    # open gripper
    move(3, 26, 0.1)

    movespeed(2, 123, 159, 0.2)   # arm down
    movespeed(3, 26, 45, 1.0)     # push outwards
    movespeed(4, 60, 130, 0.6)    # close gripper
    movespeed(2, 159, 123, 0.2)   # arm up

    # 4) return pan to tower (use the aligned angle as the "from")
    movespeed(1, current, 0, 1.0)

    move(3, 33, 0.1)
    movespeed(2, 123, 140, 0.6)   # arm down
    move(4, 0, 0.1)               # open gripper
    movespeed(2, 140, 123, 0.3)   # arm up


def placedownlast():
    movespeed(1, 90, 0, 1.5) #turn to rightmost position
    move(3, 33, 0.2)
    movespeed(2, 123, 147, 1) #arm down
    move(4, 30, 1) #open gripper
    movespeed(2, 147, 159, 0.2) #arm down
    move(4, 80, 0.5) #close gripper
    movespeed(1, 0, 95, 0.5) #return to middle position
    move(4, 30, 0.5) #open gripper
    movespeed(3, 33, 75, 0.5) #angle outwards
    movespeed(2, 159, 177, 0.5) #arm down
    movespeed(3, 75, 95, 0.5) #angle outwards

    setuppos()

    
def wait_for_button():
    print("Waiting for button on D12 to be pressed (pyfirmata2, .value)...")
    while True:
        # .value gets updated by iterator (None = no data yet)
        if button_pin.value is False:
            print("Button pressed!")
            break
        time.sleep(0.02)  # small sleep to avoid busy loop

# ---------- Minimal color classifier ----------
def get_color_label(mean_rgb,
                    s_floor=35,    # lower S floor so dim colors aren't thrown out
                    v_floor=20,    # lower V floor so dark purple isn't 'u'
                    dark_v=90,     # what counts as "dark" for rescue
                    red_ratio=0.7, # require R >= 70% of B to call purple in blue-ish hues
                    g_limit=0.8):  # require G <= 80% of max(R,B) to keep green out
    """
    mean_rgb: (R,G,B)
    Returns: 'r','g','b','y','p','u'
    Rule added: if hue is blue-ish AND it's dark AND R is present -> treat as purple.
    """
    r, g, b = mean_rgb
    # OpenCV uses BGR
    bgr = np.uint8([[[b, g, r]]])
    h, s, v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0]

    # be forgiving with dim colors
    if s < s_floor or v < v_floor:
        return 'u'

    # --- Purple rescue in blue-ish hues ---
    # If hue is in the blue neighborhood and it's dark, and red is present (not negligible),
    # prefer 'p' over 'b' unless green is too strong.
    if 90 <= h <= 135:  # blue-ish (covers cyan→blue→blue/purple edge)
        if (v <= dark_v) and (r >= red_ratio * b) and (g <= g_limit * max(r, b)):
            return 'p'

    # Standard bands
    if (0 <= h <= 10) or (160 <= h <= 179):
        return 'r'
    if 20 <= h <= 35:
        return 'y'
    if 36 <= h <= 85:
        return 'g'

    # Cyan bridge: prefer green unless blue clearly dominates
    if 86 <= h <= 100:
        return 'b' if b > 1.1 * g else 'g'

    if 90 <= h <= 130:
        return 'b'
    if 131 <= h <= 159:
        return 'p'

    return 'u'

# ---------- Intrinsics from HFOV ----------
def intrinsics_from_hfov(width, height, hfov_deg):
    hfov_rad = math.radians(hfov_deg)
    fx = width / (2.0 * math.tan(hfov_rad/2.0))
    fy = fx
    cx, cy = width/2.0, height/2.0
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], np.float32), np.zeros((5,), np.float32)

# ---------- ArUco yaw (draws marker + axes with cv2.drawFrameAxes) ----------
def aruco_yaw_deg(frame, K, dist, aruco_dict, marker_length_m, marker_id, yaw_offset_deg):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ad = cv2.aruco.getPredefinedDictionary(aruco_dict)
    det = cv2.aruco.ArucoDetector(ad, cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None

    # pick specific id or first
    pick = 0
    if marker_id is not None:
        pick = None
        for i,_id in enumerate(ids.flatten()):
            if int(_id) == int(marker_id):
                pick = i; break
        if pick is None:
            return None

    # draw marker border + id
    cv2.aruco.drawDetectedMarkers(frame, [corners[pick]], np.array([ids.flatten()[pick]]))

    # pose (needs contrib)
    if not hasattr(cv2.aruco, "estimatePoseSingleMarkers"):
        return None

    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers([corners[pick]], marker_length_m, K, dist)
    if rvecs is None:
        return None

    rvec = rvecs[0,0]
    tvec = tvecs[0,0]

    # draw XYZ axes
    try:
        cv2.drawFrameAxes(frame, K, dist, rvec, tvec, marker_length_m * 0.5)
    except Exception:
        pass

    R,_ = cv2.Rodrigues(rvec)
    yaw = math.degrees(math.atan2(R[1,0], R[0,0])) + yaw_offset_deg
    return yaw

# ---------- YOLO donut: returns bearing, color, list of detections, target idx ----------
def donut_bearing_and_color(frame, model, hfov_deg, conf_thresh):
    H,W = frame.shape[:2]
    cx_frame = W//2
    pred = model(frame)[0]

    detections = []  # list of dicts: {'bbox':(x1,y1,x2,y2), 'lab': 'r/g/b/y/p/u', 'cx': int}
    for b in pred.boxes:
        if float(b.conf) < conf_thresh: 
            continue
        x1,y1,x2,y2 = map(int, b.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        if crop.size:
            mean_bgr = cv2.mean(crop)[:3]
            mean_rgb = tuple(int(v) for v in mean_bgr[::-1])
            lab = get_color_label(mean_rgb)
        else:
            lab = 'u'
        detections.append({'bbox':(x1,y1,x2,y2), 'lab': lab, 'cx': (x1+x2)//2})

    if not detections:
        return None, None, detections, None

    # pick rightmost one (your preference)
    target_idx = max(range(len(detections)), key=lambda i: detections[i]['cx'])
    deg_per_px = hfov_deg / float(W)
    bearing_deg = (detections[target_idx]['cx'] - cx_frame) * deg_per_px
    return bearing_deg, detections[target_idx]['lab'], detections, target_idx

# ---------- One-shot scan: return angles/colors for exactly 4 donuts ----------
def normalize_angle_deg(a):
    # Normalize to (-180, 180]
    while a <= -180.0: a += 360.0
    while a >   180.0: a -= 360.0
    return a

def scan_four_donuts_after_button(
    model_path='yellowdonutmodel.pt',
    cam_index=0,
    hfov_deg=108.0,
    conf_thresh=0.6,
    frame_size=(640,360),
    pick='rightmost'   # 'rightmost' or 'left_to_right'
):
    """
    Live preview + detection overlay. Waits for the physical button press (active-low).
    On press, captures the current frame and returns up to 4 donuts with:
      [{'abs_yaw_deg': float|None, 'bearing_deg': float, 'color': 'r/g/b/y/p/u', 'bbox': (x1,y1,x2,y2)}]
    Right-most first by default. If ArUco not visible, abs_yaw_deg is None but bearing_deg is still valid.
    """

    # --- Open camera and model ---
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  frame_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])
    model = YOLO(model_path)

    print("Live preview up. Press the button to capture this frame.")

    last_button_state = None
    captured_frame = None

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        H, W = frame.shape[:2]
        cx_frame = W // 2

        # YOLO detections for overlay only
        pred = model(frame)[0]
        boxes = getattr(pred, "boxes", None)
        detections = []
        if boxes is not None:
            for b in boxes:
                try:
                    conf = float(b.conf[0] if hasattr(b.conf, "__len__") else b.conf)
                except Exception:
                    conf = 0.0
                if conf < conf_thresh:
                    continue
                try:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                except Exception:
                    x1, y1, x2, y2 = map(int, b.xyxy)
                crop = frame[y1:y2, x1:x2]
                lab = 'u'
                if crop.size:
                    mean_bgr = cv2.mean(crop)[:3]
                    mean_rgb = (int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0]))
                    lab = get_color_label(mean_rgb)
                detections.append({'bbox': (x1,y1,x2,y2), 'lab': lab, 'cx': (x1+x2)//2})

        # Draw overlay (boxes + labels)
        for d in detections:
            x1,y1,x2,y2 = d['bbox']
            col = {'r':(0,0,255), 'g':(0,255,0), 'b':(255,0,0), 'y':(0,255,255), 'p':(255,0,255)}.get(d['lab'], (200,200,200))
            cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
            cv2.putText(frame, d['lab'], (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

        # HUD + center line
        cv2.line(frame, (cx_frame, 0), (cx_frame, H-1), (255,255,255), 1)
        cv2.putText(frame, f"Detected: {len(detections)}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.putText(frame, "Press button to CAPTURE this frame",
                    (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        cv2.imshow("Scan 4 donuts (wait for button)", frame)
        cv2.waitKey(1)

        # Button handling (active-low, edge-trigger)
        btn = button_pin.value  # None until iterator updates; then True/False
        if btn is not None:
            if (last_button_state in (True, None)) and (btn is False):
                # pressed
                captured_frame = frame.copy()
                time.sleep(0.15)  # debounce
                break
            last_button_state = btn

        time.sleep(0.001)

    # Process the captured frame once and return up to 4 donuts with angles/colors
    out = []
    if captured_frame is not None:
        H, W = captured_frame.shape[:2]
        K, dist = intrinsics_from_hfov(W, H, hfov_deg)

        # get ArUco yaw from the SAME captured frame
        yaw = aruco_yaw_deg(captured_frame, K, dist, cv2.aruco.DICT_4X4_50, 0.03, None, 0.0)

        # Run YOLO on captured frame (fresh)
        pred = model(captured_frame)[0]
        boxes = getattr(pred, "boxes", None)
        detections = []
        if boxes is not None:
            for b in boxes:
                try:
                    conf = float(b.conf[0] if hasattr(b.conf, "__len__") else b.conf)
                except Exception:
                    conf = 0.0
                if conf < conf_thresh:
                    continue
                try:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                except Exception:
                    x1, y1, x2, y2 = map(int, b.xyxy)
                crop = captured_frame[y1:y2, x1:x2]
                lab = 'u'
                if crop.size:
                    mean_bgr = cv2.mean(crop)[:3]
                    mean_rgb = (int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0]))
                    lab = get_color_label(mean_rgb)
                detections.append({'bbox': (x1,y1,x2,y2), 'lab': lab, 'cx': (x1+x2)//2})

        # sort and keep 4
        if pick == 'rightmost':
            detections.sort(key=lambda d: d['cx'], reverse=True)
        else:
            detections.sort(key=lambda d: d['cx'])
        detections = detections[:4]

        # compute bearings & abs_yaw
        cx_frame = W // 2
        deg_per_px = hfov_deg / float(W)

        def norm_deg(a):
            while a <= -180.0: a += 360.0
            while a >   180.0: a -= 360.0
            return a

        for d in detections:
            bearing_deg = (d['cx'] - cx_frame) * deg_per_px
            abs_yaw = None if yaw is None else norm_deg(yaw + bearing_deg)
            out.append({
                'abs_yaw_deg': abs_yaw,
                'bearing_deg': bearing_deg,
                'color': d['lab'],
                'bbox': d['bbox']
            })

    cap.release()
    cv2.destroyAllWindows()
    return out

def yaw_to_servo_angle(yaw_deg):
    """
    Convert yaw (-180..180) to servo angle (0..180)
    where:
        -90 yaw = servo 0  (right)
         0  yaw = servo 90 (center)
        +90 yaw = servo 180 (left)
    Values outside ±90° are clamped.
    """
    # Clamp yaw to avoid going beyond servo range
    yaw_deg = max(-90, min(90, yaw_deg))
    return int(round(90 - yaw_deg))

# ---------- helpers ----------
def clamp_deg(a, lo=0, hi=180):
    return int(max(lo, min(hi, round(a))))

# OPTIONAL: simple yaw->servo mapper you can tune (handles distortion with scale+offset)
# Example defaults assume center≈90 and ±90 yaw spans ~0..180. Tweak k and c to fit your camera.
def yaw_to_servo(yaw_deg, k=1.0, c=90.0):
    # servo = c - k * yaw ; yaw +90 -> left(>90), yaw -90 -> right(<90)
    return clamp_deg(c - k * float(yaw_deg))

# ---------- DROP-IN: refined aligner ----------
def turn_until_align(
    model_path='donutmodel.pt',
    cam_index=0,
    marker_length_m=0.03,
    aruco_dict=cv2.aruco.DICT_4X4_50,
    marker_id=None,
    hfov_deg=108.0,
    yaw_offset_deg=0.0,
    start_angle_deg=90.0,  # pass your rough guess here
    kp=0.9,
    min_step=0.7,
    max_step=3.0,
    tol=1.5,
    conf_thresh=0.6,
    frame_size=(640,360),
    max_runtime=25.0,
    last_color_hint='u'
):
    # use servo 1 as the pan
    pan = servos[1]

    def write_servo(a):
        pan.write(clamp_deg(a))

    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  frame_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])

    ok, frame = cap.read()
    if not ok:
        cap.release(); cv2.destroyAllWindows()
        raise RuntimeError("No camera feed")

    H, W = frame.shape[:2]
    K, dist = intrinsics_from_hfov(W, H, hfov_deg)
    model = YOLO(model_path)

    current = float(start_angle_deg)
    write_servo(current)
    t0 = time.time()

    # exponential moving average to reduce jitter
    sm_bearing, sm_yaw = None, None
    alpha = 0.3  # 0..1 (lower = smoother)
    last_seen_color = last_color_hint if last_color_hint else 'u'

    while time.time() - t0 < max_runtime:
        ok, frame = cap.read()
        if not ok:
            continue

        # 1) ArUco yaw (absolute)
        yaw = aruco_yaw_deg(frame, K, dist, aruco_dict, marker_length_m, marker_id, yaw_offset_deg)

        # 2) Donut bearing (relative to image center) + color
        donut_bearing, color, dets, target_idx = donut_bearing_and_color(frame, model, hfov_deg, conf_thresh)

        # remember most recent non-'u' color for this target
        if color is not None and color != 'u' and target_idx is not None:
            last_seen_color = color

        # smooth signals
        if donut_bearing is not None:
            sm_bearing = donut_bearing if sm_bearing is None else (1 - alpha) * sm_bearing + alpha * donut_bearing
        if yaw is not None:
            sm_yaw = yaw if sm_yaw is None else (1 - alpha) * sm_yaw + alpha * yaw

        if sm_yaw is None or sm_bearing is None:
            # show and continue; you can add a small sweep here if desired
            cv2.imshow("Align", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        # control error: flip sign here if it moves the wrong way
        error = sm_yaw - sm_bearing
        while error > 180: error -= 360
        while error < -180: error += 360

        # done?
        if abs(error) <= tol:
            final_color = color if (color is not None and color != 'u') else last_seen_color
            cap.release(); cv2.destroyAllWindows()
            return clamp_deg(current), final_color

        # bounded step
        step = kp * error
        step = max(min_step, min(max_step, abs(step))) * (1 if error > 0 else -1)
        current = clamp_deg(current + step)
        write_servo(current)

        # HUD (optional)
        cv2.putText(frame, f"yaw:{sm_yaw:.1f}  donut:{sm_bearing:.1f}  err:{error:.1f}  servo:{current}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.imshow("Align", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        time.sleep(0.03)

    cap.release(); cv2.destroyAllWindows()
    return clamp_deg(current), last_seen_color

def main():
    try:
        setuppos()
        last_angle = 90

        results = scan_four_donuts_after_button(pick='rightmost')

        # keep only r/g/b and items that have an angle
        priority = {'r': 0, 'g': 1, 'b': 2}
        filtered = [d for d in results if d['color'] in priority and d['abs_yaw_deg'] is not None]

        # sort by color (r→g→b), THEN by angle (ascending). 
        # If you want biggest angle first, add reverse=True or negate the angle in the key.
        sorted_results = sorted(filtered, key=lambda d: (priority[d['color']], d['abs_yaw_deg']))


        for d in sorted_results:
            if d['abs_yaw_deg'] is None:
                continue
            new_angle = yaw_to_servo_angle(d['abs_yaw_deg'])
            color = d['color']
            print(f"Picking {color}: yaw={d['abs_yaw_deg']} → servo={new_angle}")
            pickup(int(last_angle), int(new_angle))
            last_angle = 0  # reset to center after pickup
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user (Ctrl+C). Shutting down...")
    finally:
        board.exit()  # ปิดพอร์ตอย่างปลอดภัยเมื่อจบโปรแกรม
    
if __name__ == "__main__":
    main()
