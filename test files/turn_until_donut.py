# pip install pyfirmata2 ultralytics opencv-contrib-python numpy opencv-python

import cv2, math, time
import numpy as np
from ultralytics import YOLO
from pyfirmata2 import Arduino, util

# ---------- Minimal color classifier ----------
def get_color_label(mean_rgb):
    defs = [
        ('r', (0,0,255),   [(0,10),(160,179)]),
        ('y', (0,255,255), [(20,35)]),
        ('g', (0,255,0),   [(36,85)]),
        ('b', (255,0,0),   [(90,130)]),
        ('p', (255,0,255), [(131,159)]),
    ]
    r,g,b = mean_rgb
    bgr = np.uint8([[[b,g,r]]])
    h,s,v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0,0]
    if s < 50 or v < 50: return 'u'
    for lab,_,ranges in defs:
        for lo,hi in ranges:
            if lo <= h <= hi: return lab
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

    # pick a specific id, else first
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

    # pose
    if not hasattr(cv2.aruco, "estimatePoseSingleMarkers"):
        # Contrib not available -> can't get yaw
        return None

    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers([corners[pick]], marker_length_m, K, dist)
    if rvecs is None:
        return None

    rvec = rvecs[0,0]
    tvec = tvecs[0,0]

    # draw XYZ axes with cv2.drawFrameAxes (in core cv2)
    try:
        cv2.drawFrameAxes(frame, K, dist, rvec, tvec, marker_length_m * 0.5)
    except Exception:
        pass  # if unavailable in your build, just skip drawing

    R,_ = cv2.Rodrigues(rvec)
    yaw = math.degrees(math.atan2(R[1,0], R[0,0])) + yaw_offset_deg
    return yaw

# ---------- YOLO donut ----------
def donut_bearing_and_color(frame, model, hfov_deg, conf_thresh):
    H,W = frame.shape[:2]
    cx_frame = W//2
    pred = model(frame)[0]
    best = None
    for b in pred.boxes:
        if float(b.conf) < conf_thresh: continue
        x1,y1,x2,y2 = map(int, b.xyxy[0])
        cx = (x1+x2)//2
        if best is None or abs(cx-cx_frame) < best['err']:
            crop = frame[y1:y2, x1:x2]
            lab = 'u'
            if crop.size:
                mean_bgr = cv2.mean(crop)[:3]
                mean_rgb = tuple(int(v) for v in mean_bgr[::-1])
                lab = get_color_label(mean_rgb)
            best = {'cx': cx, 'err': abs(cx-cx_frame), 'color': lab}
    if best is None: return None, None
    deg_per_px = hfov_deg / float(W)
    bearing_deg = (best['cx'] - cx_frame) * deg_per_px
    return bearing_deg, best['color']

# ---------- Main align function ----------
def turn_until_align(board,
    servo_pin='d:2:s',
    model_path='donutmodel.pt',
    cam_index=0,
    marker_length_m=0.03,
    aruco_dict=cv2.aruco.DICT_4X4_50,
    marker_id=None,
    hfov_deg=108.0,
    yaw_offset_deg=0.0,
    start_angle_deg=3.0,
    kp=0.9,
    min_step=1.0,
    max_step=4.0,
    tol=1.5,
    conf_thresh=0.6,
    frame_size=(640,360),
    max_runtime=25.0):

    servo = board.get_pin(servo_pin)
    def write_servo(a):
        servo.write(float(max(0,min(180,a))))

    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])

    ret, frame = cap.read()
    if not ret: raise RuntimeError("No camera feed")
    H,W = frame.shape[:2]
    K, dist = intrinsics_from_hfov(W,H,hfov_deg)
    model = YOLO(model_path)

    current = start_angle_deg
    write_servo(current)
    t0 = time.time()
    final_color = 'u'

    while time.time()-t0 < max_runtime:
        ok, frame = cap.read()
        if not ok: continue

        # ArUco yaw (draws marker + axes)
        yaw = aruco_yaw_deg(frame, K, dist, aruco_dict, marker_length_m, marker_id, yaw_offset_deg)

        # Donut bearing + color
        donut_bearing, color = donut_bearing_and_color(frame, model, hfov_deg, conf_thresh)

        cv2.putText(frame, f"yaw:{None if yaw is None else round(yaw,1)}  donut:{None if donut_bearing is None else round(donut_bearing,1)}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        if yaw is None or donut_bearing is None:
            # show frame even if one is missing (so you can debug visually)
            cv2.imshow("Align", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        # Control: align marker yaw to donut bearing
        error = yaw - donut_bearing
        while error > 180: error -= 360
        while error < -180: error += 360


        if abs(error) <= tol:
            final_color = color
            cv2.putText(frame, "Aligned!", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.imshow("Align", frame)
            cv2.waitKey(10)
            break

        step = kp * error
        step = max(min_step, min(max_step, abs(step))) * (1 if error>0 else -1)
        current = max(0,min(180,current+step))
        write_servo(current)

        cv2.putText(frame, f"servo:{current:.1f}  step:{step:.1f}",
                    (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.imshow("Align", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        time.sleep(0.03)

    cap.release()
    cv2.destroyAllWindows()
    return current, final_color

# ---------- Run ----------
if __name__ == "__main__":
    board = Arduino('COM4')
    it = util.Iterator(board); it.start()
    try:
        angle, color = turn_until_align(
            board,
            yaw_offset_deg=0.0,   # your setting
            start_angle_deg=20.0   # your setting
        )
        print(f"Stopped at {angle:.1f}°, color={color}")
    finally:
        board.exit()
