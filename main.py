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
    move(1, 0, 0.1)
    move(2, 157, 0.1)
    move(3, 23, 0.1)
    move(4, 0, 0.1)

setuppos() #calling it as early as possible
    
def pickup(current_angle): # leftmost donut position
    
    move(2, 123, 0.1) #arm up
    move(4, 60, 0.1) #open gripper
    move(3, 26, 0.1)
    movespeed(2, 123, 159, 0.2) #arm down
    movespeed(3, 26, 45, 1) #push outwards
    movespeed(4, 60, 130, 0.6) #close gripper
    movespeed(2, 159, 123, 0.2) #arm up
    movespeed(1, current_angle, 0, 1) #return to tower
    move(3, 33, 0.1)
    movespeed(2, 123, 140, 0.6) #arm down
    move(4, 0, 0.1) #open gripper
    movespeed(2, 140, 123, 0.3) #arm up
    move(1, 40, 0.4) #turn a bit left to avoid tower
    move(3, 23, 0.2)
    move(2, 157, 0.3) #arm down
    setuppos()
    
def trash(current_angle): # leftmost donut position
    
    move(2, 123, 0.1) #arm up
    move(4, 60, 0.1) #open gripper
    move(3, 26, 0.1)
    movespeed(2, 123, 159, 0.2) #arm down
    movespeed(3, 26, 45, 1) #push outwards
    movespeed(4, 60, 130, 0.6) #close gripper
    movespeed(2, 157, 123, 0.2) #arm up
    movespeed(1, current_angle, 180, 2) #return to tower
    move(3, 33, 0.1)
    movespeed(2, 123, 157, 0.6) #arm down
    move(4, 0, 0.1) #open gripper
    movespeed(2, 157, 123, 0.3) #arm up
    movespeed(1, 180, current_angle, 2) #return to tower
    move(3, 23, 0.2)
    move(2, 157, 0.3) #arm down
    setuppos()

    
def placedownfinal(): # leftmost donut position
    move(4, 30, 0.1) #open gripper
    movespeed(2, 123, 159, 0.2) #arm down
    movespeed(3, 37, 75, 0.2) #angle outwards
    movespeed(2, 159, 170, 0.2) #arm down
    movespeed(3, 75, 95, 0.2) #angle outwards
    movespeed(2, 170, 159, 0.2) #arm down
    movespeed(3, 95, 75, 0.2) #angle outwards
    # setuppos()



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


    target_idx = max(range(len(detections)), key=lambda i: detections[i]['cx']) #pick rightmost one
    deg_per_px = hfov_deg / float(W)
    bearing_deg = (detections[target_idx]['cx'] - cx_frame) * deg_per_px
    return bearing_deg, detections[target_idx]['lab'], detections, target_idx

# ---------- Main align function ----------
def turn_until_align(
    model_path='donutmodel.pt',
    cam_index=0,
    marker_length_m=0.03,
    aruco_dict=cv2.aruco.DICT_4X4_50,
    marker_id=None,
    hfov_deg=108.0,
    yaw_offset_deg=0.0,
    start_angle_deg=12.0,
    kp=0.9,
    min_step=1.0,
    max_step=4.0,
    tol=1.5, #tolerance for obj placement
    conf_thresh=0.6,
    frame_size=(640,360),
    max_runtime=25.0,
    last_color_hint='u'   # NEW: seed the memory with a previous/expected color
):

    servo = servos[1]

    def write_servo(a):
        servo.write(float(max(0, min(180, a))))

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

    # NEW: memory of last seen (non-'u') color for the CURRENT target
    last_seen_color = last_color_hint if last_color_hint else 'u'

    while time.time()-t0 < max_runtime:
        ok, frame = cap.read()
        if not ok:
            continue

        # ArUco yaw (draws marker + axes)
        yaw = aruco_yaw_deg(frame, K, dist, aruco_dict, marker_length_m, marker_id, yaw_offset_deg)

        # Donut bearing + color + all detections
        donut_bearing, color, dets, target_idx = donut_bearing_and_color(frame, model, hfov_deg, conf_thresh)

        # --- draw donuts ---
        for i, d in enumerate(dets):
            x1,y1,x2,y2 = d['bbox']
            lab = d['lab']
            col = {'r':(0,0,255), 'g':(0,255,0), 'b':(255,0,0), 'y':(0,255,255), 'p':(255,0,255)}.get(lab,(200,200,200))
            thickness = 3 if i == target_idx else 1
            cv2.rectangle(frame, (x1,y1), (x2,y2), col, thickness)
            cv2.putText(frame, lab, (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

        # If we see a target color this frame and it's not 'u', refresh memory
        if color is not None and color != 'u' and target_idx is not None:
            last_seen_color = color

        # debug text (also show last memory)
        txt_yaw  = "None" if yaw is None else f"{yaw:.1f}"
        txt_bear = "None" if donut_bearing is None else f"{donut_bearing:.1f}"
        cv2.putText(frame, f"yaw:{txt_yaw}  donut:{txt_bear}  last:{last_seen_color}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        if yaw is None or donut_bearing is None:
            cv2.imshow("Align", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        # Control: align marker yaw to donut bearing
        error = yaw - donut_bearing
        while error > 180: error -= 360
        while error < -180: error += 360

        if abs(error) <= tol:
            # Use current color if available, else fall back to last_seen_color
            final_color = color if (color is not None and color != 'u') else last_seen_color
            cv2.putText(frame, "Aligned!", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.imshow("Align", frame)
            cv2.waitKey(10)
            cap.release()
            cv2.destroyAllWindows()
            return current, final_color

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
    # If we timed out, still return the best known color we saw
    return current, last_seen_color



def main():
    try:
        
        setuppos()
        angle, color = turn_until_align()
        currentangle = int(angle)
        print(f"Stopped at {currentangle}, color={color}")
        if color == "p" or color == "y": #if color is purple or yellow
            trash(currentangle)
        else:
            pickup(currentangle)
        
       
       
       
    except KeyboardInterrupt:
        print("\nProgram interrupted by user (Ctrl+C). Shutting down...")
    finally:
        board.exit()  # ปิดพอร์ตอย่างปลอดภัยเมื่อจบโปรแกรม
    
if __name__ == "__main__":
    main()
