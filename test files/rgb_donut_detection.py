from pyfirmata2 import Arduino, util        
import time
import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- Arduino setup ----------------
board = Arduino('COM4')
print("starting program")
it = util.Iterator(board)
it.start()

# NOTE: you used D13 in code. If your button is actually on D12, change here.
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
        time.sleep(0.02)  # 20ms

def setuppos():
    move(1, 90, 0.1)
    move(2, 162, 0.1)
    move(3, 37, 0.1)
    move(4, 60, 0.1)

def pickup1():
    move(1, 180, 0.2)
    move(2, 162, 0.2)
    move(3, 37, 0.2)
    move(4, 30, 0.2)

# --------------- Detection utils ----------------
def get_color_label(mean_rgb):
    r, g, b = mean_rgb
    if r > g and r > b:
        return 'r', (0, 0, 255)  # red label, BGR draw color
    elif g > r and g > b:
        return 'g', (0, 255, 0)  # green
    elif b > r and b > g:
        return 'b', (255, 0, 0)  # blue
    else:
        return 'u', (200, 200, 200)  # unknown

# --------------- Live preview + gating ---------------
def live_detect_until_three(conf_thresh=0.6, stable_frames_needed=5):

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    model = YOLO("donutmodel.pt")  # <-- your trained donut model
    print("Live feed running. Press the hardware button to start the sequence.")

    button_pressed = False
    stable_count = 0
    last_button_state = None
    final_colors = []

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Run YOLO on this frame
        results = model(frame)[0]
        boxes = []
        for box in results.boxes:
            conf = float(box.conf)
            if conf < conf_thresh:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            boxes.append({'bbox': (x1, y1, x2, y2), 'label': label, 'conf': conf})

        # Sort detections left-to-right
        boxes.sort(key=lambda b: b['bbox'][0])

        color_results = []
        for b in boxes:
            x1, y1, x2, y2 = b['bbox']
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                color_label, draw_color = 'u', (200, 200, 200)
                mean_rgb = (0, 0, 0)
            else:
                mean_bgr = cv2.mean(crop)[:3]
                mean_rgb = tuple(int(v) for v in mean_bgr[::-1])
                color_label, draw_color = get_color_label(mean_rgb)

            # draw bbox + class(color)
            cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
            txt = f"{b['label']} ({color_label})"
            cv2.putText(frame, txt, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)

            color_results.append({
                'class': b['label'],
                'bbox': b['bbox'],
                'confidence': b['conf'],
                'dominant_rgb': mean_rgb,
                'color_label': color_label
            })

        # HUD
        cv2.putText(frame,
                    f"Detected: {len(color_results)} | Button State: {button_pressed}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if not button_pressed:
            cv2.putText(frame, "Press button to start", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Button State: Waiting for EXACTLY 3 donuts...",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Show the live feed with detections
        cv2.imshow("Donut Detection (YOLO live)", frame)
        cv2.waitKey(1)

        # --- Button handling (active-low per your earlier behavior) ---
        btn = button_pin.value  # None until iterator updates; then True/False
        if btn is not None:
            # Edge detect: go armed on transition to pressed (False)
            if (last_button_state in (True, None)) and (btn is False):
                button_pressed = True
                stable_count = 0  # reset stability counter on arm
                print("Button pressed -> ARMED")
                time.sleep(0.15)  # debounce
            last_button_state = btn

        # --- Armed gating: need EXACTLY 3, stably for N frames ---
        if button_pressed:
            if len(color_results) == 3:
                stable_count += 1
            else:
                stable_count = 0

            if stable_count >= stable_frames_needed:
                # lock in the current colors (left->right)
                final_colors = color_results
                break

        # Keyboard escape (debug)
        if cv2.getWindowProperty("Donut Detection (YOLO live)", cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # small yield; YOLO is the main time sink anyway
        time.sleep(0.001)

    cap.release()
    cv2.destroyAllWindows()
    return final_colors

# ---------------- Main ----------------
def main():
    setuppos()
    results = live_detect_until_three(conf_thresh=0.6, stable_frames_needed=5)

    if len(results) != 3:
        print("Aborted or not exactly 3 detected; exiting safely.")
        board.exit()
        return

    color_seq = ''.join([r['color_label'] for r in results])  # e.g. 'grb'
    print("Color sequence (left to right):", color_seq)

    # Run your servo sequence
    # pickup1()
    board.exit()

if __name__ == "__main__":
    main()
