from pyfirmata2 import Arduino, util        
import time      
import cv2
import numpy as np
from ultralytics import YOLO                     

board = Arduino('COM3')
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
    move(3, 37, 0.1)
    move(4, 30, 0.1)

setuppos() #calling it as early as possible
    
def pickup1(): # leftmost donut position
    move(2, 123, 1) #arm up
    move(4, 30, 0.2) #open gripper
    movespeed(1, 90, 180, 1.5) #turn to leftmost
    move(3, 37, 0.2)
    movespeed(2, 123, 159, 1) #arm down
    move(4, 80, 1) #close gripper
    movespeed(2, 159, 123, 1) #arm up
    movespeed(1, 180, 90, 1.5) #return to middle position
    
def pickup2(): # topleft donut position
    
    move(2, 123, 1) #arm up
    move(4, 30, 0.2) #open gripper
    movespeed(1, 90, 135, 1.5) #turn to topleft
    move(3, 37, 0.2)
    movespeed(2, 123, 159, 1) #arm down
    move(4, 80, 1) #close gripper
    movespeed(2, 159, 123, 1) #arm up
    movespeed(1, 135, 90, 1.5) #return to middle position
    
def pickup3(): # toprightt donut position
    
    move(2, 123, 1) #arm up
    move(4, 30, 0.2) #open gripper
    movespeed(1, 90, 48, 1.5) #turn to topright
    move(3, 37, 0.2)
    movespeed(2, 123, 159, 1) #arm down
    move(4, 80, 1) #close gripper
    movespeed(2, 159, 123, 1) #arm up
    movespeed(1, 48, 90, 1.5) #return to middle position
    
    
def placedown(): # leftmost donut position

    movespeed(1, 90, 0, 1.5) #turn to rightmost position
    move(3, 33, 0.2)
    movespeed(2, 123, 147, 1) #arm down
    move(4, 30, 1) #open gripper
    movespeed(2, 147, 123, 1) #arm up
    movespeed(1, 0, 90, 1.5) #return to middle position
    movespeed(2, 123, 159, 1) #arm down
    setuppos()
    
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
    movespeed(2, 159, 170, 0.5) #arm down
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

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    model = YOLO("donutmodel.pt")  # <-- your trained donut model
    print("showing the camera feed window now")

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
            cv2.putText(frame, "Button State: Starting program...",
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
        if stable_count >= stable_frames_needed:
            final_colors = color_results
            break

        # small yield; YOLO is the main time sink anyway
        time.sleep(0.001)

    cap.release()
    cv2.destroyAllWindows()
    return final_colors



def main():
    try:
        
        setuppos()
        results = live_detect_until_three(conf_thresh=0.6, stable_frames_needed=5)

        color_seq = ''.join([r['color_label'] for r in results])  # e.g. 'grb'
        print("Color sequence (left to right):", color_seq)
        
        # Priority order
        priority = ['r', 'g', 'b']
        count = 0

        # Loop by priority
        for target_color in priority:
            # Find position of that color in sequence
            pos = color_seq.index(target_color) + 1  # +1 to make it 1-based if needed
            print(f"Running action for {target_color} in position {pos}")

            if pos == 1:
                pickup1()
            elif pos == 2:
                pickup2()
            elif pos == 3:
                pickup3()
            count += 1 # +1 every donut picked up
        
            if count == 3:
               placedownlast()
            else:
                placedown()


        
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user (Ctrl+C). Shutting down...")
    finally:
        board.exit()  # ปิดพอร์ตอย่างปลอดภัยเมื่อจบโปรแกรม
    
if __name__ == "__main__":
    main()
