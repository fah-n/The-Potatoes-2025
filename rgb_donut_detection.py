import cv2
import numpy as np
from ultralytics import YOLO

def detect_and_classify_colors(
        conf_thresh=0.8,
        save_path="output_with_boxes.png"
    ):
    # 1. Load model
    model = YOLO("best.pt")

    # 2. Grab one frame from camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Could not capture image from camera.")

    # 3. Run detection
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

    if not boxes:
        cv2.imwrite(save_path, frame)  # Log blank
        return []

    # 4. Sort boxes left-to-right (by x1)
    boxes.sort(key=lambda b: b['bbox'][0])

    # 5. For each box, get dominant color and classify as red, green, or blue
    color_results = []
    for box in boxes:
        x1, y1, x2, y2 = box['bbox']
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            color_label, dom_rgb = "unknown", (0, 0, 0)
        else:
            # Find mean color (in BGR)
            mean_bgr = cv2.mean(crop)[:3]
            mean_rgb = tuple(int(v) for v in mean_bgr[::-1])  # Convert BGR to RGB
            r, g, b = mean_rgb

            # Simple heuristic for color class
            if r > g and r > b:
                color_label = "red"
                draw_color = (0, 0, 255)
            elif g > r and g > b:
                color_label = "green"
                draw_color = (0, 255, 0)
            elif b > r and b > g:
                color_label = "blue"
                draw_color = (255, 0, 0)
            else:
                color_label = "unknown"
                draw_color = (200, 200, 200)
            dom_rgb = mean_rgb

            # Draw box with color
            cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
            text = f"{box['label']} ({color_label})"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)

        color_results.append({
            'class': box['label'],
            'bbox': box['bbox'],
            'confidence': box['conf'],
            'dominant_rgb': dom_rgb,
            'color_label': color_label
        })

    # 6. Save annotated output image
    cv2.imwrite(save_path, frame)

    return color_results

# Usage example:
if __name__ == "__main__":
    results = detect_and_classify_colors()
    print("Detection results (left to right):")
    for r in results:
        print(r)
