import cv2
import numpy as np

def get_dominant_color_name(rgb):
    r, g, b = rgb
    if r > g and r > b:
        return "Red"
    elif g > r and g > b:
        return "Green"
    elif b > r and b > g:
        return "Blue"
    else:
        return "Unclear"

# Start video capture
cap = cv2.VideoCapture(0)

# Size of square (you can change this)
square_size = 50

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Coordinates of center square
    cx, cy = w // 2, h // 2
    top_left = (cx - square_size // 2, cy - square_size // 2)
    bottom_right = (cx + square_size // 2, cy + square_size // 2)

    # Draw square
    cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), 2)

    # Get color from the square area
    square_region = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    avg_color_bgr = np.mean(square_region, axis=(0, 1))
    avg_color_rgb = avg_color_bgr[::-1]  # Convert BGR to RGB

    # Identify dominant color
    dominant_color = get_dominant_color_name(avg_color_rgb)

    # Display detected color
    cv2.putText(frame, f"Dominant Color: {dominant_color}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the result
    cv2.imshow('Color Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
