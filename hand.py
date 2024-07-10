import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands and Drawing Utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Initialize a blank canvas
canvas = None

# Variables to store the previous finger position
prev_x, prev_y = None, None

def calculate_brush_size(z):
    # Map the z value to a brush size. Adjust the range as needed.
    # z values are negative, and more negative means further away
    min_z, max_z = -0.5, -0.1  # These values may need adjustment based on your setup
    min_brush, max_brush = 5, 50  # Minimum and maximum brush sizes
    if z < min_z:
        z = min_z
    if z > max_z:
        z = max_z
    brush_size = int(min_brush + (max_z - z) / (max_z - min_z) * (max_brush - min_brush))
    return brush_size

with mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hands
        results = hands.process(frame_rgb)

        # Initialize the canvas if it is not already initialized
        if canvas is None:
            canvas = np.zeros_like(frame)

        # Draw hand annotations on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the coordinates of the index finger tip and thumb tip
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                h, w, _ = frame.shape
                index_cx, index_cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                thumb_cx, thumb_cy = int(thumb_tip.x * w), int(thumb_tip.y * h)
                
                # Calculate brush size based on Z-coordinate (distance)
                index_z = index_finger_tip.z
                brush_size = calculate_brush_size(index_z)

                # Draw with index finger
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (index_cx, index_cy), (0, 255, 0), brush_size)
                
                # Erase with thumb
                erase_size = calculate_brush_size(thumb_tip.z)
                cv2.circle(canvas, (thumb_cx, thumb_cy), erase_size, (0, 0, 0), -1)

                # Update the previous position
                prev_x, prev_y = index_cx, index_cy

        else:
            # Reset previous position if no hand is detected
            prev_x, prev_y = None, None

        # Combine the canvas and the frame
        combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

        # Display the result
        cv2.imshow('Finger Painting', combined)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
