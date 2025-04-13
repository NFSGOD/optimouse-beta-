import cv2
import mediapipe as mp
import time
import pyautogui
pyautogui.alert("Starting Vidcp standby for live view and press Q to exit")
# Initialize Mediapipe and video capture
vid = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpdraw = mp.solutions.drawing_utils

# Initialize time variables for FPS calculation
ptime = 0
frame_count = 0
process_interval = 2  # Process every 2nd frame

# Get screen dimensions for cursor mapping
screen_width, screen_height = pyautogui.size()

# Smoothing variables
prev_x, prev_y = 0, 0
smoothing_factor = 0.7  # Adjust smoothing (0 = no smoothing, 1 = slow but smooth)

# Initialize results to avoid NameError
results = None

while True:
    success, img = vid.read()
    if not success:
        break

    # Flip the image for natural mirroring and get dimensions
    img = cv2.flip(img, 1)
    h, w, c = img.shape

    # Downscale the image for faster processing
    small_img = cv2.resize(img, (w // 2, h // 2))

    # Process every nth frame for efficiency
    frame_count += 1
    if frame_count % process_interval == 0:
        small_imgRGB = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
        results = hands.process(small_imgRGB)

    # Process results if landmarks are detected
    if results and results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # Map normalized coordinates to original frame size
                cx, cy = int(lm.x * w), int(lm.y * h)

                # Map id == 17 (pinky knuckle) to cursor movement
                if id == 8:
                    # Highlight the landmark with a circle
                    cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

                    # Map hand position to screen dimensions
                    target_x = int(lm.x * screen_width)
                    target_y = int(lm.y * screen_height)

                    # Smooth cursor movement
                    cursor_x = int(prev_x * smoothing_factor + target_x * (1 - smoothing_factor))
                    cursor_y = int(prev_y * smoothing_factor + target_y * (1 - smoothing_factor))

                    # Move the cursor
                    pyautogui.moveTo(cursor_x, cursor_y)

                    # Update previous position
                    prev_x, prev_y = cursor_x, cursor_y

            # Draw hand landmarks on the frame
            mpdraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Calculate and display FPS
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # Display the video feed
    cv2.imshow("Efficient Hand Tracking with Cursor Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy windows
vid.release()
cv2.destroyAllWindows()
