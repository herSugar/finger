# Import required libraries
import cv2  # OpenCV for camera capture and image processing
import mediapipe as mp  # MediaPipe for hand tracking
import pyautogui  # For mouse and scroll control
import math  # For distance calculations

def main():
    """
    Main function that runs the finger tracking application.
    Tracks hand movements to control cursor, click, and scroll.
    """
    
    # Initialize MediaPipe Hand solution
    mp_hands = mp.solutions.hands
    # Configure hand tracking parameters:
    # static_image_mode=False for video input
    # max_num_hands=1 to track only one hand
    # min_detection_confidence=0.7 for reliable detection
    hands = mp_hands.Hands(static_image_mode=False, 
                          max_num_hands=1,
                          min_detection_confidence=0.7)
    
    # Utility for drawing hand landmarks (for visualization)
    mp_drawing = mp.solutions.drawing_utils

    # Get screen dimensions for mapping hand position to cursor
    screen_w, screen_h = pyautogui.size()

    # Initialize webcam capture
    # 0 = default camera, cv2.CAP_DSHOW is for Windows DirectShow
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Click detection variables
    click_threshold = 0.05  # Distance between thumb and index to trigger click
    clicking = False  # Track click state

    # Scroll detection variables
    scroll_threshold = 0.1  # Distance between fingers to trigger scroll
    prev_y = None  # Store previous y position for scroll delta
    scroll_active = False  # Track scroll state

    def calculate_distance(point1, point2):
        """
        Calculate Euclidean distance between two points.
        Args:
            point1: First point (x,y coordinates)
            point2: Second point (x,y coordinates)
        Returns:
            float: Distance between points
        """
        return math.hypot(point1.x - point2.x, point1.y - point2.y)

    def map_to_screen(norm_x, norm_y):
        """
        Convert normalized coordinates (0-1) to screen coordinates.
        Args:
            norm_x: Normalized x coordinate
            norm_y: Normalized y coordinate
        Returns:
            tuple: (x, y) screen coordinates
        """
        return (int(norm_x * screen_w), int(norm_y * screen_h))

    # Main application loop
    while cap.isOpened():
        # Read frame from camera
        success, image = cap.read()
        if not success:
            continue  # Skip failed frames
        
        # Flip image horizontally for mirror effect (more intuitive control)
        image = cv2.flip(image, 1)
        
        # Convert from BGR to RGB color space (MediaPipe requires RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe hand tracking
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:  # If hands are detected
            for hand_landmarks in results.multi_hand_landmarks:
                # Get key landmarks (tips of fingers)
                index_tip = hand_landmarks.landmark[8]  # Index finger tip
                thumb_tip = hand_landmarks.landmark[4]  # Thumb tip
                middle_tip = hand_landmarks.landmark[12]  # Middle finger tip

                # Convert normalized coordinates to screen position
                x, y = map_to_screen(index_tip.x, index_tip.y)

                # 1. CLICK DETECTION (thumb touching index finger)
                thumb_index_dist = calculate_distance(thumb_tip, index_tip)
                
                if thumb_index_dist < click_threshold:
                    if not clicking:  # Only click once until released
                        pyautogui.mouseDown()
                        clicking = True
                        # Visual feedback
                        cv2.putText(image, "CLICKING", (50, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    if clicking:  # Release if previously clicking
                        pyautogui.mouseUp()
                        clicking = False

                # 2. SCROLL DETECTION (index and middle fingers together)
                index_middle_dist = calculate_distance(index_tip, middle_tip)
                
                if index_middle_dist < scroll_threshold:
                    if not scroll_active:  # Initialize scroll mode
                        scroll_active = True
                        prev_y = y  # Store initial position
                        # Visual feedback
                        cv2.putText(image, "SCROLL MODE", (50, 100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    
                    # Calculate scroll amount based on vertical movement
                    # Inverted (prev_y - y) for natural scrolling direction
                    scroll_amount = prev_y - y
                    
                    # Only scroll if movement exceeds threshold (avoid micro-scrolls)
                    if abs(scroll_amount) > 5:
                        pyautogui.scroll(scroll_amount)
                    
                    prev_y = y  # Update previous position
                else:
                    scroll_active = False  # Exit scroll mode
                    # Normal cursor movement when not scrolling
                    pyautogui.moveTo(x, y)

                # Draw hand landmarks (for visualization)
                mp_drawing.draw_landmarks(
                    image, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
                )

        # Display the processed image
        cv2.imshow('Finger Tracking', image)
        
        # Exit on ESC key press
        if cv2.waitKey(5) & 0xFF == 27:
            break

    # Cleanup
    if clicking:  # Ensure mouse button is released
        pyautogui.mouseUp()
    cap.release()  # Release camera
    cv2.destroyAllWindows()  # Close windows

if __name__ == "__main__":
    main()