import cv2
import numpy as np
import dlib
from time import sleep
import pyautogui

# Setup
cursor_speed = 20  # Cursor speed
loop_time_count = 0  # Count the actual time of the loops

# File destination
direct = 'D:\\Downloads\\Ton Minh\\'
datfile = direct + 'shape_predictor_68_face_landmarks.dat'

# Initialize the variable to read images from the camera
cap = cv2.VideoCapture(0)

# Initialize functions to detect and predict
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(datfile)

# Function to calculate the Euclidean distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Function to draw facial landmarks on the frame
def draw_facial_landmarks(landmarks, frame):
    for i in range(68):
        coordinates = (landmarks.part(i).x, landmarks.part(i).y)
        cv2.circle(frame, coordinates, 0, (255, 0, 0), 5)

# Function to extract the coordinate values of a landmark point
def get_landmark_point(landmarks, index):
    return (landmarks.part(index).x, landmarks.part(index).y)

# Function to handle eye aspect ratio and blinking
def handle_eye_aspect_ratio(landmarks, frame):
    left_eye_width = calculate_distance(get_landmark_point(landmarks, 38), get_landmark_point(landmarks, 40))
    right_eye_width = calculate_distance(get_landmark_point(landmarks, 43), get_landmark_point(landmarks, 47))
    base_distance = calculate_distance(get_landmark_point(landmarks, 28), get_landmark_point(landmarks, 29))
    
    # Detect if eyes are closed
    if left_eye_width < (base_distance * 7.5 / 18) or right_eye_width < (base_distance * 7.5 / 18):
        cv2.putText(frame, 'Eyes closed, click!', (230, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0))
        pyautogui.click()  # Mouse click
    else:
        cv2.putText(frame, 'Eyes opened', (230, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))

# Function to handle mouth movement
def handle_mouth_movement(landmarks, frame, current_speed):
    mouth_opening = calculate_distance(get_landmark_point(landmarks, 62), get_landmark_point(landmarks, 66))
    cv2.putText(frame, 'M: ' + str(round(mouth_opening, 2)), (10, 130), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0))
    if mouth_opening > 8:
        cv2.putText(frame, 'Talking', (255, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
        # Throttle the cursor speed changes and add a small delay to prevent rapid changes
        sleep(0.2)
        new_speed = min(80, current_speed + 20)
    else:
        new_speed = 20

    return new_speed

# Function to detect face orientation and move the cursor accordingly
def handle_face_orientation(landmarks, frame, current_speed):
    ratio_right = calculate_distance(get_landmark_point(landmarks, 2), get_landmark_point(landmarks, 30)) / \
                  calculate_distance(get_landmark_point(landmarks, 30), get_landmark_point(landmarks, 13))
    ratio_left = calculate_distance(get_landmark_point(landmarks, 13), get_landmark_point(landmarks, 30)) / \
                 calculate_distance(get_landmark_point(landmarks, 30), get_landmark_point(landmarks, 2))
    
    if ratio_right >= 1.5:
        cv2.putText(frame, 'Turn Right', (355, 110), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
        pyautogui.move(current_speed, 0, duration=pyautogui.MINIMUM_DURATION)
    elif ratio_left >= 1.5:
        cv2.putText(frame, 'Turn Left', (155, 110), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
        pyautogui.move(-current_speed, 0, duration=pyautogui.MINIMUM_DURATION)
    else:
        ratio_up = calculate_distance(get_landmark_point(landmarks, 8), get_landmark_point(landmarks, 33)) / \
                   calculate_distance(get_landmark_point(landmarks, 33), get_landmark_point(landmarks, 27))
        ratio_down = calculate_distance(get_landmark_point(landmarks, 23), get_landmark_point(landmarks, 43)) / \
                     calculate_distance(get_landmark_point(landmarks, 43), get_landmark_point(landmarks, 47))
        


        print("Outside: " + "Ratio up: " + str(ratio_up) + " - Ratio down: " + str(ratio_down) + "\n")
        if ratio_down < 3 and ratio_up < 1.5 : 
            print("Look down" + "Ratio up: " + str(ratio_up) + " - Ratio down: " + str(ratio_down) + "\n")
            cv2.putText(frame, 'Turn Down', (255, 390), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
            pyautogui.move(0, current_speed, duration=pyautogui.MINIMUM_DURATION)
        elif ratio_up >= 1.5:
            print("Look up" + "Ratio up: " + str(ratio_up) + " - Ratio down: " + str(ratio_down) + "\n")
            cv2.putText(frame, 'Turn Up', (255, 90), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
            pyautogui.move(0, -current_speed, duration=pyautogui.MINIMUM_DURATION)
        else:
            print("No ")
            print("Normal: " + "Ratio up: " + str(ratio_up) + " - Ratio down: " + str(ratio_down) + "\n")
            cv2.putText(frame, 'Straight look', (255, 110), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

            

while True:
    loop_time_count += 1
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_detector(frame)

    for face in detected_faces:
        facial_landmarks = shape_predictor(gray_frame, face)

        draw_facial_landmarks(facial_landmarks, frame)
        handle_eye_aspect_ratio(facial_landmarks, frame)
        cursor_speed = handle_mouth_movement(facial_landmarks, frame, cursor_speed)
        handle_face_orientation(facial_landmarks, frame, cursor_speed)

        # Display time
        cv2.putText(frame, 'Time: ' + str(round(loop_time_count / 30, 2)), (10, 210), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0))

    # Display camera feed
    cv2.imshow('App', frame)
    if cv2.waitKey(1) == 27:  # 27 is the ESC key
        break