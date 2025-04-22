from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import numpy as np
import geocoder
import time
from twilio.rest import Client
import pygame

TWILIO_SID = "ACac605f28e0d1000d11ba2921bc5ed4e8"
TWILIO_AUTH_TOKEN = "d1265c8f16b8b668b0406f228550eb17"
TWILIO_PHONE = "+17043875241"  # Your Twilio phone number
YOUR_PHONE_NUMBER = "+919347339722"  # Your testing phone number

# Initialize pygame for sound alerts
pygame.mixer.init()

# Load alert sounds
alert_sound1 = "alert1.mp3"
alert_sound2 = "alert2.mp3"
alert_sound3 = "alert3.mp3"

def play_sound(file_path):
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

# Eye Aspect Ratio (EAR) function
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Thresholds
eye_threshold = 0.25
consecutive_frames = 20
alert_reset_time = 30  # Seconds before sending another alert

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
cap = cv2.VideoCapture(1)

count = 0
detection_count = 0
last_alert_time = 0  # Track last alert time

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't capture frame")
        break
    
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        left_eye_asp_ratio = eye_aspect_ratio(leftEye)
        right_eye_asp_ratio = eye_aspect_ratio(rightEye)

        eye_asp_ratio = (left_eye_asp_ratio + right_eye_asp_ratio) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if eye_asp_ratio < eye_threshold:
            count += 1
            if count >= consecutive_frames:
                detection_count += 1
                print(f"Drowsiness detected {detection_count} times")

                cv2.putText(frame, "*****ALERT!*****", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "*****ALERT!*****", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Play different alert sounds based on detection count
                if detection_count == 2:
                    play_sound(alert_sound1)
                elif detection_count == 7:
                    play_sound(alert_sound1)
                    play_sound(alert_sound2)
                elif detection_count == 15:
                    play_sound(alert_sound1)
                    play_sound(alert_sound3)

                # Send SMS only if the last alert was more than 'alert_reset_time' seconds ago
                current_time = time.time()
                if current_time - last_alert_time > alert_reset_time:
                    g = geocoder.ip("me")
                    latitude, longitude = g.latlng if g.latlng else ("Unknown", "Unknown")
                    maps_link = f"https://www.google.com/maps/search/?api=1&query={latitude},{longitude}"

                    client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
                    sms_message = f"🚨 Drowsy driving detected! Live location: {maps_link}"

                    sms = client.messages.create(
                        body=sms_message,
                        from_=TWILIO_PHONE,
                        to=YOUR_PHONE_NUMBER
                    )
                    print(f"SMS Alert Sent: {sms.sid}")
                    last_alert_time = current_time

                count = 0  # Reset frame count after detection
        else:
            count = 0  # Reset if eyes are open

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()



