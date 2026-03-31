# drowsiness_yawn.py
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import playsound
import os

def sound_alarm(path):
    global alarm_status, alarm_status2, saying
    while alarm_status:
        playsound.playsound(path)
    if alarm_status2:
        saying = True
        playsound.playsound(path)
        saying = False

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    return ear, leftEye, rightEye

def lip_distance(shape):
    top_lip = np.concatenate((shape[50:53], shape[61:64]))
    low_lip = np.concatenate((shape[56:59], shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    return abs(top_mean[1] - low_mean[1])

# ====================== Argument Parser ======================
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
ap.add_argument("-a", "--alarm", type=str, default="Alert.wav",
                help="path to alarm .WAV file")
args = vars(ap.parse_args())

# ====================== Constants ======================
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20

alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0
NO_FACE_COUNTER = 0

print("-> Loading the predictor and detector...")
if not os.path.isfile("haarcascade_frontalface_default.xml"):
    raise FileNotFoundError("Could not find haarcascade_frontalface_default.xml in the current directory")
if not os.path.isfile("shape_predictor_68_face_landmarks.dat"):
    raise FileNotFoundError("Could not find shape_predictor_68_face_landmarks.dat in the current directory")

cascade_path = "haarcascade_frontalface_default.xml"
shape_path = "shape_predictor_68_face_landmarks.dat"

detector = cv2.CascadeClassifier(cascade_path)
if detector.empty():
    raise ValueError(f"Failed to load cascade classifier from '{cascade_path}'")

predictor = dlib.shape_predictor(shape_path)

print("-> Starting Video Stream...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)
if not hasattr(vs, 'stream') or not vs.stream.isOpened():
    raise RuntimeError(f"Unable to open webcam at index {args['webcam']}. Check the index and webcam availability.")

while True:
    frame = vs.read()
    if frame is None:
        continue

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = np.ascontiguousarray(gray, dtype=np.uint8)

    # Convert to dlib image type explicitly (ensures compatibility across dlib versions)
    dlib_gray = gray

    # Face detection using Haar Cascade
    rects = detector.detectMultiScale(gray,
                                      scaleFactor=1.1,
                                      minNeighbors=5,
                                      minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
 
        shape = predictor(dlib_gray, rect)
        shape = face_utils.shape_to_np(shape)

        ear, leftEye, rightEye = final_ear(shape)
        distance = lip_distance(shape)

        # Draw contours
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        # ====================== Drowsiness Detection ======================
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not alarm_status:
                    alarm_status = True
                    if args["alarm"]:
                        t = Thread(target=sound_alarm, args=(args["alarm"],))
                        t.daemon = True
                        t.start()

                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            alarm_status = False

        # ====================== Yawn Detection ======================
        if distance > YAWN_THRESH:
            cv2.putText(frame, "Yawn Alert", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not alarm_status2 and not saying:
                alarm_status2 = True
                if args["alarm"]:
                    t = Thread(target=sound_alarm, args=(args["alarm"],))
                    t.daemon = True
                    t.start()
        else:
            alarm_status2 = False

        # Display values
        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"YAWN: {distance:.2f}", (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # ====================== No Face Alert ======================
    if len(rects) == 0:
        cv2.putText(frame, "NO FACE DETECTED!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        NO_FACE_COUNTER += 1
        if NO_FACE_COUNTER >= 30:  # only alert after 30 frames (~1 sec)
            if not alarm_status:
                alarm_status = True
                if args["alarm"]:
                    t = Thread(target=sound_alarm, args=(args["alarm"],))
                    t.daemon = True
                    t.start()
    else:
        
        NO_FACE_COUNTER = 0
        alarm_status = False
    cv2.imshow("Drowsiness & Yawn Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()