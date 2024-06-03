import cv2
import dlib
import time
from scipy.spatial import distance as dist
from imutils import face_utils
from playsound import playsound
import multiprocessing
import pygame
import sys


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


EYE_AR_THRESH = 0.23 # 본인이 눈 크다 싶으면 키우셈
EYE_AR_CONSEC_FRAMES = 20 # 덜 깐깐하게 하고 싶으면 키우셈


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

vs = cv2.VideoCapture(0)
alarm_on = False
CLOSE_COUNTER = 0
OPEN_COUNTER = 0

pygame.init()
pygame.mixer.init()
pygame.mixer.music.load('siren.mp3')

while True:
    ret, frame = vs.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        
        ear = (leftEAR + rightEAR) / 2.0
        
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        # print(leftEAR,rightEAR)
        if ear < EYE_AR_THRESH:
            CLOSE_COUNTER += 1
            OPEN_COUNTER = 0
            if CLOSE_COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not alarm_on:
                    alarm_on = True
                    pygame.mixer.music.play()
                    print("눈 떠!!!")
                    
        else:
            CLOSE_COUNTER = 0
            OPEN_COUNTER += 1
            if alarm_on and OPEN_COUNTER == 2:
                print("오 다시 떴네")
                pygame.mixer.music.stop()
                alarm_on = False
            
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()
