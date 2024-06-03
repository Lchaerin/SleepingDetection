import cv2
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
import pygame
from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.core.audio import SoundLoader

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

EYE_AR_THRESH = 0.23  # 본인이 눈 크다 싶으면 키우셈
EYE_AR_CONSEC_FRAMES = 20  # 덜 깐깐하게 하고 싶으면 키우셈

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

class CamApp(App):

    def build(self):
        fontName = '../NanumSquareR.otf'
        self.img = Image()
        self.message_label = Label(text="", font_name=fontName)

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.img)
        layout.add_widget(self.message_label)

        self.capture = cv2.VideoCapture(0)
        self.alarm_on = False
        self.CLOSE_COUNTER = 0
        self.OPEN_COUNTER = 0
        self.sound = SoundLoader.load('siren.mp3')

        Clock.schedule_interval(self.update, 1.0 / 30.0)
        return layout

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

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

            if ear < EYE_AR_THRESH:
                self.CLOSE_COUNTER += 1
                self.OPEN_COUNTER = 0
                if self.CLOSE_COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if not self.alarm_on:
                        self.alarm_on = True
                        if self.sound:
                            self.sound.play()
                        self.message_label.text = "눈 떠!!!"
            else:
                self.CLOSE_COUNTER = 0
                self.OPEN_COUNTER += 1
                if self.alarm_on and self.OPEN_COUNTER == 2:
                    self.message_label.text = "오 다시 떴네"
                    if self.sound:
                        self.sound.stop()
                    self.alarm_on = False

        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.img.texture = image_texture

if __name__ == '__main__':
    CamApp().run()
