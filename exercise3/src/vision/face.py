#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import math
import cv2
import os

pwd = os.path.dirname(os.path.realpath(__file__))
face_cascade = cv2.CascadeClassifier(os.path.join(pwd, 'haarcascade_frontalface_default.xml'))
body_cascade = cv2.CascadeClassifier(os.path.join(pwd, 'haarcascade_fullbody.xml'))
font = cv2.FONT_HERSHEY_SIMPLEX

#Field of view degrees
fov = 60
#face width cm
fw = 15
#body height cm
bh = 175
#focal length pixelcm
fl = 650

cap = cv2.VideoCapture(1)

def open_cam():
    if not cap.isOpened():
	    cap.open(0)


def close_cam():
    cap.release()
    cv2.destroyAllWindows()

def to_radians(angle):
    '''converts bens format into one used by everything else'''
    actual_degrees = 90.0 - angle
    return math.radians(actual_degrees)


def get_faces():
    '''function takes the current frame and returns all of the
     faces currently in it in two separate arrays'''
    distances = []
    angles = []

    ret, frame = cap.read()
    if not ret:
        open_cam()
        return [], []

    cv2.namedWindow('video', 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        dist = (fw * fl) / w
        dist /= 100.0 # distance in metres
        distances.append(dist)
        cv2.putText(frame, str(dist) + 'cm', (x, y), font, 1, (0, 0, 255), 2)
        angle = to_radians(((x + (w / 2.0)) / cap.get(3)) * fov - fov / 2)
        angles.append(angle)
        cv2.putText(frame, str(angle) + 'degrees', (x, y + 20), font, 0.5, (0, 0, 255), 1)

    cv2.imshow('video',frame)
    cv2.waitKey(1)
    '''bodies = body_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        dist = (bh * fl) / h
        cv2.putText(frame, str(dist) + 'cm', (x, y), font, 1, (0, 0, 255), 2)
        angle = ((x + (w / 2.0)) / cap.get(3)) * fov - fov / 2
        cv2.putText(frame, str(angle) + 'degrees', (x, y + 20), font, 0.5, (0, 0, 255), 1)
    '''

    return distances, angles
