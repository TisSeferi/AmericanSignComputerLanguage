# AmericanSignComputerLanguage

American Sign Language recognizer that utilizes time series recognizer JackKnife in tandem with OpenCV and MediaPipe. Currently the most stable build.
The current application allows you to record MP4s to use as templates however doesn't load the videos the program needs to be re-run. All you need to do is to hold your hand out onto the camera and let the buffer fill up (should take about 3 seconds). You can also run templates and test video/npy in the run text box. As of now, saving and recording does not work as intended however real-time recognition works.

DataForGui.py is the class you need to run for project.

Current ASL Gestures are:

Forgot,
Thank you,
Like,
No,
Don't


Might need to install these libraries

import cv2

import mediapipe as mp

import numpy as np

import os
