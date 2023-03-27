# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 22:00:32 2023
https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
@author: clin4
"""

import cv2
import mediapipe as mp
import time
import numpy as np

class hand_detector():
    def __init__(self,mode=False, max_hands = 2, complexity = 1, detect_confidence=.5, track_confidence=.5):
        self.mode = mode
        self.max_hands = max_hands
        self.complexity = complexity
        self.detect_conf = detect_confidence
        self.track_conf = track_confidence
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, self.complexity, 
                                        self.detect_conf, self.track_conf)
        self.mpDraw = mp.solutions.drawing_utils
        
    def find_hands(self,img,draw=True):
        #convert
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        self.results = self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks and draw:
            for detection in self.results.multi_hand_landmarks:
                #detection iterable object, ID and (x,y,z) coordinate, decimals representing ratio of image
                # landmark_id=0, base of hand
                self.mpDraw.draw_landmarks(img, detection, self.mpHands.HAND_CONNECTIONS)
        return img
        
    def find_position(self, img, landmark_list, img_shape, hand_number=0, draw=True):
        found_landmarks = []
        
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_number]
            for landmark_id, landmark in enumerate(my_hand.landmark):
                #shape = h, w, c
                cx, cy = int(landmark.x*img_shape[1]), int(landmark.y*img_shape[0])
                if landmark_id in landmark_list:
                    found_landmarks.append([landmark_id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 10, (255,0,255), cv2.FILLED)
        
        return found_landmarks
        
def touching_landmarks(img, landmarks, threshold=15):
    
    
    for i, land1 in enumerate(landmarks[:-1]):
        for land2 in landmarks[i+1:]:
            if threshold>np.sqrt((land1[1]-land2[1])**2+(land1[2]-land2[2])**2):
                cv2.putText(img=img, text=str('touching tips'), org = (400,500), fontFace = cv2.FONT_HERSHEY_PLAIN, 
                            fontScale=1, color=(255,0,255), thickness=1)
                print('touching',(land1[0],land2[0]))
                return img    
            


def main():
    # vidocapture object to read frames
    # 0 is camera to pick
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    detector = hand_detector()
    
    previous_time = 0
    current_time = 0
    
    draw = True
    
    while True:
        #read in frame
        success, img = cap.read()
        img = detector.find_hands(img)
        
        #display fps
        current_time = time.time()
        fps = 1/(current_time-previous_time)
        previous_time = current_time
        cv2.putText(img=img, text=str(int(fps)), org = (10,20), fontFace = cv2.FONT_HERSHEY_PLAIN, 
                    fontScale=1, color=(255,0,255), thickness=1)
        
        fingertip_landmark_list = detector.find_position(img,[4,8,12,16,20],img.shape,)
        
        if len(fingertip_landmark_list)>0:
            touching_landmarks(img,fingertip_landmark_list)
        
        if not success:
            print("cant read frame, exiting")
            break
                
        cv2.imshow('Cam Feed', img)
            
        #pressing 'q' will exit 
        if cv2.waitKey(1) == ord('q'):
            break

    # end stream
    cap.release()
    cv2.destroyAllWindows()


if __name__=='__main__':
    main()