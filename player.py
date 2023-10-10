import numpy as np
import cv2
from game_interactions import game_functions as gf

class Player:
    def __init__(self, confidence = True) -> None:
        self.hp = 37
        self.ap = 37
        self.paused = True
        self.enemies = False
        self.direction = False
        self.confident = confidence
        self.look_value = 0
    
    # detect health to know when to heal
    def health_detection(self,h_frame):
        # 37 total health bars
        current_health = 0
        # itterate through possible health bar ticks
        for i in list(range(0,37)):
            color = h_frame[6,4+int(i*8.973)]
            if color[0] in list(range(173,177)) and color[1] in list(range(170,175)) and color[2] in list(range(173,177)):
                current_health += 1
                pass
        self.hp = current_health

    # detect AP for when can use VATs
    def ap_detection(self,ap_frame):
        # 37 total ap bars
        current_ap = 0
        # itterate through possible ap bar ticks
        for i in list(reversed(list(range(0,37)))):
            color = ap_frame[6,4+int(i*8.973)]
            if color[0] in list(range(173,177)) and color[1] in list(range(170,175)) and color[2] in list(range(173,177)):
                current_ap += 1
                pass
        self.ap = current_ap

    # detect which direction to point, look there and start walking
    def compass_detection(self,c_frame):
        # itterate through compass
        for i in list(range(329)):
            color = c_frame[50,i]
            if color[0] == 255 and color[1] == 255 and color[2] == 255:
                if not (i <= 175 and i >= 135):
                    self.look_value = -1*int(abs(i-165)/10) if i < 165 else 1*int(abs(i-165)/10)
                    self.direction = False
                else:
                    self.direction = True
                return None

    # main function where actions are decided based on factors provided through computer vision
    def exist(self):
        if self.confident:
            if self.hp < 10:
                gf().heal()
        if self.direction:
            gf().forward()
        else:
            gf().look_x(self.look_value)
            gf().stop()
        pass

    def detect_y_axis(self,frame):
        ground_color_lower = np.array([60, 60, 60])  # Lower threshold for ground color (in BGR)
        ground_color_upper = np.array([100, 100, 100])

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Threshold the frame to find the ground
        _, thresholded = cv2.threshold(gray_frame, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours in the thresholded frame
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the contour with the largest area (assuming it's the ground)
        max_contour = max(contours, key=cv2.contourArea)
        
        # Get the Y-coordinate of the centroid of the largest contour
        M = cv2.moments(max_contour)
        if M["m00"] != 0:
            ground_y = int(M["m01"] / M["m00"])
            return ground_y
        else:
            return None